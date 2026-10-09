#!/bin/bash
# TPU VM startup script for kwt runs: install, train, upload, and always delete itself.
# Settings come from instance metadata set by scripts/tpu.sh (kwt-* attributes).
#
# Kill switches (any one deletes this VM):
#   1. job exits (success or crash)      -> upload outputs, delete
#   2. job log silent for HANG_MIN       -> upload, delete
#   3. wall-clock deadline (budget cap)  -> upload, delete
#   4. heartbeat from the owner's PC stale for HEARTBEAT_MIN -> upload, delete
#   5. (cloud side) project budget -> Pub/Sub -> function deletes kwt=1 VMs
set -uo pipefail

md() { curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

NAME=$(md kwt-name); ZONE=$(md kwt-zone); PROJECT=$(md kwt-project)
DEADLINE=$(md kwt-deadline); GCS=$(md kwt-gcs); HEARTBEAT=$(md kwt-heartbeat || echo "")
ARGS=$(md kwt-args || echo "")
HANG_MIN=${HANG_MIN:-40}; HEARTBEAT_MIN=${HEARTBEAT_MIN:-30}

W=/opt/kwt; mkdir -p "$W"; cd "$W"
exec > >(tee -a "$W/startup.log") 2>&1
echo "kwt startup $(date -u) name=$NAME deadline=$(date -u -d @"$DEADLINE")"

upload() {
  gcloud storage cp "$W/startup.log" "$W/job.log" "$GCS/logs/" 2>/dev/null || true
  [ -d "$W/out" ] && gcloud storage cp -r "$W/out" "$GCS/" 2>/dev/null || true
  # Tunix checkpoints locally (gs:// needs gcsfs); mirror them for spot resume.
  [ -d "$W/ckpt" ] && gcloud storage rsync -r "$W/ckpt" "$GCS/ckpt" 2>/dev/null || true
}
delete_self() {
  echo "kwt: deleting VM ($1) at $(date -u)"
  echo "$1" > "$W/REASON"; gcloud storage cp "$W/REASON" "$GCS/logs/REASON" 2>/dev/null || true
  upload
  for _ in 1 2 3 4 5; do
    gcloud compute tpus tpu-vm delete "$NAME" --zone="$ZONE" --project="$PROJECT" --quiet --async && break
    sleep 20
  done
  sleep 600; shutdown -h now  # last resort if delete never went through
}

# Gate: prove this VM can delete itself before spending anything on training.
if ! gcloud compute tpus tpu-vm describe "$NAME" --zone="$ZONE" --project="$PROJECT" \
     --format='value(labels.kwt)' | grep -q 1; then
  echo "kwt: cannot see own TPU resource; refusing to run"; shutdown -h now; exit 1
fi
echo ok | gcloud storage cp - "$GCS/logs/GUARD_OK"

# Guard loop (kill switches 2-4), independent of the job process.
(
  BOOT=$(date +%s)
  while sleep 60; do
    NOW=$(date +%s)
    [ "$NOW" -ge "$DEADLINE" ] && delete_self "deadline"
    if [ -f "$W/job.log" ]; then
      AGE=$(( NOW - $(stat -c %Y "$W/job.log") ))
      [ "$AGE" -ge $(( HANG_MIN * 60 )) ] && delete_self "hang: job log silent ${AGE}s"
    fi
    if [ -n "$HEARTBEAT" ] && [ $(( NOW - BOOT )) -ge $(( HEARTBEAT_MIN * 60 )) ]; then
      BEAT=$(gcloud storage cat "$HEARTBEAT" 2>/dev/null | tr -dc 0-9)
      if [ -z "$BEAT" ] || [ $(( NOW - BEAT )) -ge $(( HEARTBEAT_MIN * 60 )) ]; then
        delete_self "heartbeat stale (last=${BEAT:-none})"
      fi
    fi
    (( NOW % 300 < 60 )) && upload  # live logs every ~5 min
  done
) &

# Job (kill switch 1): any exit path deletes the VM.
trap 'delete_self "job exited rc=${RC:-?}"' EXIT
{
  gcloud storage cp "$GCS/code.tar.gz" code.tar.gz && tar -xzf code.tar.gz
  mkdir -p "$W/ckpt" && gcloud storage rsync -r "$GCS/ckpt" "$W/ckpt" 2>/dev/null || true
  python3 -m pip install -q uv==0.12.23
  python3 -m uv python install 3.12.14
  python3 -m uv venv -q --python 3.12.14 .venv
  python3 -m uv pip sync -q --python .venv/bin/python --require-hashes requirements-tunix-tpu.lock
  export PYTHONPATH=$W:$W/.vendor/tunix HF_HOME=$W/.cache/hf JAX_COMPILATION_CACHE_DIR=$W/.cache/jax
  .venv/bin/python -c "import jax; print('devices', jax.devices())"
  # shellcheck disable=SC2086
  .venv/bin/python -m kwt.grpo --out "$W/out" \
    rl_training_config.checkpoint_root_directory="$W/ckpt" $ARGS
} > "$W/job.log" 2>&1
RC=$?
echo "kwt job rc=$RC"
