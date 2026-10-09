#!/usr/bin/env bash
# Spot TPU runs for kwt, headless: no SSH or scp from Windows needed.
#
#   scripts/tpu.sh launch <run> [max_hours] [-- kwt.grpo args]   upload code, create VM, start job
#   scripts/tpu.sh heartbeat <run>     keep running while the job runs; if this PC dies, the VM dies
#   scripts/tpu.sh logs <run>          tail the job log (synced every ~5 min)
#   scripts/tpu.sh status              list every TPU VM in the project (want: empty)
#   scripts/tpu.sh delete <run>        delete now
#   scripts/tpu.sh fetch <run>         download outputs to runs/kwt/<run>
#
# Env overrides: ZONE, TYPE, RATE (USD/hour for the whole slice), NO_HEARTBEAT=1.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT=${PROJECT:-serious-unison-452122-i2}
ZONE=${ZONE:-us-west4-a}
TYPE=${TYPE:-v5litepod-1}
VERSION=${VERSION:-v2-alpha-tpuv5-lite}
RATE=${RATE:-0.50}  # spot v5e ≈ $0.494/chip-hour (docs/recovery-price-check.json)
BUCKET=gs://keeper-file-storage/knowing-when-to-switch/kwt
SPEND=runs/kwt/spend.csv

cmd=${1:-}; shift || true
run=${1:-}; [ -n "$run" ] && shift || true
gcs="$BUCKET/$run"
vm="kwt-$run"

case "$cmd" in
launch)
  hours=${1:-3}; shift || true; [ "${1:-}" = "--" ] && shift
  args="$*"
  deadline=$(awk "BEGIN{printf \"%d\", $(date +%s) + $hours*3600}")
  echo "max cost: $(awk "BEGIN{print $hours*$RATE}") USD ($TYPE, ${hours}h at \$$RATE/h)"
  tmp=$(mktemp -d)
  tar --exclude=__pycache__ -czf "$tmp/code.tar.gz" kwt configs/kwt_grpo.yaml data/recovery-v2 \
      requirements-tunix-tpu.lock .vendor/tunix/tunix .vendor/tunix/LICENSE
  ls -la "$tmp/code.tar.gz"
  gcloud storage cp "$tmp/code.tar.gz" "$gcs/code.tar.gz"
  heartbeat=""
  if [ -z "${NO_HEARTBEAT:-}" ]; then
    heartbeat="$gcs/heartbeat"; date +%s | gcloud storage cp - "$heartbeat"
  fi
  gcloud compute tpus tpu-vm create "$vm" --project="$PROJECT" --zone="$ZONE" \
    --accelerator-type="$TYPE" --version="$VERSION" --spot --labels=kwt=1 \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata="kwt-name=$vm,kwt-zone=$ZONE,kwt-project=$PROJECT,kwt-deadline=$deadline,kwt-gcs=$gcs,kwt-heartbeat=$heartbeat,kwt-args=$args" \
    --metadata-from-file=startup-script=scripts/vm_startup.sh
  mkdir -p runs/kwt
  [ -f "$SPEND" ] || echo "run,type,started_utc,max_hours,rate_usd_h,max_usd" > "$SPEND"
  echo "$run,$TYPE,$(date -u +%FT%TZ),$hours,$RATE,$(awk "BEGIN{print $hours*$RATE}")" >> "$SPEND"
  echo "waiting for the self-delete gate (GUARD_OK)..."
  for _ in $(seq 1 40); do
    if gcloud storage cat "$gcs/logs/GUARD_OK" >/dev/null 2>&1; then
      echo "GUARD_OK: VM can delete itself. Now run: scripts/tpu.sh heartbeat $run"; exit 0
    fi
    sleep 15
  done
  echo "no GUARD_OK after 10 min; deleting $vm to be safe"
  "$0" delete "$run"; exit 1
  ;;
heartbeat)
  [ -n "${NO_HEARTBEAT:-}" ] && { echo "heartbeat disabled"; exit 0; }
  echo "heartbeat for $vm every 5 min (Ctrl-C stops it; the VM then deletes itself within ~30 min)"
  while gcloud compute tpus tpu-vm describe "$vm" --zone="$ZONE" --project="$PROJECT" \
          --format='value(state)' >/dev/null 2>&1; do
    date +%s | gcloud storage cp - "$gcs/heartbeat" >/dev/null 2>&1 || true
    sleep 300
  done
  echo "$vm is gone; heartbeat stopped"
  ;;
logs)
  gcloud storage cat "$gcs/logs/REASON" 2>/dev/null && echo "(VM finished, reason above)"
  gcloud storage cat "$gcs/logs/job.log" 2>/dev/null | tail -n "${1:-60}"
  ;;
status)
  gcloud compute tpus tpu-vm list --zone="$ZONE" --project="$PROJECT" \
    --format='table(name,acceleratorType,state,createTime,labels.kwt)'
  ;;
delete)
  gcloud compute tpus tpu-vm delete "$vm" --zone="$ZONE" --project="$PROJECT" --quiet || true
  "$0" status
  ;;
fetch)
  mkdir -p "runs/kwt/$run"
  gcloud storage cp -r "$gcs/out/*" "$gcs/logs" "runs/kwt/$run/"
  ;;
*)
  sed -n 2,12p "$0"; exit 1
  ;;
esac
