"""Bounded, preemptible RL-only recovery from a verified SFT checkpoint."""
import argparse
import datetime
import json
import math
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import cloud


def restart_allowance(prior, requested, total_cap=50, storage_reserve=5):
    if not all(math.isfinite(x) and x >= 0 for x in (prior, requested, total_cap, storage_reserve)):
        raise ValueError("invalid spending inputs")
    allowance = min(requested, total_cap - storage_reserve - prior)
    # Observed provisioning/setup + first update/validation + teardown require
    # roughly fourteen minutes. Refuse an attempt that cannot reach useful evidence.
    minimum = cloud.RATE_BOUND * 1.15 * 14 / 60
    if requested <= 0 or allowance < minimum:
        raise RuntimeError("insufficient compute allowance for setup and one measured RL update; reconcile billing or revise budget")
    return allowance


def startup(session, uri, checkpoint):
    run = session["run_id"]
    prefix = f"gs://{cloud.BUCKET}/knowing-when-to-switch/{run}"
    # Reuse the locked runtime setup, stopping before any supervised training.
    setup = cloud.startup(session, uri).split("python -m pip freeze >")[0]
    continuation = isinstance(checkpoint, list)
    checkpoints = checkpoint if continuation else [checkpoint]
    phase = "rl" if continuation else "sft"
    restore = []
    for rank, source in enumerate(checkpoints):
        destination = f"runs/{run}/restore/rank{rank}" if continuation else f"runs/{run}/sft"
        restore.append(f"python -c 'import json; from switching.storage import Checkpoints; c=Checkpoints(\"runs/{run}/restore-scratch\"); p=c.restore_gcs(\"{source}\",\"{destination}\"); m=json.loads((p/\"manifest.json\").read_text()); assert m[\"progress\"][\"phase\"]==\"{phase}\" and m[\"rank\"]=={rank}; c.close()'")
    restore_commands = "\n".join(restore)
    resume = f'"$RUN/restore/rank{{rank}}" --restore-optimizer' if continuation else '"$RUN/sft"'
    maximum = f" --max-steps {session['pilot_stop_step']}" if session.get("pilot_stop_step") else "" if continuation else " --max-steps 3"
    return setup + f'''python -m pip freeze > "$RUN/requirements-lock.txt"
{restore_commands}
python -c 'import json; from pathlib import Path; from switching.inspect_checkpoint import inspect; folders=[Path("runs/{run}/restore/rank"+str(i)) for i in range(4)] if {continuation!r} else [Path("runs/{run}/sft")]; records=[inspect(p) for p in folders]; assert all(r["finite_parameters"] and r["finite_optimizer"] and r["finite_reference"] for r in records), "invalid restored numerical state"; Path("runs/{run}/restore-audit.json").write_text(json.dumps(records,indent=2))'
SECONDS_LEFT=$(python scripts/stage_limit.py --budget "$RUN/budget.json" --phase rl --rate {cloud.RATE_BOUND} --started {session['created']})
TRAIN_SECONDS=$((SECONDS_LEFT-120))
if [ "$TRAIN_SECONDS" -lt 60 ]; then exit 2; fi
set +e
setsid timeout --signal=TERM --kill-after=60 "$TRAIN_SECONDS" python -m switching.train --phase rl --resume {resume} --validation data/validation30.jsonl{maximum} --output "$RUN" --hourly-rate {cloud.RATE_BOUND} --started {session['created']} --gcs {prefix}/checkpoints &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  if compgen -G "$RUN/worker-error-*.txt" >/dev/null; then
    kill -TERM -- "-$TRAIN_PID" 2>/dev/null || true
    sleep 15
    kill -KILL -- "-$TRAIN_PID" 2>/dev/null || true
    break
  fi
  sleep 5
done
wait "$TRAIN_PID"
RESULT=$?
set -e
echo "$RESULT" > "$RUN/rl-exit-code.txt"
gcloud storage cp -r "$RUN" {prefix}/artifacts/
exit "$RESULT"
'''



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--continue-rl", action="store_true")
    parser.add_argument("--pilot", action="store_true",help="Stop after two new RL updates and validate each")
    parser.add_argument("--total-cap", type=float, default=50)
    parser.add_argument("--source-run", default="20261006-122830")
    parser.add_argument("--max-dollars", type=float, default=1.8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path(args.source_run).name != args.source_run:
        raise ValueError("invalid source run")
    cloud.ZONE = "us-west4-a"
    prior = 0.0
    for path in (root / "runs").glob("*/cloud-session.json"):
        old = json.loads(path.read_text())
        if old["state"] == "created":
            raise RuntimeError("unreconciled prior session; verify deletion before restarting")
        prior += old.get("estimated_compute_upper_bound", 0)
    allowance = restart_allowance(prior, args.max_dollars, args.total_cap)
    from preflight import audit
    audit(root)
    source = f"gs://{cloud.BUCKET}/knowing-when-to-switch/{args.source_run}"
    config = json.loads((root / "configs/experiment.json").read_text())
    checkpoints = []; positions = []
    for rank in range(4 if args.continue_rl else 1):
        tag = "latest" if args.continue_rl else "best-sft"
        pointer = json.loads(cloud.gcloud("storage", "cat", source + f"/artifacts/{args.source_run}/checkpoints/{tag}-rank{rank}.json").stdout)
        directory = pointer["directory"]
        if Path(directory).name != directory or "\\" in directory or not directory.startswith("step-"):
            raise ValueError("invalid checkpoint pointer")
        checkpoint_uri = source + "/checkpoints/" + directory
        manifest_text = cloud.gcloud("storage", "cat", checkpoint_uri + "/manifest.json").stdout
        manifest = json.loads(manifest_text)
        marker = cloud.gcloud("storage", "cat", checkpoint_uri + "/COMPLETE").stdout.strip()
        if marker != hashlib.sha256(manifest_text.encode()).hexdigest():
            raise ValueError("checkpoint completion checksum mismatch")
        expected_phase = "rl" if args.continue_rl else "sft"
        if manifest["progress"]["phase"] != expected_phase or manifest["rank"] != rank:
            raise ValueError("checkpoint rank/phase mismatch")
        if manifest["progress"]["model"] != config["model"]:
            raise ValueError("checkpoint model mismatch")
        if args.continue_rl:
            if manifest["progress"]["world_size"] != 4:
                raise ValueError("continuation requires four-replica checkpoint")
            if manifest["progress"]["data_sha256"] != hashlib.sha256((root/"data/train.jsonl").read_bytes()).hexdigest():
                raise ValueError("continuation dataset mismatch")
            positions.append((manifest["progress"]["step"],manifest["progress"]["cursor"]))
        checkpoints.append(checkpoint_uri)
    if len(set(positions)) > 1:
        raise ValueError("replica checkpoints are not synchronized")
    checkpoint = checkpoints if args.continue_rl else checkpoints[0]
    pilot_stop=(positions[0][0] if positions else 0)+2 if args.pilot else None
    config.update(prior_spend_upper_bound=prior, hard_ceiling=args.total_cap,
                  max_rl_steps=pilot_stop or (300 if args.continue_rl else 3),
                  rl_validation_limit=4 if args.pilot else 8 if args.continue_rl else 6, rl_validation_interval=1 if args.pilot else 5)
    config["budget"]["rl"] = allowance
    import time
    created = time.time()
    run = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    session = dict(run_id=run, name=f"kws-{run}", project=cloud.PROJECT,
                   zone=cloud.ZONE, accelerator=cloud.ACCELERATOR, created=created,
                   deadline=created + allowance / (cloud.RATE_BOUND * 1.15) * 3600,
                   rate_upper_bound=cloud.RATE_BOUND, prior_spend_upper_bound=prior,
                   state="prepared", source_checkpoint=checkpoint, fresh_optimizer=not args.continue_rl,
                   hard_ceiling=args.total_cap, storage_reserve=5,
                   allowance=allowance)
    if args.pilot:session["pilot_stop_step"]=pilot_stop
    folder = root / "runs" / run
    folder.mkdir()
    session_path = folder / "cloud-session.json"
    session_path.write_text(json.dumps(session, indent=2))
    (folder / "experiment.json").write_text(json.dumps(config, indent=2))
    (folder / "budget.json").write_text(json.dumps({"stages": {"contingency": prior}, "limit": args.total_cap}))
    payload = folder / "payload.tar.gz"
    with tarfile.open(payload, "w:gz") as archive:
        for name in ("switching", "tests", "scripts", "pyproject.toml", "README.md"):
            archive.add(root / name, arcname=name, filter=lambda item: None if "__pycache__" in item.name else item)
        archive.add(folder / "experiment.json", arcname="configs/experiment.json")
        archive.add(folder / "budget.json", arcname=f"runs/{run}/budget.json")
        for name in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json", "GSM8K_LICENSE", "validation30.jsonl"):
            archive.add(root / "data" / name, arcname=f"data/{name}")
    uri = f"gs://{cloud.BUCKET}/knowing-when-to-switch/{run}/payload.tar.gz"
    script = folder / "startup.sh"
    script.write_text(startup(session, uri, checkpoint), encoding="utf-8", newline="\n")
    print(json.dumps(session, indent=2), flush=True)
    if not args.launch:
        return
    cloud.gcloud("storage", "cp", str(payload), uri)
    try:
        cloud.gcloud("compute", "tpus", "tpu-vm", "create", session["name"],
                     "--project", cloud.PROJECT, "--zone", cloud.ZONE,
                     "--accelerator-type", cloud.ACCELERATOR, "--version", "v2-alpha-tpuv5-lite",
                     "--preemptible", "--labels", f"kws_run={run}",
                     "--scopes", "https://www.googleapis.com/auth/cloud-platform",
                     "--metadata-from-file", f"startup-script={script}", timeout=600)
        session["state"] = "created"
        session_path.write_text(json.dumps(session, indent=2))
        options = {"stdout": open(folder / "watchdog.log", "a"), "stderr": subprocess.STDOUT}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
        subprocess.Popen([sys.executable, str(root / "scripts/cloud.py"), "--watchdog", str(session_path)], **options)
    except Exception as exc:
        session.update(state="failed", error=str(exc))
        session_path.write_text(json.dumps(session, indent=2))
        cloud.delete_owned(session)
        raise


if __name__ == "__main__":
    main()
