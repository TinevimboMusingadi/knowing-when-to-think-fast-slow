"""Bounded, preemptible RL-only recovery from a verified SFT checkpoint."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import cloud


def startup(session, uri, checkpoint):
    run = session["run_id"]
    prefix = f"gs://{cloud.BUCKET}/knowing-when-to-switch/{run}"
    # Reuse the locked runtime setup, stopping before any supervised training.
    setup = cloud.startup(session, uri).split("python -m pip freeze >")[0]
    return setup + f'''python -m pip freeze > "$RUN/requirements-lock.txt"
python -c 'import json; from switching.storage import Checkpoints; from pathlib import Path; c=Checkpoints("$RUN/restore-scratch"); p=c.restore_gcs("{checkpoint}","$RUN/sft"); m=json.loads((p/"manifest.json").read_text()); assert m["progress"]["phase"]=="sft" and m["rank"]==0; c.close()'
SECONDS_LEFT=$(python scripts/stage_limit.py --budget "$RUN/budget.json" --phase rl --rate {cloud.RATE_BOUND} --started {session['created']})
# Reserve time for interrupted checkpoint upload and teardown.
TRAIN_SECONDS=$((SECONDS_LEFT-120))
if [ "$TRAIN_SECONDS" -lt 60 ]; then exit 2; fi
printf '%s\\n' 'Fresh optimizer; SFT policy initialization; maximum three RL updates.'
set +e
timeout --signal=TERM --kill-after=60 "$TRAIN_SECONDS" python -m switching.train --phase rl --resume "$RUN/sft" --validation data/validation30.jsonl --max-steps 3 --output "$RUN" --hourly-rate {cloud.RATE_BOUND} --started {session['created']} --gcs {prefix}/checkpoints
RESULT=$?
set -e
echo "$RESULT" > "$RUN/rl-exit-code.txt"
gcloud storage cp -r "$RUN" {prefix}/artifacts/
exit "$RESULT"
'''.replace('"$RUN/restore-scratch"', f'"runs/{run}/restore-scratch"').replace('"$RUN/sft"', f'"runs/{run}/sft"')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
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
    allowance = min(args.max_dollars, 45 - prior)
    if allowance <= 0 or args.max_dollars <= 0:
        raise RuntimeError("no compute allowance after preserving the $5 storage reserve")
    from preflight import audit
    audit(root)
    source = f"gs://{cloud.BUCKET}/knowing-when-to-switch/{args.source_run}"
    pointer = json.loads(cloud.gcloud("storage", "cat", source + f"/artifacts/{args.source_run}/checkpoints/best-sft-rank0.json").stdout)
    directory = pointer["directory"]
    if Path(directory).name != directory or not directory.startswith("step-"):
        raise ValueError("invalid checkpoint pointer")
    checkpoint = source + "/checkpoints/" + directory
    manifest = json.loads(cloud.gcloud("storage", "cat", checkpoint + "/manifest.json").stdout)
    if manifest["progress"]["phase"] != "sft" or manifest["rank"] != 0:
        raise ValueError("restart requires SFT checkpoint")
    cloud.gcloud("storage", "cat", checkpoint + "/COMPLETE")
    config = json.loads((root / "configs/experiment.json").read_text())
    if manifest["progress"]["model"] != config["model"]:
        raise ValueError("checkpoint model mismatch")
    config.update(prior_spend_upper_bound=prior, max_rl_steps=3, rl_validation_limit=6)
    config["budget"]["rl"] = allowance
    import time
    created = time.time()
    run = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    session = dict(run_id=run, name=f"kws-{run}", project=cloud.PROJECT,
                   zone=cloud.ZONE, accelerator=cloud.ACCELERATOR, created=created,
                   deadline=created + allowance / (cloud.RATE_BOUND * 1.15) * 3600,
                   rate_upper_bound=cloud.RATE_BOUND, prior_spend_upper_bound=prior,
                   state="prepared", source_sft=checkpoint, fresh_optimizer=True,
                   allowance=allowance)
    folder = root / "runs" / run
    folder.mkdir()
    session_path = folder / "cloud-session.json"
    session_path.write_text(json.dumps(session, indent=2))
    (folder / "experiment.json").write_text(json.dumps(config, indent=2))
    (folder / "budget.json").write_text(json.dumps({"stages": {"contingency": prior}, "limit": 50}))
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
