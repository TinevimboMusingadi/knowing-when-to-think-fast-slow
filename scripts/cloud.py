"""Owned-resource TPU launcher and independent deadline watchdog (standard library)."""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PROJECT="serious-unison-452122-i2"
BUCKET="keeper-file-storage"
ZONE="us-central1-a"
ACCELERATOR="v5litepod-4"
# Conservative on-demand upper bound despite requesting preemptible capacity.
# Current on-demand v5e price is $1.20/chip-hour, four chips. Spot savings are
# never assumed by the budget guard until a verified rate is supplied.
RATE_BOUND=4.80

def gcloud(*args,check=True,timeout=120):
    executable=shutil.which("gcloud.cmd") or shutil.which("gcloud")
    if not executable:raise RuntimeError("Google Cloud CLI unavailable")
    result=subprocess.run([executable,*args],capture_output=True,text=True,timeout=timeout)
    if check and result.returncode:raise RuntimeError(result.stderr.strip())
    return result

def describe(session):
    result=gcloud("compute","tpus","tpu-vm","describe",session["name"],"--project",session["project"],"--zone",session["zone"],"--format=json",check=False)
    if result.returncode:
        if "NOT_FOUND" in result.stderr or "not found" in result.stderr.lower():return None
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)

def delete_owned(session):
    resource=describe(session)
    if resource is None:return False
    if resource.get("labels",{}).get("kws_run")!=session["run_id"]:
        raise RuntimeError("refusing to delete a resource without this run's ownership label")
    gcloud("compute","tpus","tpu-vm","delete",session["name"],"--project",session["project"],"--zone",session["zone"],"--quiet")
    return True

def watchdog(path):
    session=json.loads(Path(path).read_text()); misses=0
    while time.time()<session["deadline"]:
        try:resource=describe(session)
        except (RuntimeError,subprocess.TimeoutExpired):
            time.sleep(30);continue
        if resource is None:
            misses+=1
            if misses>=3:return
        else:misses=0
        time.sleep(30)
    for attempt in range(3):
        try:
            delete_owned(session);return
        except (RuntimeError,subprocess.TimeoutExpired):time.sleep(10)
    raise RuntimeError("watchdog could not verify/delete its owned resource")

def startup(session,archive_uri):
    prefix=f"gs://{BUCKET}/knowing-when-to-switch/{session['run_id']}"
    return f'''#!/bin/bash
set -euo pipefail
mkdir -p /opt/kws
cd /opt/kws
export PJRT_DEVICE=TPU
export HF_HOME=/opt/kws/hf-cache
export TOKENIZERS_PARALLELISM=false
cleanup() {{
  gcloud storage cp -r runs/{session['run_id']} {prefix}/logs/ || true
  gcloud compute tpus tpu-vm delete {session['name']} --project={PROJECT} --zone={ZONE} --quiet || true
}}
trap cleanup EXIT
gcloud storage cp {archive_uri} payload.tar.gz
tar -xzf payload.tar.gz
python3 -m pip install uv==0.8.22
python3 -m uv python install 3.11
python3 -m uv venv --python 3.11 --seed .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 'torch_xla[tpu]==2.8.0' -f https://storage.googleapis.com/libtpu-releases/index.html
python -m pip install transformers==4.57.3 peft==0.18.0 google-cloud-storage safetensors
python -m unittest discover -s tests -v
python -c 'import torch_xla; print(torch_xla.devices())'
mkdir -p runs/{session['run_id']}
RUN=runs/{session['run_id']}
python -m switching.train --phase pilot --output "$RUN" --hourly-rate {RATE_BOUND} --started {session['created']} --gcs {prefix}/checkpoints
MICRO=$(python -c 'import json; print(json.load(open("runs/{session['run_id']}/pilot-selection.json"))["microbatch"])')
python -m switching.train --phase sft --output "$RUN" --hourly-rate {RATE_BOUND} --microbatch "$MICRO" --gcs {prefix}/checkpoints
SFT=$(python -c 'import json; print("runs/{session['run_id']}/checkpoints/"+json.load(open("runs/{session['run_id']}/checkpoints/best-sft-rank0.json"))["directory"])')
# Each evaluation is separately time-limited within the $5 combined allowance.
timeout 900 python -m switching.evaluate --checkpoint "$SFT" --data data/val.jsonl --output "$RUN/evaluation-sft" --limit 24 --budget-path "$RUN/budget.json" --hourly-rate {RATE_BOUND}
python -m switching.train --phase rl --resume "$SFT" --output "$RUN" --hourly-rate {RATE_BOUND} --gcs {prefix}/checkpoints
RL=$(python -c 'import json; print("runs/{session['run_id']}/checkpoints/"+json.load(open("runs/{session['run_id']}/checkpoints/best-rl-rank0.json"))["directory"])')
python -m switching.compare --sft "$SFT" --rl "$RL" --output "$RUN/comparisons" --budget-path "$RUN/budget.json" --hourly-rate {RATE_BOUND}
gcloud storage cp -r "$RUN" {prefix}/artifacts/
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--watchdog");parser.add_argument("--launch",action="store_true");parser.add_argument("--cleanup");parser.add_argument("--zone",choices=["us-central1-a","us-west4-a","us-west1-c"],default=ZONE)
    args=parser.parse_args()
    if args.watchdog:return watchdog(args.watchdog)
    if args.cleanup:return delete_owned(json.loads(Path(args.cleanup).read_text()))
    if not args.launch:parser.error("choose --launch, --watchdog, or --cleanup")
    globals()["ZONE"]=args.zone
    root=Path(__file__).resolve().parents[1]
    # Always validate the exact sanitized payload before uploading it.
    sys.path.insert(0,str(root/"scripts"))
    from preflight import audit
    audit(root)
    run_id=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    created=time.time()
    session={"run_id":run_id,"name":f"kws-{run_id}","project":PROJECT,"zone":ZONE,"accelerator":ACCELERATOR,"created":created,"deadline":created+(45/(RATE_BOUND*1.15))*3600,"rate_upper_bound":RATE_BOUND,"state":"prepared"}
    folder=root/"runs"/run_id;folder.mkdir(parents=True);session_path=folder/"cloud-session.json"
    session_path.write_text(json.dumps(session,indent=2))
    payload=folder/"payload.tar.gz"
    allowed=["switching","tests","configs","pyproject.toml","README.md"]
    with tarfile.open(payload,"w:gz") as archive:
        for name in allowed:archive.add(root/name,arcname=name,filter=lambda info:None if "__pycache__" in info.name else info)
        for name in ("train.jsonl","val.jsonl","test.jsonl","manifest.json","GSM8K_LICENSE"):archive.add(root/"data"/name,arcname=f"data/{name}")
    uri=f"gs://{BUCKET}/knowing-when-to-switch/{run_id}/payload.tar.gz"
    gcloud("storage","cp",str(payload),uri)
    script=folder/"startup.sh";script.write_text(startup(session,uri),encoding="utf-8",newline="\n")
    try:
        gcloud("compute","tpus","tpu-vm","create",session["name"],"--project",PROJECT,"--zone",ZONE,"--accelerator-type",ACCELERATOR,"--version","v2-alpha-tpuv5-lite","--preemptible","--labels",f"kws_run={run_id}","--scopes","https://www.googleapis.com/auth/cloud-platform","--metadata-from-file",f"startup-script={script}",timeout=600)
        session["state"]="created";session_path.write_text(json.dumps(session,indent=2))
        kwargs={"stdout":open(folder/"watchdog.log","a"),"stderr":subprocess.STDOUT}
        if os.name=="nt":kwargs["creationflags"]=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS
        subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--watchdog",str(session_path)],**kwargs)
        print(json.dumps(session,indent=2))
    except Exception as exc:
        session.update(state="failed",error=str(exc));session_path.write_text(json.dumps(session,indent=2))
        resource=describe(session)
        if resource:delete_owned(session)
        raise

if __name__=="__main__":main()
