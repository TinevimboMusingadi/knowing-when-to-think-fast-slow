"""Replace one owned stalled evaluation with budget-limited real RL on its TPU."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--supervisor",type=int,required=True);p.add_argument("--evaluation",type=int,required=True)
    args=p.parse_args()
    if not re.fullmatch(r"\d{8}-\d{6}",args.run):raise ValueError("invalid experiment ID")
    root=Path("runs")/args.run;name="kws-"+args.run
    identity=["--project=serious-unison-452122-i2","--zone=us-west4-a"]
    label=subprocess.check_output(["gcloud","compute","tpus","tpu-vm","describe",name,*identity,"--format=value(labels.kws_run)"],text=True).strip()
    if label!=args.run:raise ValueError("resource ownership mismatch")
    supervisor=Path(f"/proc/{args.supervisor}/cmdline").read_bytes().replace(b"\0",b" ").decode()
    evaluation=Path(f"/proc/{args.evaluation}/cmdline").read_bytes().replace(b"\0",b" ").decode()
    parent=subprocess.check_output(["ps","-o","ppid=","-p",str(args.evaluation)],text=True).strip()
    if "/startup-script" not in supervisor or "timeout 900 python -m switching.evaluate" not in evaluation or str(root) not in evaluation or int(parent)!=args.supervisor:
        raise ValueError("process ownership mismatch; no process was signaled")
    # Pause the old supervisor before its failed evaluation can trigger teardown.
    os.kill(args.supervisor,signal.SIGSTOP)
    os.kill(args.evaluation,signal.SIGTERM)
    time.sleep(2)
    os.kill(args.supervisor,signal.SIGKILL)
    prefix=f"gs://keeper-file-storage/knowing-when-to-switch/{args.run}"
    from switching.storage import Budget
    config=json.loads(Path("configs/experiment.json").read_text())
    metrics=[json.loads(line) for line in (root/"metrics.jsonl").read_text().splitlines()]
    evaluation_started=max(row["timestamp"] for row in metrics if row["phase"]=="sft")
    Budget(root/"budget.json",4.8,"evaluation",config["budget"],start=evaluation_started).finish()
    (root/"handoff.json").write_text(json.dumps({"interrupted_sft_evaluation":True,"reason":"No first rollout result after several minutes; replacing dynamic decoding with static KV cache.","time":time.time()},indent=2))
    try:
        subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],check=True)
        sft=root/"checkpoints"/json.loads((root/"checkpoints/best-sft-rank0.json").read_text())["directory"]
        started=time.time()
        seconds=subprocess.check_output([sys.executable,"scripts/stage_limit.py","--budget",str(root/"budget.json"),"--phase","rl","--rate","4.8","--started",str(started)],text=True).strip()
        command=["timeout","--signal=TERM","--kill-after=120",seconds,sys.executable,"-m","switching.train","--phase","rl","--resume",str(sft),"--output",str(root),"--hourly-rate","4.8","--gcs",prefix+"/checkpoints","--started",str(started)]
        print("REAL_RL_LAUNCH",json.dumps(command),flush=True)
        result=subprocess.run(command)
        (root/"rl-exit.json").write_text(json.dumps({"returncode":result.returncode,"finished":time.time()}))
        pointer=root/"checkpoints/best-rl-rank0.json"
        if pointer.exists():
            rl=root/"checkpoints"/json.loads(pointer.read_text())["directory"]
            seconds=subprocess.check_output([sys.executable,"scripts/stage_limit.py","--budget",str(root/"budget.json"),"--phase","evaluation","--rate","4.8","--started",str(time.time())],text=True).strip()
            subprocess.run(["timeout","--signal=TERM","--kill-after=120",seconds,sys.executable,"-m","switching.compare","--sft",str(sft),"--rl",str(rl),"--output",str(root/"comparisons"),"--budget-path",str(root/"budget.json"),"--hourly-rate","4.8"])
    finally:
        subprocess.run(["gcloud","storage","cp","-r",str(root),prefix+"/artifacts/"])
        subprocess.run(["gcloud","storage","cp","/opt/kws/startup.log",prefix+"/startup.log"])
        current=subprocess.check_output(["gcloud","compute","tpus","tpu-vm","describe",name,*identity,"--format=value(labels.kws_run)"],text=True).strip()
        if current==args.run:subprocess.run(["gcloud","compute","tpus","tpu-vm","delete",name,*identity,"--quiet"])

if __name__=="__main__":main()
