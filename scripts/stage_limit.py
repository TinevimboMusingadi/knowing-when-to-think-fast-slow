"""Independent wall-time allowance for a paid stage, using the shared ledger."""
import argparse
import json
import math
import time
from pathlib import Path

def seconds_remaining(ledger,phase,rate,started,now,limits):
    if rate<=0:raise ValueError("rate must be positive")
    stages=ledger.get("stages",{})
    elapsed=max(0,now-started)*rate*1.15/3600
    dollars=min(limits[phase]-stages.get(phase,0)-elapsed,50-sum(stages.values())-elapsed)
    return max(0,math.floor(dollars/(rate*1.15)*3600))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--budget",required=True);parser.add_argument("--phase",required=True);parser.add_argument("--rate",type=float,required=True);parser.add_argument("--started",type=float,required=True)
    args=parser.parse_args();path=Path(args.budget);ledger=json.loads(path.read_text()) if path.exists() else {}
    limits=json.loads(Path("configs/experiment.json").read_text())["budget"]
    seconds=seconds_remaining(ledger,args.phase,args.rate,args.started,time.time(),limits)
    if seconds<1:raise SystemExit("stage allowance exhausted")
    print(seconds)
