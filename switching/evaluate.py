"""Held-out evaluation; saves raw records and bootstrap accuracy intervals."""
import argparse
import collections
import json
import random
import time
from pathlib import Path
import torch
from .model import SwitchModel
from .runtime import rollout_group,synchronize
from .storage import Budget,Checkpoints
from .train import load_rows

def summary(records):
    if not records: raise ValueError("empty evaluation")
    rng=random.Random(42); n=len(records)
    correct=[int(r["correct"] and r["grounded"]) for r in records]
    boot=sorted(sum(rng.choice(correct) for _ in range(n))/n for _ in range(1000))
    by_behavior={}
    for behavior in sorted({r["behavior"] for r in records}):
        subset=[r for r in records if r["behavior"]==behavior]
        by_behavior[behavior]={"count":len(subset),"accuracy":sum(r["correct"] and r["grounded"] for r in subset)/len(subset)}
    brier=[]
    for record in records:
        for action in record["trace"]:
            if action["kind"] in {"decision","baseline_decision"} and "gold_index" in action:
                brier.append(sum((p-int(i==action["gold_index"]))**2 for i,p in enumerate(action["probabilities"])))
    acquisition={}
    for behavior,action in (("clarification","asked"),("lookup","lookups")):
        subset=[r for r in records if r["behavior"]==behavior]
        acquisition[behavior]=sum(bool(r.get(action)) and r["correct"] and r["grounded"] for r in subset)/len(subset) if subset else None
    return {"count":n,"accuracy":sum(correct)/n,"accuracy_95_bootstrap":[boot[25],boot[974]],"mean_tokens":sum(r["tokens"] for r in records)/n,"mean_forward_passes":sum(r["forwards"] for r in records)/n,"mean_transitions":sum(sum(a!=b for a,b in zip(r.get("modes",[]),r.get("modes",[])[1:])) for r in records)/n,"acquisition_success":acquisition,"seconds_per_episode_amortized":sum(r["batch_seconds"]/r["batch_size"] for r in records)/n,"brier_score":sum(brier)/len(brier) if brier else None,"by_behavior":by_behavior,"errors":dict(collections.Counter(r["error"] for r in records if r.get("error"))),"latency_note":"Batch latency divided by batch size is throughput, not individual request latency. First batch is retained separately as cold latency."}

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--data",default="data/test.jsonl");p.add_argument("--config",default="configs/experiment.json");p.add_argument("--output",required=True)
    p.add_argument("--policy",choices=["learned","always_direct","always_cot","always_jev","confidence"],default="learned");p.add_argument("--threshold",type=float);p.add_argument("--batch-size",type=int,default=4);p.add_argument("--limit",type=int,default=0);p.add_argument("--device",choices=["cpu","tpu"],default="tpu");p.add_argument("--budget-path");p.add_argument("--hourly-rate",type=float)
    args=p.parse_args();config=json.loads(Path(args.config).read_text())
    budget=Budget(args.budget_path,args.hourly_rate,"evaluation",config["budget"],hard_ceiling=config.get("hard_ceiling",50)) if args.budget_path else None
    if args.policy=="confidence" and args.threshold is None: p.error("confidence baseline requires a threshold chosen on validation only")
    if args.device=="tpu":
        import torch_xla; device=torch_xla.device(); dtype=torch.bfloat16
    else: device=torch.device("cpu");dtype=torch.float32
    model=SwitchModel.load(config["model"],config["lora_rank"],config["lora_alpha"],dtype).to(device); model.head.to(dtype)
    manager=Checkpoints(Path(args.output)/"restore");manager.load(args.checkpoint,model);manager.close()
    rows=load_rows(args.data);rows=rows[:args.limit] if args.limit else rows
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);records=[]
    for offset in range(0,len(rows),args.batch_size):
        if budget:
            try:budget.check()
            except RuntimeError:
                budget.finish();raise
        batch=rows[offset:offset+args.batch_size]
        results=rollout_group(model,batch,sample=False,policy=args.policy,threshold=args.threshold or .8)
        for row,result in zip(batch,results):
            for action in result["trace"]:
                if action["kind"] in {"decision","baseline_decision"}:
                    target=row["answer"] if action["candidates"]==row["candidates"] else row["decision_stages"][0]["expected"]
                    action["gold_index"]=next(i for i,c in enumerate(action["candidates"]) if c["value"]==target)
        records.extend(results)
        with (output/f"{args.policy}.jsonl").open("a") as stream:
            for result in results:stream.write(json.dumps(result)+"\n")
    report=summary(records);report.update(policy=args.policy,checkpoint=args.checkpoint,cold_batch_seconds=records[0]["batch_seconds"])
    (output/f"{args.policy}-summary.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if budget:budget.finish()

if __name__=="__main__":main()
