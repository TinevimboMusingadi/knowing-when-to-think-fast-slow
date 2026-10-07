"""One cumulative ledger with protected evaluation money and owned deadlines."""
import json
import math
import time
from pathlib import Path


def validate_config(config):
    if config["backend"] != "tunix" or not config["interruptible_only"]:
        raise ValueError("recovery requires Tunix and interruptible capacity")
    cap, prior = config["hard_ceiling"], config["prior_spend_upper_bound"]
    if cap>120 or prior<62.11:raise ValueError('changing the authorized ceiling or historical bound requires reconciliation')
    if config['model']!='Qwen/Qwen3-1.7B' or (config['lora_rank'],config['lora_alpha'],config['head_width'],config['head_layers'])!=(16,32,128,2):
        raise ValueError('model/architecture changes require a revised experiment')
    values=[cap,prior,*config["budget"].values(),config["rate_upper_bound"],config["rate_margin"]]
    if any(not math.isfinite(v) or v<0 for v in values): raise ValueError("invalid spending limits")
    if prior+sum(config["budget"].values())>cap+1e-9: raise ValueError("allocations exceed cumulative cap")
    if config["max_generated_tokens"] is not None: raise ValueError("fixed output allowance is not authorized")
    if any(run in config["source_checkpoint"] for run in config["excluded_source_runs"]): raise ValueError("invalid checkpoint explicitly excluded")
    return config


class RecoveryBudget:
    def __init__(self, config, path, now=time.time):
        self.config=validate_config(config);self.path=Path(path);self.now=now
        self.ledger=json.loads(self.path.read_text()) if self.path.exists() else {"schema_version":2,"prior_upper_bound":config["prior_spend_upper_bound"],"attempts":[],"actual_billing":None}
        if self.ledger["prior_upper_bound"]!=config["prior_spend_upper_bound"]:raise ValueError("historical spending changed; reconcile explicitly")

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix(".tmp");temp.write_text(json.dumps(self.ledger,indent=2));temp.replace(self.path)

    def spend(self, attempt):
        end=attempt.get("deleted_at",self.now())
        return max(0.,end-attempt["created_at"])/3600*attempt["whole_slice_rate"]*self.config["rate_margin"]

    def stage_spend(self,stage):
        result=0.
        for attempt in self.ledger["attempts"]:
            for segment in attempt.get("segments",[{"stage":attempt["stage"],"started_at":attempt["created_at"]}]):
                if segment["stage"]==stage:
                    end=segment.get("ended_at",attempt.get("deleted_at",self.now()))
                    result+=max(0.,end-segment["started_at"])/3600*attempt["whole_slice_rate"]*self.config["rate_margin"]
        return result

    def total(self):
        return self.ledger["prior_upper_bound"]+sum(self.spend(a) for a in self.ledger["attempts"])

    def remaining(self,stage):
        contingency=self.config["budget"]["contingency"]
        evaluation=max(0.,self.config["budget"]["evaluation"]-self.stage_spend("evaluation")) if stage!="evaluation" else 0.
        return min(self.config["budget"][stage]-self.stage_spend(stage),self.config["hard_ceiling"]-self.total()-contingency-evaluation)

    def begin(self,stage,run_id,whole_slice_rate=None):
        if any("deleted_at" not in a for a in self.ledger["attempts"]):raise RuntimeError("previous owned attempt must be verified deleted")
        if self.remaining(stage)<=0:raise RuntimeError("stage allowance exhausted")
        rate=whole_slice_rate or self.config["rate_upper_bound"]
        if rate<=0 or rate>self.config["rate_upper_bound"]:raise ValueError("rate exceeds approved bound")
        created=self.now();seconds=self.remaining(stage)/(rate*self.config["rate_margin"])*3600
        # Worker stops early; independent watchdog retains a cleanup allowance.
        item={"stage":stage,"run_id":run_id,"created_at":created,"deadline":created+seconds,"worker_deadline":created+max(0,seconds-300),"whole_slice_rate":rate}
        item["segments"]=[{"stage":stage,"started_at":created}]
        self.ledger["attempts"].append(item);self.save();return item

    def advance(self,run_id,stage):
        item=next(a for a in self.ledger["attempts"] if a["run_id"]==run_id)
        if "deleted_at" in item:raise RuntimeError("cannot advance a deleted resource")
        if stage==item["stage"]:raise ValueError("stage is already active")
        if self.remaining(stage)<=0:raise RuntimeError("next stage allowance exhausted")
        changed=self.now();item["segments"][-1]["ended_at"]=changed
        item["segments"].append({"stage":stage,"started_at":changed});item["stage"]=stage
        seconds=self.remaining(stage)/(item["whole_slice_rate"]*self.config["rate_margin"])*3600
        item["deadline"]=changed+seconds;item["worker_deadline"]=changed+max(0,seconds-300)
        self.save();return item

    def finish(self,run_id,deleted_at,absence_verified):
        if not absence_verified:raise RuntimeError("cannot close cost accounting without verified resource absence")
        item=next(a for a in self.ledger["attempts"] if a["run_id"]==run_id)
        if deleted_at<item["created_at"]:raise ValueError("invalid deletion time")
        item["deleted_at"]=deleted_at;item["absence_verified"]=True;self.save()

    def project(self,stage,warmed_seconds,remaining_steps,compilation=0,validation=0,checkpoint_cleanup=300):
        if not warmed_seconds:raise ValueError("measured warmed throughput required")
        times=sorted(warmed_seconds);p90=times[min(len(times)-1,math.ceil(.9*len(times))-1)]
        seconds=p90*remaining_steps+compilation+validation+checkpoint_cleanup
        cost=seconds/3600*self.config["rate_upper_bound"]*self.config["rate_margin"]
        return {"seconds":seconds,"cost_upper_bound":cost,"stage_remaining":self.remaining(stage),"allowed":cost<=self.remaining(stage)}
