"""Actual SFT and clipped group-relative policy optimization on CPU or replicated XLA."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import torch
from .batching import pack,supervised_items
from .model import SwitchModel
from .optim import DeviceAdamW
from .runtime import action_logps,rollout_group,synchronize
from .storage import Budget,Checkpoints,checksum

def grpo_loss(new,old,reference,advantage,epsilon=.2,beta=.02):
    ratio=(new-old).exp()
    objective=torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage)
    delta=reference-new
    kl=delta.exp()-delta-1
    return (-objective+beta*kl).mean()

def advantages(rewards):
    values=torch.tensor(rewards,dtype=torch.float32)
    return (values-values.mean())/(values.std(unbiased=False)+1e-6)

def memory_headroom(memory):
    """Support both legacy XRT and current PJRT memory counters; fail closed."""
    if "bytes_limit" in memory:
        total=int(memory["bytes_limit"])
        used=int(memory.get("peak_bytes_used",memory["bytes_used"]))
        return max(0,total-used)/total if total else 0
    if "kb_total" in memory and "kb_free" in memory:
        return int(memory["kb_free"])/int(memory["kb_total"])
    raise ValueError("unrecognized device memory counters")

def is_memory_exhaustion(error):
    text=str(error).lower()
    return any(term in text for term in ("out of memory","resource_exhausted","memory space hbm"))

def project_training_seconds(warm_times,remaining_steps,cold_times):
    if not warm_times:raise ValueError("no compilation-free timing measurements")
    return max(warm_times[-5:])*1.25*remaining_steps+2*max(cold_times,default=0)

def load_rows(path):
    with open(path,encoding="utf-8") as stream: return [json.loads(line) for line in stream if line.strip()]

def metric(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as stream: stream.write(json.dumps(record)+"\n")

def worker(index,args):
    config=json.loads(Path(args.config).read_text()); random.seed(config["seed"]); torch.manual_seed(config["seed"])
    xm=None; rank=0; world=1
    if args.device=="tpu":
        import torch_xla
        import torch_xla.runtime as xr
        import torch_xla.core.xla_model as xm
        xr.initialize_cache(str(Path(args.output)/f"compile-{index}"))
        device=torch_xla.device(); rank=xr.global_ordinal(); world=xr.world_size()
        torch_xla.manual_seed(config["seed"])
    else: device=torch.device("cpu")
    model=SwitchModel.load(config["model"],config["lora_rank"],config["lora_alpha"],torch.bfloat16 if xm else torch.float32).to(device)
    if xm:
        model.head.to(torch.bfloat16)
    optimizer=(DeviceAdamW if xm else torch.optim.AdamW)([p for p in model.parameters() if p.requires_grad],lr=config["rl_learning_rate"] if args.phase=="rl" else config["learning_rate"],**({"warmup_steps":10} if xm else {}))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1,(step+1)/10))
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    checkpoints=Checkpoints(root/"checkpoints",args.gcs)
    data_hash=checksum(args.data)
    progress={"step":0,"cursor":0,"phase":args.phase,"model":config["model"],"world_size":world,"data_sha256":data_hash,"validation_sha256":checksum(args.validation)}
    progress["training_config"]=config
    if rank==0:(root/"config.json").write_text(json.dumps(config,indent=2))
    if args.resume:
        restored=checkpoints.load(args.resume.replace("{rank}",str(rank)),model,optimizer if args.restore_optimizer else None,scheduler if args.restore_optimizer else None)
        if args.restore_optimizer:
            if restored["world_size"]!=world or restored["phase"]!=args.phase: raise ValueError("resume world size/phase mismatch")
            if restored.get("data_sha256",data_hash)!=data_hash:raise ValueError("resume dataset checksum mismatch")
            progress=restored
            progress["data_sha256"]=data_hash;progress["validation_sha256"]=checksum(args.validation)
            progress["training_config"]=config
    if args.phase=="rl" and not args.resume: raise ValueError("RL requires an SFT checkpoint")
    budget=Budget(root/"budget.json",args.hourly_rate,args.phase,config["budget"],start=args.started,prior_spend=config.get("prior_spend_upper_bound",0))
    rows=load_rows(args.data); val=load_rows(args.validation)
    best=float("inf"); start=time.perf_counter();reference=None
    def update(loss):
        loss.backward()
        synchronize(device)
    def optimizer_step():
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        norm_value=float(norm.item());finite=math.isfinite(norm_value)
        if xm:finite=xm.mesh_reduce("gradient-agreement",finite,all)
        if not finite:raise RuntimeError("non-finite gradients; optimizer update refused")
        if xm: xm.optimizer_step(optimizer,barrier=True)
        else: optimizer.step()
        scheduler.step(); optimizer.zero_grad(set_to_none=True); progress["step"]+=1
        return norm_value
    def save(tag):
        checkpoints.save(tag,model,optimizer,scheduler,progress,rank,reference=reference)
    def validate():
        model.eval(); losses=[]
        if args.phase=="rl":
            results=rollout_group(model,val[rank:24:world],sample=False)
            value=-sum(r["reward"] for r in results)/max(1,len(results))
            if xm:value=xm.mesh_reduce("validation-reward",value,lambda values:sum(values)/len(values))
            return value
        generation,decision=supervised_items(model,val[:48])
        with torch.no_grad():
            for item in generation[rank::world][:8]:
                bucket=next(b for b in config["buckets"] if b>=len(item[0]))
                batch,_,_=pack([item],bucket,1,model.tokenizer.pad_token_id,device)
                losses.append(float(model.lm(**batch).loss.item()))
            for state,candidates,target in decision[rank::world][:8]:
                logits=model.decision_logits([state],[candidates]); losses.append(float(torch.nn.functional.cross_entropy(logits,torch.tensor([target],device=device)).item()))
        value=sum(losses)/max(1,len(losses))
        if xm: value=xm.mesh_reduce("validation-loss",value,lambda values:sum(values)/len(values))
        model.train(); return value
    completed=False
    def check_budget(seconds=0):
        allowed=budget.remaining()>seconds/3600*args.hourly_rate*1.15
        if xm:allowed=xm.mesh_reduce("budget-agreement",allowed,all)
        if not allowed:raise RuntimeError("stage cannot complete within the enforced spending limit")
    def compile_count():
        if not xm:return 0
        import torch_xla.debug.metrics as metrics
        data=metrics.metric_data("CompileTime")
        return data[0] if data else 0
    try:
        if args.phase=="pilot":
            generation,_=supervised_items(model,rows[:16]); bucket=2048
            feasible=[]
            for micro in (1,2,4,8):
                check_budget(); optimizer.zero_grad(set_to_none=True)
                items=[item for item in generation if len(item[0])<=bucket][:micro]
                if not items: raise RuntimeError("no pilot examples fit bucket")
                t=time.perf_counter(); batch,_,useful=pack(items,bucket,micro,model.tokenizer.pad_token_id,device)
                try:
                    loss=model.lm(**batch).loss; update(loss); optimizer_step(); synchronize(device)
                except (RuntimeError,ValueError) as exc:
                    if is_memory_exhaustion(exc):
                        optimizer.zero_grad(set_to_none=True);break
                    raise
                compile_seconds=time.perf_counter()-t
                t=time.perf_counter()
                try:
                    loss=model.lm(**batch).loss; update(loss); optimizer_step(); synchronize(device)
                except (RuntimeError,ValueError) as exc:
                    if is_memory_exhaustion(exc):
                        optimizer.zero_grad(set_to_none=True);break
                    raise
                seconds=time.perf_counter()-t
                memory=xm.get_memory_info(device) if xm else {}
                headroom=memory_headroom(memory) if xm else 1.0
                record={"microbatch":micro,"bucket":bucket,"seconds":seconds,"compile_seconds":compile_seconds,"useful_tokens_per_second":useful/seconds,"padding_fraction":1-useful/(bucket*micro),"memory":memory,"memory_headroom":headroom,"device_utilization":None,"dollars_per_step":seconds*args.hourly_rate/3600}
                feasible.append(record)
                if rank==0: metric(root/"pilot.jsonl",record)
                if headroom<.15: break
            eligible=[r for r in feasible if r["memory_headroom"]>=.15]
            if not eligible: raise RuntimeError("pilot cannot preserve required memory headroom")
            selected=max(eligible,key=lambda r:r["useful_tokens_per_second"])
            if rank==0: (root/"pilot-selection.json").write_text(json.dumps(selected,indent=2))
            save("pilot")
        elif args.phase=="sft":
            generation,decisions=supervised_items(model,rows)
            decision_groups={count:[d for d in decisions if len(d[1])==count] for count in sorted({len(d[1]) for d in decisions})}
            # Global blocks are fixed before rank slicing, so all replicas execute
            # the same number and kind of updates; final blocks are padded.
            micro=args.microbatch; block=micro*world
            accum=max(1,math.ceil(config["effective_batch"]/block))
            total_batches=math.ceil(len(generation)/block)
            model.train(); optimizer.zero_grad(set_to_none=True);step_timings=[];warm_times=[];cold_times=[];step_started=time.perf_counter();initial_step=progress["step"];compile_before=compile_count()
            for cursor in range(progress["cursor"],total_batches):
                check_budget(); items=generation[cursor*block:(cursor+1)*block]
                while len(items)<block: items.append(generation[-1])
                bucket=next(b for b in config["buckets"] if b>=max(len(item[0]) for item in items))
                local=items[rank*micro:(rank+1)*micro]
                batch,_,useful=pack(local,bucket,micro,model.tokenizer.pad_token_id,device)
                group=min(accum,total_batches-(cursor//accum)*accum)
                t=time.perf_counter(); lm_loss=model.lm(**batch).loss
                update(lm_loss*.5/group)
                selected=list(decision_groups.values())[cursor%len(decision_groups)]
                decision_batch=[selected[(cursor*block+j)%len(selected)] for j in range(rank*micro,(rank+1)*micro)]
                decision_length=max(len(model.tokenizer.encode(d[0]+"\nCandidate: "+c["text"],add_special_tokens=False)) for d in decision_batch for c in d[1])
                decision_bucket=next(b for b in config["buckets"] if b>=decision_length)
                logits=model.decision_logits([d[0] for d in decision_batch],[d[1] for d in decision_batch],bucket=decision_bucket)
                d_loss=torch.nn.functional.cross_entropy(logits,torch.tensor([d[2] for d in decision_batch],device=device))
                update(d_loss*.5/group)
                if (cursor+1)%accum==0 or cursor+1==total_batches:
                    gradient_norm=optimizer_step(); synchronize(device)
                    progress["cursor"]=cursor+1
                    lm_value=float(lm_loss.item());decision_value=float(d_loss.item())
                    compilations=compile_count()-compile_before
                    if xm:compilations=xm.mesh_reduce("compile-agreement",compilations,max)
                    duration=time.perf_counter()-step_started
                    if xm:duration=xm.mesh_reduce("duration-agreement",duration,max)
                    step_timings.append(duration)
                    (cold_times if compilations else warm_times).append(duration)
                    seconds=time.perf_counter()-t
                    finite=math.isfinite(lm_value+decision_value)
                    if xm:finite=xm.mesh_reduce("loss-agreement",finite,all)
                    if not finite: raise RuntimeError("non-finite loss")
                    if rank==0: metric(root/"metrics.jsonl",{"phase":"sft","step":progress["step"],"lm_loss":lm_value,"decision_loss":decision_value,"gradient_norm":gradient_norm,"seconds_last_microbatch":seconds,"seconds_optimizer_update":duration,"new_compilations":compilations,"useful_tokens_last_microbatch":useful,"timestamp":time.time()})
                    # Count cold graphs separately; compilation is paid but is not
                    # repeated on every future update. Reserve two cold-step costs
                    # and 25% throughput margin instead of hiding those costs.
                    if progress["step"]-initial_step>=12 and len(warm_times)>=5:
                        projected=project_training_seconds(warm_times,math.ceil(total_batches/accum)-progress["step"],cold_times)
                        if rank==0:metric(root/"cost-projections.jsonl",{"step":progress["step"],"remaining_seconds":projected,"remaining_dollars":projected*args.hourly_rate/3600*1.15,"allowance_remaining":budget.remaining()})
                        check_budget(projected)
                    if progress["step"]%config["checkpoint_interval"]==0:
                        value=validate(); save("latest")
                        if value<best: best=value; save("best-sft")
                    if args.max_steps and progress["step"]>=args.max_steps: break
                    step_started=time.perf_counter();compile_before=compile_count()
            value=validate(); save("latest")
            if value<best: save("best-sft")
        else:
            reference=getattr(checkpoints,"loaded_reference",None) or model.trainable_state()
            for cursor in range(progress["cursor"],config["max_rl_steps"]):
                check_budget(); episode=rows[(cursor*world+rank)%len(rows)]
                rl_started=time.perf_counter()
                metric(root/f"rl-progress-rank{rank}.jsonl",{"event":"rollout-start","cursor":cursor,"behavior":episode["behavior"],"timestamp":time.time()})
                samples=rollout_group(model,[episode]*config["group_size"],config["max_generated_tokens"],config["max_transitions"],config["max_lookups"])
                metric(root/f"rl-progress-rank{rank}.jsonl",{"event":"rollout-complete","cursor":cursor,"rewards":[s["reward"] for s in samples],"tokens":[s["tokens"] for s in samples],"errors":[s["error"] for s in samples],"seconds":time.perf_counter()-rl_started,"timestamp":time.time()})
                adv=advantages([s["reward"] for s in samples]).to(device)
                with torch.no_grad(): old=[action_logps(model,s["trace"]).detach() for s in samples]
                synchronize(device)
                with model.reference(reference),torch.no_grad():
                    ref=[action_logps(model,s["trace"]).detach() for s in samples]
                    synchronize(device)
                model.eval()  # Deterministic likelihoods: LoRA/head dropout disabled.
                optimizer.zero_grad(set_to_none=True); total_loss=0
                for j,sample in enumerate(samples):
                    new=action_logps(model,sample["trace"])
                    loss=grpo_loss(new,old[j],ref[j],adv[j],config["clip_epsilon"],config["kl_coefficient"])/len(samples)
                    update(loss); total_loss+=float(loss.item())
                gradient_norm=optimizer_step(); progress["cursor"]=cursor+1
                if rank==0: metric(root/"metrics.jsonl",{"phase":"rl","step":progress["step"],"loss":total_loss,"gradient_norm":gradient_norm,"seconds_optimizer_update":time.perf_counter()-rl_started,"reward":sum(s["reward"] for s in samples)/len(samples),"tokens":sum(s["tokens"] for s in samples),"equal_reward_group":bool(torch.all(adv==0).item()),"timestamp":time.time()})
                if progress["step"]%config["checkpoint_interval"]==0 or progress["step"]==1:
                    value=validate(); save("latest")
                    if value<best: best=value; save("best-rl")
                if args.max_steps and progress["step"]>=args.max_steps: break
            save("latest")
        completed=True
    finally:
        # Preserve even an interrupted/budget-limited run; never save partial grads.
        optimizer.zero_grad(set_to_none=True)
        save("final" if completed else "interrupted")
        if rank==0: budget.finish()
        checkpoints.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/experiment.json"); p.add_argument("--phase",choices=["pilot","sft","rl"],required=True)
    p.add_argument("--device",choices=["cpu","tpu"],default="tpu"); p.add_argument("--data",default="data/train.jsonl"); p.add_argument("--validation",default="data/val.jsonl")
    p.add_argument("--output",required=True); p.add_argument("--hourly-rate",type=float,required=True); p.add_argument("--started",type=float); p.add_argument("--microbatch",type=int,default=1)
    p.add_argument("--resume"); p.add_argument("--restore-optimizer",action="store_true"); p.add_argument("--gcs"); p.add_argument("--max-steps",type=int,default=0)
    args=p.parse_args()
    if args.device=="tpu":
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(worker,args=(args,),start_method="spawn")
    else: worker(0,args)

if __name__=="__main__": main()
