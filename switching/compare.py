"""Budgeted, paired baseline comparisons with validation-only threshold selection."""
import argparse
import json
import re
from pathlib import Path
import torch
from .evaluate import summary
from .model import SwitchModel
from .protocol import equivalent
from .runtime import rollout_group,synchronize
from .storage import Budget,Checkpoints
from .train import load_rows

def choose_threshold(fast,slow):
    if [r["id"] for r in fast]!=[r["id"] for r in slow]:raise ValueError("validation records must be paired")
    candidates=[]
    slow_accuracy=sum(r["correct"] and r["grounded"] for r in slow)/len(slow)
    for threshold in (.5,.6,.7,.8,.9,.95,1.01):
        selected=[]
        for f,s in zip(fast,slow):
            decision=next((a for a in f["trace"] if a["kind"]=="baseline_decision"),None)
            confidence=max(decision["probabilities"]) if decision else 0
            selected.append(f if confidence>=threshold else s)
        accuracy=sum(r["correct"] and r["grounded"] for r in selected)/len(selected)
        tokens=sum(r["tokens"] for r in selected)/len(selected)
        candidates.append((accuracy,tokens,threshold))
    acceptable=[r for r in candidates if r[0]>=slow_accuracy-.02]
    chosen=min(acceptable,key=lambda r:r[1]) if acceptable else max(candidates)
    return {"threshold":chosen[2],"validation_accuracy":chosen[0],"validation_mean_tokens":chosen[1],"selection":"minimum tokens within 2 percentage points of validation always-CoT accuracy"}

def native_base(model,episodes):
    """Original Qwen readout and native chat format; mask the three added tokens."""
    from transformers import LogitsProcessor,LogitsProcessorList
    class OriginalVocabulary(LogitsProcessor):
        def __call__(self,ids,scores):
            scores[:,-3:]=-float("inf");return scores
    records=[];device=next(model.parameters()).device
    for row in episodes:
        if row["behavior"] in {"lookup","clarification"}:continue
        messages=[{"role":"user","content":row["prompt"]+"\nGive a final answer after reasoning. Use the candidate value."}]
        text=model.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
        ids=model.tokenizer(text,return_tensors="pt").to(device)
        synchronize(device)
        import time
        start=time.perf_counter()
        with model.lm.disable_adapter(),torch.no_grad():
            generated=model.lm.generate(**ids,max_new_tokens=512,do_sample=False,logits_processor=LogitsProcessorList([OriginalVocabulary()]),pad_token_id=model.tokenizer.pad_token_id,use_cache=True)
        synchronize(device);elapsed=time.perf_counter()-start
        new=generated[0,ids["input_ids"].shape[1]:];output=model.tokenizer.decode(new,skip_special_tokens=True)
        final=output.split("</think>")[-1].strip()
        numbers=re.findall(r"-?\d+(?:\.\d+)?",final.replace(",",""))
        yesno=re.findall(r"\b(?:yes|no|unknown)\b",final,re.I)
        answer=yesno[-1].lower() if isinstance(row["answer"],str) and yesno else numbers[-1] if numbers else final
        records.append({"id":row["id"],"behavior":row["behavior"],"answer":answer,"correct":equivalent(answer,row["answer"]),"grounded":True,"error":None,"tokens":len(new),"forwards":len(new),"batch_seconds":elapsed,"batch_size":1,"trace":[{"kind":"native_text","text":output}],"modes":[],"lookups":0,"asked":False})
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument("--sft",required=True);p.add_argument("--rl",required=True);p.add_argument("--output",required=True);p.add_argument("--budget-path",required=True);p.add_argument("--hourly-rate",type=float,required=True)
    p.add_argument("--config",default="configs/experiment.json");p.add_argument("--data",default="data/test.jsonl");p.add_argument("--validation",default="data/val.jsonl");p.add_argument("--device",choices=["tpu","cpu"],default="tpu");p.add_argument("--batch-size",type=int,default=4)
    args=p.parse_args();config=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.device=="tpu":
        import torch_xla
        device=torch_xla.device();dtype=torch.bfloat16
    else:device=torch.device("cpu");dtype=torch.float32
    model=SwitchModel.load(config["model"],config["lora_rank"],config["lora_alpha"],dtype).to(device);model.head.to(dtype)
    manager=Checkpoints(out/"restore");manager.load(args.sft,model);sft=model.trainable_state();manager.load(args.rl,model);rl=model.trainable_state();manager.close()
    budget=Budget(args.budget_path,args.hourly_rate,"evaluation",config["budget"])
    rows=load_rows(args.data);val=load_rows(args.validation)[:24]
    records={name:[] for name in ("base_qwen","always_direct","always_jev","always_cot","confidence","sft_switch","rl_switch")}
    complete=False;reason=None
    def attach_gold(results,episodes):
        for r,row in zip(results,episodes):
            for action in r["trace"]:
                if action["kind"]=="decision":
                    target=row["answer"] if action["candidates"]==row["candidates"] else row["decision_stages"][0]["expected"]
                    action["gold_index"]=next(i for i,c in enumerate(action["candidates"]) if c["value"]==target)
        return results
    try:
        model.restore_trainable(sft);budget.check()
        fast=rollout_group(model,val,sample=False,policy="always_jev")
        budget.check();slow=rollout_group(model,val,sample=False,policy="always_cot")
        selected=choose_threshold(fast,slow);(out/"threshold.json").write_text(json.dumps(selected,indent=2))
        for offset in range(0,len(rows),args.batch_size):
            batch=rows[offset:offset+args.batch_size]
            for name in records:
                budget.check()
                model.restore_trainable(rl if name=="rl_switch" else sft)
                if name=="base_qwen":results=native_base(model,batch)
                else:
                    policy="learned" if name in {"sft_switch","rl_switch"} else name
                    results=attach_gold(rollout_group(model,batch,sample=False,policy=policy,threshold=selected["threshold"]),batch)
                records[name].extend(results)
                with (out/f"{name}.jsonl").open("a") as stream:
                    for record in results:stream.write(json.dumps(record)+"\n")
            (out/"progress.json").write_text(json.dumps({"completed_paired_episodes":offset+len(batch),"target":len(rows)}))
        complete=True
    except RuntimeError as exc:
        reason=str(exc)
    finally:
        budget.finish()
        summaries={name:summary(values) for name,values in records.items() if values}
        summaries.update(complete=complete,stop_reason=reason,target_episodes=len(rows),base_applicability="Native base baseline excludes clarification and lookup because no trained tool/action protocol is attached.")
        (out/"comparison.json").write_text(json.dumps(summaries,indent=2))
    if not complete:raise RuntimeError(reason or "incomplete comparison")

if __name__=="__main__":main()
