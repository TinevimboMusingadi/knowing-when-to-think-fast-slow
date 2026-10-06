"""No-download numerical regression and tiny-Qwen dual-head update probe.

Fixed synthetic actions exercise likelihood/backward paths. This is not a
sampled-policy benchmark or evidence that the 1.7B TPU failure is resolved.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from switching.model import SwitchModel
from switching.optim import DeviceAdamW
from switching.runtime import action_logps
from switching.train import grpo_loss,advantages,gradient_norm

class ProbeTokenizer:
    pad_token_id=0
    def add_special_tokens(self,config):return 3
    def encode(self,text,add_special_tokens=False):return [1+ord(c)%60 for c in text]

def probe():
    torch.manual_seed(42);torch.set_num_threads(2)
    extreme=torch.tensor([-1000.],requires_grad=True);old=extreme.detach().clone();ref=torch.zeros(1)
    delta=ref-extreme;original=(delta.exp()-delta-1).mean()
    original_gradient=torch.autograd.grad(original,extreme)[0]
    stable=grpo_loss(extreme,old,ref,0.)
    stable_gradient=torch.autograd.grad(stable,extreme)[0]
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    config._attn_implementation="eager"
    model=SwitchModel(Qwen3ForCausalLM(config),ProbeTokenizer(),rank=2,alpha=4).eval()
    reference=model.trainable_state();frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    optimizer=DeviceAdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    candidates=[{"id":"a","text":"1","value":1},{"id":"b","text":"2","value":2}]
    traces=[[{"kind":"tokens","prompt":[2,64,3],"completion":[4+i%2,63]},{"kind":"decision","state":"A value is 1.","candidates":candidates,"choice":i%2,"bucket":32}] for i in range(4)]
    adv=advantages([1.,-1.,1.,-1.]);steps=[]
    for step in range(4):
        with torch.no_grad():old_logps=[action_logps(model,t) for t in traces]
        with model.reference(reference),torch.no_grad():ref_logps=[action_logps(model,t) for t in traces]
        optimizer.zero_grad(set_to_none=True);total=0.
        for trace,previous,reference_logps,advantage in zip(traces,old_logps,ref_logps,adv):
            loss=grpo_loss(action_logps(model,trace),previous,reference_logps,advantage)/4
            if not bool(torch.isfinite(loss)):raise RuntimeError("probe loss failed")
            loss.backward();total+=float(loss.detach())
        norm=float(gradient_norm(model))
        if not math.isfinite(norm):raise RuntimeError("probe gradients failed")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step();steps.append({"step":step+1,"loss":total,"gradient_norm":norm})
    current=model.trainable_state();changed=[n for n,v in reference.items() if not torch.equal(current[n],v)]
    unchanged_base=all(torch.equal(dict(model.named_parameters())[n],v) for n,v in frozen.items())
    result={"scope":"CPU fixed-action numerical integration probe; random tiny Qwen3, not Qwen3-1.7B evaluation","seed":42,"group_size":4,"steps":steps,"original_overflow":{"finite_loss":bool(torch.isfinite(original)),"finite_gradient":bool(torch.isfinite(original_gradient).all())},"repaired_overflow":{"finite_loss":bool(torch.isfinite(stable)),"finite_gradient":bool(torch.isfinite(stable_gradient).all()),"tail_bound":20},"changed_lora_tensors":sum("lora_" in n for n in changed),"changed_head_tensors":sum(n.startswith("head.") for n in changed),"changed_mode_rows":any(n.endswith("rows") for n in changed),"base_weights_unchanged":unchanged_base,"all_trainable_finite":all(bool(torch.isfinite(v).all()) for v in current.values()),"limitations":["Synthetic fixed actions, not sampled environment rollouts.","Exponential tails are bounded, which modifies the extreme-tail objective.","The exact original failing TPU trace was not saved.","No new TPU run or 1.7B capability benchmark performed."]}
    if not (result["changed_lora_tensors"] and result["changed_head_tensors"] and unchanged_base and result["all_trainable_finite"]):raise RuntimeError("probe parameter validation failed")
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="runs/offline-repair/numerics.json");args=parser.parse_args()
    result=probe();result["recorded_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
