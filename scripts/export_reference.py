"""Build a deterministic PyTorch fixture; run separately from the JAX process."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from switching.conversion import portable_state
from switching.model import SwitchModel


def export(output,model):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);model.eval()
    base=model.lm.get_base_model();backbone=base.model
    frozen={"model.embed_tokens.weight":backbone.embed_tokens.original.weight,
            "model.norm.weight":backbone.norm.weight}
    for i,layer in enumerate(backbone.layers):
        prefix=f"model.layers.{i}."
        for name in ("input_layernorm","post_attention_layernorm"):
            frozen[prefix+name+".weight"]=getattr(layer,name).weight
        for name in ("q_proj","k_proj","v_proj","o_proj"):
            frozen[prefix+"self_attn."+name+".weight"]=getattr(layer.self_attn,name).base_layer.weight
        for name in ("q_norm","k_norm"):
            frozen[prefix+"self_attn."+name+".weight"]=getattr(layer.self_attn,name).weight
        for name in ("gate_proj","up_proj","down_proj"):
            frozen[prefix+"mlp."+name+".weight"]=getattr(layer.mlp,name).weight
    if not base.config.tie_word_embeddings:frozen["lm_head.weight"]=base.lm_head.original.weight
    arrays,inventory=portable_state(model.trainable_state())
    np.savez(output/"trainables.npz",**arrays)
    np.savez(output/"frozen.npz",**{n:p.detach().float().numpy() for n,p in frozen.items()})
    boundary=backbone.embed_tokens.old_size
    ids=torch.tensor([[2,boundary,3,4,5,6,7,8]])
    candidate_ids=torch.tensor([[2,3,4,5,0,0,0,0],[2,3,4,6,7,0,0,0],[2,3,4,8,9,10,0,0],[0]*8])
    lengths=torch.tensor([4,5,6,1]);valid=torch.tensor([[True,True,True,False]])
    with torch.no_grad():
        logits=model.lm(input_ids=ids,use_cache=False).logits.float()
        hidden=backbone(input_ids=ids,use_cache=False).last_hidden_state
        mask=torch.arange(8)[None,:]<lengths[:,None]
        reps=backbone(input_ids=candidate_ids,attention_mask=mask,use_cache=False).last_hidden_state[torch.arange(4),lengths-1]
        candidate_logits=model.head(reps[None,:,:],valid)
        expected={"ids":ids.numpy(),"hidden":hidden.numpy(),"logits":logits.numpy(),
                  "selected_logps":logits[0,1:4].log_softmax(-1)[torch.arange(3),torch.tensor([3,4,5])].numpy(),
                  "candidate_ids":candidate_ids.numpy(),"lengths":lengths.numpy(),"valid":valid.numpy(),
                  "candidate_logits":candidate_logits.numpy(),"candidate_probabilities":candidate_logits.softmax(-1).numpy()}
    np.savez(output/"expected.npz",**expected)
    matrix=torch.nn.Parameter(torch.tensor([[1.,-2.,.3],[.5,.8,-.4]]));rows=torch.nn.Parameter(torch.tensor([[.1,.2,.3]]))
    optimizer=torch.optim.AdamW([{"params":[matrix],"weight_decay":.01},{"params":[rows],"weight_decay":0}],lr=.001,betas=(.9,.999),eps=1e-8,foreach=False)
    optimizer_expected=[]
    for step in range(100):
        for group in optimizer.param_groups:group["lr"]=.001*min(1,(step+1)/5)
        matrix.grad=torch.tensor([[.2,-.4,.1],[.7,.8,-.1]])*(1+(step%7)/10)
        rows.grad=torch.tensor([[.3,-.2,.4]])
        norm=torch.sqrt(sum((p.grad**2).sum() for p in (matrix,rows)))
        for p in (matrix,rows):p.grad.mul_(min(1.,1./(float(norm)+1e-6)))
        optimizer.step();optimizer_expected.append(np.concatenate((matrix.detach().numpy().ravel(),rows.detach().numpy().ravel())))
    np.savez(output/"optimizer-reference.npz",updates=np.stack(optimizer_expected))
    config=base.config.to_dict()
    manifest={"schema_version":2,"model":"tiny-reference" if boundary==64 else config.get("_name_or_path"),
              "hf_config":config,"vocabulary_boundary":boundary,"rank":model.config["rank"],"alpha":model.config["alpha"],
              "inventory":inventory,"atol":1e-5,"rtol":1e-4,"full_model_probability_atol":1e-3,
              "files":{n:hashlib.sha256((output/n).read_bytes()).hexdigest() for n in ("trainables.npz","frozen.npz","expected.npz","optimizer-reference.npz")}}
    (output/"fixture.json").write_text(json.dumps(manifest,indent=2));return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--checkpoint");a=p.parse_args()
    torch.manual_seed(42)
    if a.checkpoint:
        state=torch.load(Path(a.checkpoint)/"state.pt",map_location="cpu",weights_only=False)
        model=SwitchModel.load("Qwen/Qwen3-1.7B",local_files_only=True)
        model.restore_trainable(state["trainable"])
    else:
        from transformers import Qwen3Config,Qwen3ForCausalLM
        class Tokenizer:
            pad_token_id=0
            def add_special_tokens(self,_):return 3
        cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        cfg._attn_implementation="eager"
        model=SwitchModel(Qwen3ForCausalLM(cfg),Tokenizer(),rank=2,alpha=4)
        # Nonzero B verifies the delta and its scaling, rather than only base output.
        with torch.no_grad():
            for name,param in model.named_parameters():
                if "lora_B" in name:param.normal_(0,.005)
    result=export(a.output,model);print(json.dumps({"model":result["model"],"tensors":len(result["inventory"])}))


if __name__=="__main__":main()
