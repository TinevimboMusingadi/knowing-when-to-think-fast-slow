"""Checksummed, pickle-free transfer of verified PyTorch trainables to JAX."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def canonical_name(name):
    for prefix in ("lm.base_model.model.model.layers.","lm.base_model.model.model."):
        if name.startswith(prefix):return name[len(prefix):]
    return name


def portable_state(state):
    arrays={};inventory=[]
    for name,value in state.items():
        array=value.detach().float().cpu().numpy().copy()
        if not np.isfinite(array).all():raise ValueError(f"nonfinite source tensor: {name}")
        if "lora_A" in name: kind="lora_a"
        elif "lora_B" in name: kind="lora_b"
        elif name.endswith("rows"):kind="mode_rows"
        elif name.startswith("head."):kind="head"
        else:raise ValueError(f"unexpected trainable tensor: {name}")
        arrays[name]=array;inventory.append({"name":name,"shape":list(array.shape),"dtype":"float32","kind":kind})
    if sum(x["kind"]=="mode_rows" for x in inventory)!=1:raise ValueError("exactly one tied mode-row tensor required")
    a={n.replace("lora_A","lora_B") for n in arrays if "lora_A" in n};b={n for n in arrays if "lora_B" in n}
    if a!=b:raise ValueError("unpaired adapters")
    return arrays,inventory


def project_adapter(a,b,input_shape,output_shape):
    """PEFT linear: x @ A.T @ B.T; never confuse head axes and scaling."""
    if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=b.shape[1]:raise ValueError("invalid LoRA matrices")
    if a.shape[1]!=int(np.prod(input_shape)) or b.shape[0]!=int(np.prod(output_shape)):raise ValueError("projection shape mismatch")
    return a.T.reshape(*input_shape,a.shape[0]),b.T.reshape(a.shape[0],*output_shape)


def verify_source(checkpoint):
    checkpoint=Path(checkpoint)
    manifest_path=checkpoint/"manifest.json";manifest=json.loads(manifest_path.read_text())
    if (checkpoint/"COMPLETE").read_text()!=hashlib.sha256(manifest_path.read_bytes()).hexdigest():raise ValueError("manifest completion mismatch")
    if manifest["progress"]["phase"]!="sft" or manifest["progress"]["step"]!=150:raise ValueError("recovery requires the verified best SFT source")
    for name,digest in manifest["files"].items():
        path=(checkpoint/name).resolve()
        if not path.is_relative_to(checkpoint.resolve()):raise ValueError("unsafe manifest path")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError(f"checkpoint integrity failure: {name}")
    return manifest


def export(checkpoint,output):
    import torch
    checkpoint=Path(checkpoint);output=Path(output);manifest=verify_source(checkpoint)
    manifest_path=checkpoint/"manifest.json"
    state=torch.load(checkpoint/"state.pt",map_location="cpu",weights_only=False)
    arrays,inventory=portable_state(state["trainable"])
    output.mkdir(parents=True,exist_ok=True);np.savez(output/"trainables.npz",**arrays)
    tokenizer=json.loads((checkpoint/"tokenizer/tokenizer_config.json").read_text())
    modes={x["content"]:int(i) for i,x in tokenizer["added_tokens_decoder"].items() if x["content"].startswith("<mode:")}
    if set(modes)!={"<mode:jev>","<mode:direct>","<mode:cot>"}:raise ValueError("mode token inventory mismatch")
    result={"schema_version":2,"backend":"portable-jax-input","source_manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "source_state_sha256":manifest["files"]["state.pt"],"source_step":150,"model":manifest["progress"]["model"],
            "architecture":manifest["architecture"],"mode_tokens":modes,"vocabulary_boundary":min(modes.values()),"inventory":inventory,
            "npz_sha256":hashlib.sha256((output/"trainables.npz").read_bytes()).hexdigest(),
            "fresh_optimizer":True,"probability_parity_passed":False,"paid_launch_authorized":False}
    (output/"manifest.json").write_text(json.dumps(result,indent=2));return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    print(json.dumps(export(a.checkpoint,a.output),indent=2))
