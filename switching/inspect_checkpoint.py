"""Inspect real parameter-update evidence without loading the frozen Qwen base."""
import argparse
import json
from pathlib import Path
import torch
from .storage import checksum

def inspect(directory):
    directory=Path(directory);manifest_path=directory/"manifest.json"
    if (directory/"COMPLETE").read_text()!=checksum(manifest_path):raise ValueError("incomplete manifest")
    manifest=json.loads(manifest_path.read_text())
    if checksum(directory/"state.pt")!=manifest["files"]["state.pt"]:raise ValueError("state checksum mismatch")
    state=torch.load(directory/"state.pt",map_location="cpu",weights_only=True)
    parameters=state["trainable"]
    lora=[value for name,value in parameters.items() if ".lora_B." in name]
    rows=[value.float() for name,value in parameters.items() if name.endswith(".rows")]
    head_ids=[index for index,name in enumerate(parameters) if name.startswith("head.")]
    moments=state["optimizer"]["state"]
    record={"progress":state["progress"],"checkpoint_sha256":manifest["files"]["state.pt"],"finite_parameters":all(bool(torch.isfinite(p).all()) for p in parameters.values()),"nonzero_lora_B_matrices":sum(bool(torch.count_nonzero(p)) for p in lora),"lora_B_matrices":len(lora),"mode_row_difference":max(float((r-r[:1]).abs().max()) for r in rows),"head_has_nonzero_optimizer_moment":any(bool(torch.count_nonzero(moments[i]["exp_avg"])) for i in head_ids if i in moments)}
    record["parameter_updates_verified"]=record["finite_parameters"] and record["nonzero_lora_B_matrices"]>0 and record["mode_row_difference"]>0 and record["head_has_nonzero_optimizer_moment"]
    return record

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--checkpoint",required=True);parser.add_argument("--output")
    args=parser.parse_args();record=inspect(args.checkpoint);text=json.dumps(record,indent=2)
    if args.output:Path(args.output).write_text(text)
    print(text)
