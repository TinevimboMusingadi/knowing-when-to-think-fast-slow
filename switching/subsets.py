"""Deterministic balanced holdout selection, independent of model outcomes."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

BEHAVIORS=("fast","direct","reasoning","reason_decide","clarification","lookup")

def balanced_rows(rows,per_behavior=10,seed=42):
    if per_behavior<1:raise ValueError("per-behavior count must be positive")
    if len({r["id"] for r in rows})!=len(rows):raise ValueError("duplicate episode IDs")
    groups={name:[] for name in BEHAVIORS}
    for row in rows:
        if row["behavior"] not in groups:raise ValueError("unknown behavior")
        groups[row["behavior"]].append(row)
    for name,group in groups.items():
        if len(group)<per_behavior:raise ValueError(f"insufficient {name} episodes")
        group.sort(key=lambda r:hashlib.sha256(f"{seed}/{r['id']}".encode()).hexdigest())
    return [groups[name][index] for index in range(per_behavior) for name in BEHAVIORS]

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output",required=True);p.add_argument("--per-behavior",type=int,default=10);p.add_argument("--seed",type=int,default=42)
    args=p.parse_args();source=Path(args.data);output=Path(args.output)
    if source.resolve()==output.resolve():raise ValueError("subset must not overwrite its source")
    rows=[json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected=balanced_rows(rows,args.per_behavior,args.seed)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in selected),encoding="utf-8",newline="\n")
    manifest={"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"subset_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"seed":args.seed,"selection":"seeded ID hash within behavior; interleaved behaviors","count":len(selected),"behaviors":dict(collections.Counter(r["behavior"] for r in selected)),"ids":[r["id"] for r in selected]}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k!="ids"},indent=2))

if __name__=="__main__":main()
