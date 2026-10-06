"""Fail closed on invalid manifests, public protocol leaks, and source overlap."""
import argparse
import hashlib
import json
import re
from pathlib import Path

def audit(root):
    root=Path(root); manifest=json.loads((root/"data/manifest.json").read_text());groups=[]
    modes={"jev","direct","cot"}; errors=[];problems=set()
    for split,info in manifest["splits"].items():
        path=root/"data"/f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=info["sha256"]:errors.append(f"checksum mismatch: {split}")
        with path.open(encoding="utf-8") as stream:
            rows=[json.loads(line) for line in stream if line.strip()]
        if len(rows)!=info["count"]:errors.append(f"count mismatch: {split}")
        current=set()
        for row in rows:
            problem=" ".join(row["prompt"].split(" Candidates: ")[0].lower().split())
            if problem in problems:errors.append(f"duplicate source problem: {row['id']}")
            problems.add(problem)
            if row["id"] in current:errors.append(f"duplicate episode: {row['id']}")
            current.add(row["id"])
            if not row.get("verified"):errors.append(f"unverified row: {row['id']}")
            for step in row["steps"]:
                for mode in re.findall(r"<mode:([^>]+)>",step["completion"]):
                    if mode not in modes:errors.append(f"unsupported public mode in {row['id']}")
                # JSON actions contain plain reasoning, never operator markup.
                for tag in re.findall(r"<([A-Za-z_][A-Za-z_0-9]*)>",step["completion"]):
                    errors.append(f"unexpected reasoning markup in {row['id']}")
        groups.append({row["group"] for row in rows})
    for i,g in enumerate(groups):
        for h in groups[i+1:]:
            if g&h:errors.append("source groups overlap between splits")
    if errors:raise ValueError("\n".join(errors[:20]))
    return {"verified":True,"train":manifest["splits"]["train"]["count"],"reused":manifest["reused_examples"],"allowed_modes":sorted(modes)}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",default=".");args=parser.parse_args();print(json.dumps(audit(args.root),indent=2))
