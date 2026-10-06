"""Audit exact Qwen episode lengths before allocating paid compute."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from switching.protocol import MODES,EpisodeEnv,decision_state,parse_action

def audit(path,model_id="Qwen/Qwen3-1.7B",offline=False):
    tokenizer=AutoTokenizer.from_pretrained(model_id,local_files_only=offline)
    tokenizer.add_special_tokens({"additional_special_tokens":list(MODES)})
    lengths=[];oversized=[];decision_lengths=[]
    for line in Path(path).open(encoding="utf-8"):
        row=json.loads(line);env=EpisodeEnv(row)
        for step in row["steps"]:
            text="".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in env.messages)
            text+="<|im_start|>assistant\n"+step["completion"]+"<|im_end|>"
            length=len(tokenizer.encode(text,add_special_tokens=False));lengths.append(length)
            if length>2048:oversized.append({"id":row["id"],"tokens":length})
            if parse_action(step["completion"])[1]["action"]=="decide":
                for candidate in env.current_candidates:
                    length=len(tokenizer.encode(decision_state(env.messages)+"\nCandidate: "+candidate["text"],add_special_tokens=False))
                    decision_lengths.append(length)
                    if length>2048:oversized.append({"id":row["id"],"decision_tokens":length})
            env.step(step["completion"],step.get("decision"))
    return {"generation_examples":len(lengths),"max_generation_tokens":max(lengths),"max_decision_tokens":max(decision_lengths),"total_useful_tokens":sum(lengths),"oversized":oversized}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--data",default="data/train.jsonl");parser.add_argument("--output");parser.add_argument("--offline",action="store_true")
    args=parser.parse_args();record=audit(args.data,offline=args.offline);text=json.dumps(record,indent=2)
    if args.output:Path(args.output).write_text(text)
    print(text)
    if record["oversized"]:raise SystemExit("oversized episodes must be repaired; targets cannot be truncated")
