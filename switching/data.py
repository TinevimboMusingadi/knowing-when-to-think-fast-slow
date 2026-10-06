"""Verified procedural episodes plus conservative reuse of existing math prompts."""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path
from .protocol import MODES, encode_action,parse_action

COUNTS = {"fast": 2000, "direct": 1000, "reasoning": 2000, "reason_decide": 1000, "clarification": 1000, "lookup": 1000}

def candidates(answer, rng):
    values = [answer, answer + 1, answer + 7, answer - 3]
    rng.shuffle(values)
    return [{"id": f"c{i}", "value": value, "text": str(value)} for i, value in enumerate(values)]

def make_episode(split, behavior, index):
    seed = int(hashlib.sha256(f"{split}/{behavior}/{index}".encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    offset = {"fast":0,"direct":5000,"reasoning":0,"reason_decide":4000,"clarification":0,"lookup":0}[behavior]
    unique = index + offset
    unique += {"train":0,"val":10000,"test":20000}[split]
    a, b, c = 3 + unique // (97*97), 3 + (unique // 97) % 97, 3 + unique % 97
    # Disjoint expression families across splits; never split variants of one source.
    formulas = {"train": (a * b + c, f"({a} * {b}) + {c}"), "val": ((a + b) * c, f"({a} + {b}) * {c}"), "test": (a * b - c, f"({a} * {b}) - {c}")}
    answer, expr = formulas[split]
    if behavior in {"fast", "direct"}:
        a, b = 3 + unique // 97, 3 + unique % 97
        answer, expr = {"train": (a + b, f"{a} + {b}"), "val": (a - b, f"{a} - {b}"), "test": (max(a, b), f"the larger of {a} and {b}")}[split]
    state = f"Calculate {expr}."
    opts = candidates(answer, rng)
    question_type="choice"
    if behavior=="fast" and index%3==1:
        if index%2:a,b=b,a
        answer="yes" if a>b else "no"
        state=f"Are {a} items more than {b} items? Use unknown only if the quantities are unavailable."
        opts=[{"id":v,"value":v,"text":v} for v in ("yes","no","unknown")]
        question_type="yes_no_unknown"
    elif behavior=="fast" and index%3==2:
        number=index%9
        if split=="train":
            answer=min(5,number);state=f"Score the count {number} using the rubric score=min(5,count). Return an integer from 0 through 5."
        elif split=="val":
            answer=max(0,5-number);state=f"Score {number} defects using the rubric score=max(0,5-defects). Return an integer from 0 through 5."
        else:
            answer=min(5,number//2);state=f"Score {number} completed items using the rubric score=min(5,floor(items/2)). Return an integer from 0 through 5."
        opts=[{"id":f"score-{v}","value":v,"text":str(v)} for v in range(6)]
        question_type="bounded_score"
    row = {"id": f"{split}-{behavior}-{index}", "group": f"{split}-{behavior}-family", "behavior": behavior, "question_type":question_type,"prompt": state, "answer": answer, "candidates": opts, "verified": True, "source": "deterministic_arithmetic_v1", "steps": []}
    if behavior == "fast":
        row["steps"] = [{"completion": encode_action(MODES[0], "decide", state=state), "decision": answer}]
    elif behavior == "direct":
        row["steps"] = [{"completion": encode_action(MODES[1], "answer", value=answer)}]
    elif behavior in {"reasoning", "reason_decide"}:
        row["steps"] = [{"completion": encode_action(MODES[2], "reason", text=f"Evaluating {expr} gives {answer}.")}, {"completion": encode_action(MODES[0] if behavior == "reason_decide" else MODES[1], "decide" if behavior == "reason_decide" else "answer", **({"state": f"The computed result is {answer}; select the matching candidate."} if behavior == "reason_decide" else {"value": answer})), **({"decision": answer} if behavior == "reason_decide" else {})}]
    else:
        # Context is necessary; candidate labels do not uniquely reveal the missing value.
        key = f"record-{split}-{index}"
        row["prompt"] = f"A record has an undisclosed numeric value. Return its value. Record ID: {key}. " + ("Ask me for the value." if behavior == "clarification" else f"Use lookup with query {key}.")
        answer = rng.randint(100, 9999)
        row.update(answer=answer, candidates=candidates(answer, rng))
        action = "ask" if behavior == "clarification" else "lookup"
        row["steps"] = [{"completion": encode_action(MODES[1], action, **({"question": "What is the record value?"} if action == "ask" else {"query": key}))}, {"completion": encode_action(MODES[0], "decide", state=f"The record value is {answer}; select its matching candidate."), "decision": answer}]
        row["clarification" if action == "ask" else "facts"] = str(answer) if action == "ask" else {key: str(answer)}
    row["prompt"] += " Candidates: " + json.dumps(opts)
    row["optimal_actions"] = len(row["steps"])
    for step in row["steps"]:
        if "decision" in step:step["completion"]=MODES[0]
    return row

def reused_rows(source, gold_file=None):
    """Reuse only prompts whose answer we independently calculate; never import completions."""
    seen = set()
    audit = collections.Counter()
    rows = []
    gold = {}
    if gold_file:
        for line in Path(gold_file).read_text(encoding="utf-8").splitlines():
            record=json.loads(line)
            key=" ".join(record["question"].lower().split())
            value=record["answer"].split("####")[-1].strip().replace(",", "")
            if re.fullmatch(r"-?\d+(?:\.\d+)?",value):
                gold[key]=(float(value), record["answer"].split("####")[0].strip())
    with Path(source).open(encoding="utf-8") as stream:
        for line in stream:
            audit["source_rows"] += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                audit["invalid_json"] += 1
                continue
            prompt = item.get("prompt", "")
            key = hashlib.sha256(" ".join(prompt.lower().split()).encode()).hexdigest()
            if key in seen:
                audit["duplicates"] += 1
                continue
            seen.add(key)
            verified = gold.get(" ".join(prompt.lower().split()))
            if verified:
                rows.append({"prompt":prompt,"answer":verified[0],"reasoning":verified[1],"source_hash":key,"answer_source":"gsm8k_train"})
                audit["verified_against_original_training_answer"]+=1
                continue
            # Strict, independently verifiable arithmetic subset only.
            match = re.fullmatch(r"\s*(?:What is |Calculate )?(\d+)\s*([+*−-])\s*(\d+)\??\s*", prompt)
            if not match:
                audit["not_independently_verified"] += 1
                continue
            a, op, b = int(match[1]), match[2], int(match[3])
            answer = a + b if op == "+" else a * b if op == "*" else a - b
            rows.append({"prompt": prompt, "answer": answer, "source_hash": key})
            audit["accepted"] += 1
    return rows, dict(audit)

def build(output, source=None, gold_file=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    reused, audit = reused_rows(source, gold_file) if source else ([], {})
    manifest = {"schema_version": 1, "source_audit": audit, "splits": {}, "limitations": "Procedural arithmetic and controlled context fixtures; not evidence of broad domain generalization. Old teacher completions are not imported."}
    source_sha = hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None
    used = 0
    for split in ("train", "val", "test"):
        rows = []
        for behavior, count in COUNTS.items():
            for index in range(count if split == "train" else 100):
                row = make_episode(split, behavior, index)
                if split == "train" and behavior in {"reasoning","reason_decide","direct"} and used < len(reused):
                    old = reused[used]; used += 1
                    opts=candidates(old["answer"],random.Random(used))
                    row.update(prompt=old["prompt"]+" Candidates: "+json.dumps(opts), answer=old["answer"], candidates=opts, source="existing_verified_prompt", answer_source=old.get("answer_source","independent_arithmetic"), group=old["source_hash"])
                    if behavior=="direct": row["steps"]=[{"completion":encode_action(MODES[1],"answer",value=old["answer"])}]
                    else:
                        reasoning=old.get("reasoning",f"The calculated answer is {old['answer']}.")
                        row["steps"]=[{"completion":encode_action(MODES[2],"reason",text=reasoning)}, {"completion":encode_action(MODES[0],"decide",state=f"The computed result is {old['answer']}; select the matching candidate."),"decision":old["answer"]}] if behavior=="reason_decide" else [{"completion":encode_action(MODES[2],"reason",text=reasoning)}, {"completion":encode_action(MODES[1],"answer",value=old["answer"])}]
                    row["optimal_actions"]=len(row["steps"])
                    for step in row["steps"]:
                        if "decision" in step:step["completion"]=MODES[0]
                rows.append(row)
        random.Random(42).shuffle(rows)
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        manifest["splits"][split] = {"count": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "behaviors": dict(collections.Counter(row["behavior"] for row in rows))}
    manifest.update(reused_examples=used, source_sha256=source_sha, gold_source_sha256=hashlib.sha256(Path(gold_file).read_bytes()).hexdigest() if gold_file else None)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="data"); parser.add_argument("--source");parser.add_argument("--gold-file")
    args = parser.parse_args(); print(json.dumps(build(args.output, args.source,args.gold_file), indent=2))

if __name__ == "__main__": main()
