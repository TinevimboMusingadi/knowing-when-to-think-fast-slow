"""Deterministic corrective episodes; import only independently verified public gold."""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path
from .episode_v2 import EpisodeEnvV2
from .protocol import MODES, encode_action

BEHAVIORS = ("fast", "direct", "reasoning", "reason_decide", "clarification", "lookup")
FAMILIES = {"train": ("addition", "multiply_add", "inventory", "half_sum"),
            "val": ("difference", "distribute", "perimeter", "net_stock"),
            "test": ("maximum", "multiply_subtract", "double_net", "paired_groups")}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def problem(family, a, b, c):
    templates = {
        "addition": (a + b, f"Add {a} and {b}.", f"{a} + {b} = {a+b}."),
        "multiply_add": (a*b+c, f"There are {a} groups of {b} items and {c} extra items. How many items?", f"The groups contain {a*b} items. Adding {c} gives {a*b+c}."),
        "inventory": (a+b-c, f"Stock starts at {a}, receives {b}, then uses {c}. Find the remaining stock.", f"After delivery there are {a+b}; subtracting {c} leaves {a+b-c}."),
        "half_sum": (a+b, f"Split {2*a} and {2*b} items equally between two people. How many does each receive?", f"There are {2*a+2*b} items. Dividing by two gives {a+b}."),
        "difference": (a-b, f"Find how much {a} exceeds {b}.", f"Subtracting {b} from {a} gives {a-b}."),
        "distribute": ((a+b)*c, f"Each of {c} bags contains {a} red and {b} blue beads. Find the total beads.", f"Each bag has {a+b} beads. {c} bags have {(a+b)*c}."),
        "perimeter": (2*(a+b), f"A rectangle has sides {a} and {b}. Find its perimeter.", f"Add the sides to get {a+b} and double to get {2*(a+b)}."),
        "net_stock": (a-b+c, f"A store owns {a} units, sells {b}, and obtains {c}. How many units remain?", f"Selling leaves {a-b}; obtaining {c} results in {a-b+c}."),
        "maximum": (max(a,b), f"Which is greater, {a} or {b}?", f"Comparing the two numbers gives {max(a,b)}."),
        "multiply_subtract": (a*b-c, f"A delivery has {a} boxes containing {b} parts each. Remove {c} parts. How many are left?", f"There are {a*b} parts initially; removing {c} leaves {a*b-c}."),
        "double_net": (2*(a-b), f"Two stations each receive {a} packages and dispatch {b}. Find their combined remaining packages.", f"One station retains {a-b}; two retain {2*(a-b)}."),
        "paired_groups": (a*(b+c), f"Each of {a} teams has {b} players and {c} substitutes. Count all participants.", f"A team has {b+c} participants. Multiplying by {a} gives {a*(b+c)}."),
    }
    return templates[family]


def choices(answer, rng):
    if isinstance(answer, str): values = ["yes", "no", "unknown"]
    else:
        offsets=rng.sample([x for x in range(-19,20) if x],3)
        values=[answer]+[answer+x for x in offsets]
    rows = [{"id": digest(str(value)+str(rng.random()))[:12], "kind": "answer", "value": value, "text": str(value)} for value in values]
    rows.append({"id": "defer", "kind": "defer", "text": "Defer this decision; continue the unfinished task."})
    rng.shuffle(rows)
    return rows


def make(split, behavior, index, seed=42):
    pair = index//2
    group = digest(f"recovery-v2/{seed}/{split}/{behavior}/{pair}")
    rng = random.Random(int(group[:16],16))
    family = FAMILIES[split][pair % len(FAMILIES[split])]
    a,b,c = rng.randint(1000,9999),rng.randint(2,79),rng.randint(1,30)
    answer,prompt,reason = problem(family,a,b,c)
    # Variants of a single source stay in one split, but context availability differs.
    if index%2: prompt = "Please solve this task: " + prompt
    opts = choices(answer,rng)
    visible = {"prompt": prompt, "candidates": opts, "clarification_fields": [], "lookup_queries": []}
    oracle = {"answer": answer, "required_fields": [], "clarifications": {}, "facts": {}}
    steps = []
    if behavior == "fast" and pair%3 == 1:
        b=a+(1 if pair%2 else -1)
        answer = "yes" if {"train":a>b,"val":a>=b,"test":a<b}[split] else "no"
        comparison={"train":"more than","val":"at least as many as","test":"fewer than"}[split]
        family=f"comparison-{split}"
        visible["prompt"] = f"Are {a} units {comparison} {b} units?"
        if index%2:
            answer="unknown";visible["prompt"]=f"Are {a} units {comparison} an unavailable quantity?"
        visible["candidates"]=choices(answer,rng);oracle["answer"]=answer
    elif behavior == "fast" and pair%3 == 2:
        answer = (pair//3)%6
        family=f"rubric-{split}"
        if split=="train":visible["prompt"] = f"Using score=min(5,max(0,count-{a})), score count {a+answer} on the integer scale 0 through 5."
        elif split=="val":visible["prompt"] = f"Using score=min(5,max(0,{a+5}-defects)), score {a+5-answer} defects on the integer scale 0 through 5."
        else:visible["prompt"] = f"Using score=min(5,max(0,floor((items-{a})/2))), score {a+2*answer+index%2} items on the integer scale 0 through 5."
        visible["candidates"]=[{"id":f"score-{v}","kind":"answer","value":v,"text":str(v)} for v in range(6)]+[{"id":"defer","kind":"defer","text":"Defer this decision; continue the unfinished task."}]
        rng.shuffle(visible["candidates"]);oracle["answer"]=answer
    if behavior in {"clarification","lookup"}:
        family={"train":"record-value","val":"record-double","test":"record-offset"}[split]
        record = digest(group+"record")[:12]
        field="record.value";query=f"record:{record}"
        record_value=rng.randint(100,9999);missing=index%2==0
        answer={"train":record_value,"val":2*record_value,"test":record_value+c}[split]
        task={"train":f"Return the numeric value of record {record}.","val":f"Two batches each have the quantity stored in record {record}. Return their combined quantity.","test":f"Start with the quantity in record {record} and add {c}. Return the resulting quantity."}[split]
        visible.update(prompt=task + (" Its value is unavailable in the current context." if missing else f" Its value is {record_value}."),
                       candidates=choices(answer,rng),clarification_fields=[field],lookup_queries=[query])
        oracle.update(answer=answer,required_fields=[field] if missing else [],clarifications={field:str(record_value)},facts={query:{"text":f"{field}={record_value}","fields":[field]}})
        visible["candidates"] += [{"id":"ask-value","kind":"ask","text":"Ask for record.value.","payload":{"field":field,"question":"What is record value?"}},
                                  {"id":"lookup-value","kind":"lookup","text":f"Look up {query}.","payload":{"query":query}}]
        rng.shuffle(visible["candidates"])
        if missing:
            action={"action":"ask","field":field,"question":"What is record value?"} if behavior=="clarification" else {"action":"lookup","query":query}
            # Ordinary CoT can acquire information as well as direct mode.
            mode=MODES[2] if pair%2 else MODES[1]
            steps.append({"text":mode+json.dumps(action)})
    if behavior=="direct": steps.append({"text":encode_action(MODES[1],"answer",value=answer)})
    else:
        if behavior=="reasoning" and pair%2==0:
            steps.append({"text":MODES[0],"choice_id":"defer"})
        if behavior in {"reasoning","reason_decide"}:
            steps.append({"text":encode_action(MODES[2],"reason",text=reason)})
        if behavior=="reasoning" and pair%2:
            steps.append({"text":encode_action(MODES[1],"answer",value=answer)})
        else:
            chosen=next(x["id"] for x in visible["candidates"] if x["kind"]=="answer" and x["value"]==answer)
            steps.append({"text":MODES[0],"choice_id":chosen})
    return {"schema_version":2,"id":digest(group+str(index%2))[:24],"group":group,"family":family,"behavior":behavior,
            "source":{"kind":"deterministic_v2","problem_hash":digest(visible["prompt"]),"verified":True,"license":"repository"},
            "visible":visible,"oracle":oracle,"supervision":steps}


def public_gold(path):
    result={}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line);final=row["answer"].split("####")[-1].strip().replace(",","")
            if re.fullmatch(r"-?\d+(?:\.\d+)?",final):
                result[digest(" ".join(row["question"].lower().split()))]=(row["question"],float(final),re.sub(r"<<[^>]*>>","",row["answer"].split("####")[0]).strip())
    return result


def inventory_previous(root,gold_rows):
    root=Path(root);seen=set();files=[];verified=0
    paths=sorted((root/"synthesized").glob("*.jsonl"))+[root/name for name in ("train_phase2_25k.jsonl","train_phase3_60k.jsonl","val_phase3_5k.jsonl")]
    for path in paths:
        if not path.exists():continue
        count=0;invalid=0;matched=0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                count+=1
                try:row=json.loads(line)
                except json.JSONDecodeError:invalid+=1;continue
                prompt=row.get("prompt","")
                if not isinstance(prompt,str) or not prompt:continue
                key=digest(" ".join(prompt.lower().split()));seen.add(key)
                if key in gold_rows:matched+=1
        verified+=matched
        files.append({"file":path.relative_to(root).as_posix(),"rows":count,"invalid_json":invalid,"public_gold_matches":matched,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    return seen,{"files":files,"unique_problem_hashes":len(seen),"matched_rows_not_unique":verified,"raw_completions_imported":0,"limitations":"Overlapping collections; matching public gold verifies provenance, not base-model pretraining contamination."}


def build(output, legacy="data/train.jsonl", gold="data/gsm8k_train.jsonl", previous="D:/mode-switch-llms/data"):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    gold_rows=public_gold(gold);verified=[];prior_hashes,source_inventory=inventory_previous(previous,gold_rows)
    gold_ids={digest(' '.join(json.loads(line)['question'].lower().split())):index
              for index,line in enumerate(Path(gold).read_text(encoding='utf-8').splitlines())}
    with Path(legacy).open(encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line);question=row["prompt"].split(" Candidates: ")[0]
            question=re.sub(r"^First provide a typed assessment.*?problem\. ","",question)
            key=digest(" ".join(question.lower().split()));prior_hashes.add(key)
            if key in gold_rows and key not in {x[0] for x in verified}:verified.append((key,*gold_rows[key]))
    manifest={"schema_version":2,"splits":{},"reused_public_gold":0,"private_completions_imported":0,"source_inventory":source_inventory,
              "legacy_train_sha256":hashlib.sha256(Path(legacy).read_bytes()).hexdigest(),"gold_sha256":hashlib.sha256(Path(gold).read_bytes()).hexdigest()}
    groups=[];reuse=iter(verified)
    for split,count in (("train",256),("val",20),("test",100)):
        rows=[make(split,b,i) for b in BEHAVIORS for i in range(count)]
        if split=="train":
            for row in rows:
                if row["behavior"] not in {"reasoning","reason_decide"}:continue
                if int(row["id"],16)%2:continue
                old=next(reuse,None)
                if old is None:break
                key,question,answer,reason=old;rng=random.Random(key)
                row["group"]=key;row["family"]="public_gsm8k_train";row["visible"]["prompt"]=question
                row["visible"]["candidates"]=choices(answer,rng);row["oracle"]["answer"]=answer
                row["source"]={"kind":"gsm8k_train","problem_hash":key,"verified":True,"license":"MIT",
                               'original_source_id':f'cached-gsm8k-train-line:{gold_ids[key]}','license_file':'licenses/GSM8K-MIT.txt'}
                chosen=next(c["id"] for c in row["visible"]["candidates"] if c["kind"]=="answer" and c["value"]==answer)
                row["supervision"]=[{"text":encode_action(MODES[2],"reason",text=reason)},{"text":MODES[0],"choice_id":chosen}]
                manifest["reused_public_gold"]+=1
        current=set()
        for row in rows:
            if split!="train" and digest(" ".join(row["visible"]["prompt"].lower().split())) in prior_hashes:raise ValueError("prior training overlap")
            env=EpisodeEnvV2(row)
            for action in row["supervision"]:env.step(action["text"],action.get("choice_id"))
            _,info=env.reward(0)
            if not info["correct"] or not info["grounded"] or info["error"]:raise ValueError(f"invalid teacher {row['id']}: {info}")
            current.add(row["group"])
        if any(current & old for old in groups):raise ValueError("source group overlap")
        groups.append(current);random.Random(42).shuffle(rows)
        path=output/f"{split}.jsonl";path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
        manifest["splits"][split]={"count":len(rows),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"behaviors":dict(collections.Counter(r["behavior"] for r in rows)),"families":sorted({r["family"] for r in rows})}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="data/recovery-v2");args=parser.parse_args()
    print(json.dumps(build(args.output),indent=2))
