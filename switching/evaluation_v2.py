"""Honest paired summaries and predeclared balanced selection; no model dependency."""
import collections
import math
import random


def wilson(successes,total,z=1.959963984540054):
    if not total:return None
    p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    width=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,center-width),min(1.,center+width)]


def select(rows,count,seed=42):
    if count not in (60,120) or count%6:raise ValueError("predeclared balanced test size required")
    groups=collections.defaultdict(list)
    for row in rows:groups[row["behavior"]].append(row)
    if len(groups)!=6:raise ValueError("all six behavior strata required")
    result=[]
    for behavior in sorted(groups):
        families=collections.defaultdict(list)
        for row in groups[behavior]:families[row["group"]].append(row)
        current=sorted(families);random.Random(f"{seed}/{behavior}").shuffle(current)
        needed=count//6;selected=[]
        for key in current:
            pair=sorted(families[key],key=lambda r:r["id"])
            if len(pair)!=2:raise ValueError("held-out source groups must retain their two paired variants")
            if len(selected)<needed:selected.extend(pair)
        if len(selected)!=needed:raise ValueError("insufficient held-out pairs")
        result.extend(selected)
    return result


def summary(records):
    n=len(records)
    if not n:return {"count":0,"status":"incomplete"}
    correct=sum(bool(r["correct"]) for r in records)
    grounded=sum(bool(r["correct"] and r["grounded"]) for r in records)
    strata=collections.defaultdict(list)
    for r in records:strata[r["behavior"]].append(r)
    means={key:sum(float(r.get(key,0)) for r in records)/n for key in ("tokens","candidate_encoding_tokens","prefill_calls","prefill_tokens","reused_prefix_tokens","decode_calls","head_calls","warmed_seconds")}
    return {"count":n,"accuracy":correct/n,"grounded_accuracy":grounded/n,"accuracy_95_wilson":wilson(correct,n),
            "protocol_validity":sum(not r.get("error") for r in records)/n,"means":means,
            "by_behavior":{b:{"count":len(rs),"grounded_accuracy":sum(r["correct"] and r["grounded"] for r in rs)/len(rs)} for b,rs in strata.items()},
            "errors":dict(collections.Counter(r["error"] for r in records if r.get("error")))}


def paired_interval(a,b,iterations=2000,seed=42):
    aa={r["id"]:r for r in a};bb={r["id"]:r for r in b}
    if set(aa)!=set(bb) or len(aa)!=len(a) or len(bb)!=len(b):raise ValueError("paired policies require exactly matching unique IDs")
    groups=collections.defaultdict(list)
    for identifier in aa:
        ra,rb=aa[identifier],bb[identifier]
        groups[ra.get("group",identifier)].append(int(rb["correct"] and rb["grounded"])-int(ra["correct"] and ra["grounded"]))
    keys=list(groups);rng=random.Random(seed);values=[]
    for _ in range(iterations):
        sample=[v for key in rng.choices(keys,k=len(keys)) for v in groups[key]]
        values.append(sum(sample)/len(sample))
    values.sort();return [values[int(iterations*.025)],values[min(iterations-1,int(iterations*.975))]]
