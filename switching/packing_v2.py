"""Attention-isolated supervised packing with reset positions and target indices."""
import numpy as np


def packs(events,pad,buckets=(512,1024,2048)):
    groups=[];current=[];size=0
    for event in sorted(events,key=lambda e:len(e["prompt"])+len(e["completion"]),reverse=True):
        length=len(event["prompt"])+len(event["completion"])
        if not event["prompt"] or not event["completion"]:raise ValueError("nonempty prompt and assistant completion required")
        if length>max(buckets):raise ValueError("supervision exceeds actual context capacity; refusing to truncate")
        if size+length>max(buckets):groups.append(current);current=[];size=0
        current.append(event);size+=length
    if current:groups.append(current)
    result=[]
    for group in groups:
        total=sum(len(e["prompt"])+len(e["completion"]) for e in group)
        width=next(b for b in buckets if b>=total)
        ids=np.full(width,pad,dtype=np.int32);segments=np.zeros(width,dtype=np.int32);positions=np.zeros(width,dtype=np.int32)
        selected=[];targets=[];offset=0;useful=0
        for index,event in enumerate(group,1):
            values=event["prompt"]+event["completion"];length=len(values);prompt=len(event["prompt"])
            ids[offset:offset+length]=values;segments[offset:offset+length]=index;positions[offset:offset+length]=np.arange(length)
            selected.extend(offset+prompt-1+np.arange(len(event["completion"])));targets.extend(event["completion"])
            offset+=length;useful+=length
        mask=(segments[:,None]==segments[None,:])&(segments[:,None]>0)&np.tri(width,dtype=bool)
        # Padded queries attend only themselves and never affect valid positions.
        mask|=np.diag(segments==0)
        result.append({"ids":ids[None,:],"positions":positions[None,:],"mask":mask[None,:,:],"segments":segments[None,:],
                       "selected":np.asarray(selected,dtype=np.int32),"targets":np.asarray(targets,dtype=np.int32),
                       "useful_tokens":useful,"allocation_tokens":width,"episodes":len(group)})
    return result


def batch_packs(items,pad,microbatch=1):
    if microbatch not in (1,2,4):raise ValueError("pilot microbatches are 1, 2, or 4")
    buckets={}
    for item in items:
        count=len(item["selected"]);projection=next(x for x in (32,64,128,256,512,1024,2048) if x>=count)
        buckets.setdefault((item["allocation_tokens"],projection),[]).append(item)
    result=[]
    for (_,projection),group in buckets.items():
        for start in range(0,len(group),microbatch):
            rows=group[start:start+microbatch]
            batch={name:np.concatenate([r[name] for r in rows]) for name in ("ids","positions","mask","segments")}
            batch["selected"]=np.stack([np.pad(r["selected"],(0,projection-len(r["selected"]))) for r in rows])
            batch["targets"]=np.stack([np.pad(r["targets"],(0,projection-len(r["targets"]))) for r in rows])
            batch["target_mask"]=np.stack([np.arange(projection)<len(r["targets"]) for r in rows])
            batch["useful_tokens"]=sum(r["useful_tokens"] for r in rows)
            batch["allocation_tokens"]=sum(r["allocation_tokens"] for r in rows)
            result.append(batch)
    return result
