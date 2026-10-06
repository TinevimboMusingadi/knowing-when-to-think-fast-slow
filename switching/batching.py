"""Prompt masking and independently attended packed generation examples."""
import torch
from .protocol import EpisodeEnv, parse_action,decision_state

def supervised_items(model, episodes):
    generation, decisions = [], []
    for episode in episodes:
        env = EpisodeEnv(episode)
        for step in episode["steps"]:
            prompt = model.prompt_ids(env.messages)
            completion = model.tokenizer.encode(step["completion"] + "<|im_end|>", add_special_tokens=False)
            generation.append((prompt + completion, [-100] * len(prompt) + completion))
            _, action = parse_action(step["completion"])
            if action["action"] == "decide":
                target = next(i for i,c in enumerate(episode["candidates"]) if c["value"] == step["decision"])
                decisions.append((decision_state(env.messages), episode["candidates"], target))
            env.step(step["completion"], step.get("decision"))
    return generation, decisions

def pack(items, bucket, microbatch, pad_id, device):
    rows, labels, segments = [], [], []
    for ids, loss_labels in items:
        if len(ids) > bucket: raise ValueError("example exceeds bucket; never truncate targets")
        if not rows or len(rows[-1]) + len(ids) > bucket:
            if len(rows) == microbatch: break
            rows.append([]); labels.append([]); segments.append([])
        segment = max(segments[-1], default=-1) + 1
        rows[-1].extend(ids); labels[-1].extend(loss_labels); segments[-1].extend([segment]*len(ids))
    consumed = sum(max(s, default=-1)+1 for s in segments)
    if not rows: raise ValueError("empty batch")
    while len(rows) < microbatch: rows.append([]); labels.append([]); segments.append([])
    lengths = [len(r) for r in rows]
    positions = []
    for seg in segments:
        current=-1; offset=0; pos=[]
        for s in seg:
            if s != current: offset=0; current=s
            pos.append(offset); offset+=1
        positions.append(pos+[0]*(bucket-len(pos)))
    ids = torch.tensor([r+[pad_id]*(bucket-len(r)) for r in rows],device=device)
    target = torch.tensor([r+[-100]*(bucket-len(r)) for r in labels],device=device)
    seg = torch.tensor([r+[-1]*(bucket-len(r)) for r in segments],device=device)
    causal = torch.arange(bucket,device=device)[:,None] >= torch.arange(bucket,device=device)[None,:]
    allowed = (seg[:,:,None] == seg[:,None,:]) & (seg[:,:,None] >= 0) & causal
    # Padding queries may see their own pad position to avoid all-masked softmax.
    allowed |= (seg[:,:,None] < 0) & torch.eye(bucket,device=device,dtype=torch.bool)[None,:,:]
    attention = torch.zeros_like(allowed,dtype=torch.float32).masked_fill(~allowed, torch.finfo(torch.float32).min)[:,None,:,:]
    batch = {"input_ids":ids,"labels":target,"position_ids":torch.tensor(positions,device=device),"attention_mask":attention,"use_cache":False}
    return batch, consumed, sum(lengths)
