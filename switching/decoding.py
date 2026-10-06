"""Fixed-shape autoregressive decoding for XLA; one static KV cache per action."""
import torch
from transformers import StaticCache


@torch.no_grad()
def static_generate(lm, prompts, max_new_tokens, pad_id, eos_id, jev_id=None,
                    sample=False, capacity=2048, prefill_buckets=(512,1024,2048)):
    if not prompts or any(not p for p in prompts):
        raise ValueError("nonempty prompts required")
    width=next((b for b in prefill_buckets if b>=max(map(len,prompts))),None)
    if width is None or width+max_new_tokens>capacity:
        raise ValueError("static decoding exceeds cache capacity")
    device=next(lm.parameters()).device;dtype=next(lm.parameters()).dtype
    batch=len(prompts)
    ids=torch.tensor([[pad_id]*(width-len(p))+p for p in prompts],device=device)
    valid=torch.tensor([[False]*(width-len(p))+[True]*len(p)+[False]*(capacity-width) for p in prompts],device=device)
    keys=torch.arange(capacity,device=device)
    positions=(valid[:,:width].long().cumsum(-1)-1).clamp(min=0)
    cache=StaticCache(config=lm.config,max_cache_len=capacity)
    causal=keys[None,:]<=torch.arange(width,device=device)[:,None]
    mask=torch.zeros((batch,1,width,capacity),device=device,dtype=dtype).masked_fill(~(causal[None,None,:,:]&valid[:,None,None,:]),torch.finfo(dtype).min)
    output=lm(input_ids=ids,position_ids=positions,attention_mask=mask,past_key_values=cache,cache_position=torch.arange(width,device=device),use_cache=True,logits_to_keep=1)
    logits=output.logits[:,-1,:].float()
    finished=torch.zeros(batch,dtype=torch.bool,device=device);generated=[]
    logical=torch.tensor([len(p) for p in prompts],device=device)
    # Values vary on device; query and KV tensor shapes stay fixed after prefill.
    cursor=torch.tensor([width],device=device)
    for step in range(max_new_tokens):
        token=torch.multinomial(logits.softmax(-1),1).squeeze(-1) if sample else logits.argmax(-1)
        token=torch.where(finished,torch.full_like(token,pad_id),token)
        generated.append(token)
        finished=finished|(token==eos_id)
        if step==0 and jev_id is not None:finished=finished|(token==jev_id)
        if device.type=="xla":
            # Materialize the live cache and sever lazy dependencies on the
            # previous step; .item() alone does not establish an XLA step.
            import torch_xla
            torch_xla.sync(wait=True)
        if bool(finished.all().item()) or step+1==max_new_tokens:break
        valid=valid|((keys[None,:]==cursor)&(~finished[:,None]))
        mask=torch.zeros((batch,1,1,capacity),device=device,dtype=dtype).masked_fill(~valid[:,None,None,:],torch.finfo(dtype).min)
        output=lm(input_ids=token[:,None],position_ids=logical[:,None],attention_mask=mask,past_key_values=cache,cache_position=cursor,use_cache=True,logits_to_keep=1)
        logits=output.logits[:,-1,:].float();logical=logical+(~finished).long();cursor=cursor+1
    return torch.stack(generated,dim=1)
