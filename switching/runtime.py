"""Batched rollouts and executable mode switching, shared by RL and evaluation."""
import time
import torch
from .protocol import EpisodeEnv, MODES, parse_action,decision_state

def synchronize(device):
    if device.type == "xla":
        import torch_xla
        torch_xla.sync(wait=True)
    elif device.type == "cuda": torch.cuda.synchronize(device)

def token_logps(model, prompt, completion, bucket=2048):
    if len(prompt)+len(completion)>bucket: raise ValueError("rollout exceeds context bucket")
    device=next(model.parameters()).device
    values=prompt+completion
    ids=torch.tensor([values+[model.tokenizer.pad_token_id]*(bucket-len(values))],device=device)
    mask=torch.arange(bucket,device=device)[None,:]<len(values)
    logits=model.lm(input_ids=ids,attention_mask=mask,use_cache=False).logits[0].float()
    # Every token, including the mode action, participates in policy likelihood.
    target=torch.tensor(completion,device=device)
    return logits[len(prompt)-1:len(values)-1].log_softmax(-1).gather(1,target[:,None]).squeeze(1)

def rollout_group(model, episodes, max_tokens=512, max_transitions=4, max_lookups=2, sample=True, policy="learned", threshold=0.8):
    device=next(model.parameters()).device
    envs=[EpisodeEnv(e,max_transitions,max_lookups) for e in episodes]
    traces=[[] for _ in episodes]; tokens=[0]*len(episodes); forwards=[0]*len(episodes)
    sync_start=time.perf_counter(); synchronize(device); start=time.perf_counter()
    model.eval()
    from transformers import StoppingCriteria,StoppingCriteriaList
    class StopAtDecision(StoppingCriteria):
        def __init__(self,width):self.width=width
        def __call__(self,ids,scores,**kwargs):
            return ids[:,self.width]==model.tokenizer.convert_tokens_to_ids(MODES[0])
    for action_index in range(8):
        active=[i for i,e in enumerate(envs) if not e.done and tokens[i]<max_tokens]
        if not active: break
        # Static regimes are behavior baselines; learned policies select their own mode.
        if policy in {"always_jev","confidence"} and action_index==0:
            for i in active:
                e=episodes[i]
                if len(e.get("candidates",[]))<2: continue
                with torch.no_grad(): logits=model.decision_logits([e["prompt"].split(" Candidates: ")[0]],[e["candidates"]])
                probabilities=logits[0].softmax(-1); best=int(probabilities.argmax().item()); forwards[i]+=1
                if policy=="always_jev" or float(probabilities.max().item())>=threshold:
                    envs[i].step(MODES[0]+'{"action":"decide","state":"Original context"}',e["candidates"][best]["value"])
                    traces[i].append({"kind":"baseline_decision","probabilities":probabilities.cpu().tolist()})
            active=[i for i in active if not envs[i].done]
            if not active: continue
        prompts=[model.prompt_ids(envs[i].messages) for i in active]
        forced=MODES[1] if policy=="always_direct" else (MODES[2] if action_index==0 else MODES[1]) if policy in {"always_cot","confidence"} else ""
        forced_ids=model.tokenizer.encode(forced,add_special_tokens=False) if forced else []
        actual_prompts=[p+forced_ids for p in prompts]
        width=max(len(p) for p in actual_prompts)
        if width>=2048:
            for i in active: envs[i].error="context budget exhausted"; envs[i].done=True
            break
        input_ids=torch.tensor([[model.tokenizer.pad_token_id]*(width-len(p))+p for p in actual_prompts],device=device)
        attention=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in actual_prompts],device=device)
        remaining=min(max_tokens-tokens[i] for i in active)
        length=min(remaining,2048-width)
        with torch.no_grad():
            outputs=model.lm.generate(input_ids=input_ids,attention_mask=attention,max_new_tokens=length,do_sample=sample,**({"temperature":1.0,"top_p":1.0,"top_k":0} if sample else {}),pad_token_id=model.tokenizer.pad_token_id,eos_token_id=model.tokenizer.convert_tokens_to_ids("<|im_end|>"),stopping_criteria=StoppingCriteriaList([StopAtDecision(width)]),use_cache=True)
        for batch_index,i in enumerate(active):
            generated=outputs[batch_index,width:].cpu().tolist()
            if generated and generated[0]==model.tokenizer.convert_tokens_to_ids(MODES[0]):generated=generated[:1]
            eos=model.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if eos in generated: generated=generated[:generated.index(eos)+1]
            completion=forced_ids+generated
            text=forced+model.tokenizer.decode(generated,skip_special_tokens=False).removesuffix("<|im_end|>").strip()
            tokens[i]+=len(completion); forwards[i]+=len(generated)
            record={"kind":"tokens","prompt":prompts[batch_index],"completion":completion,"text":text}
            traces[i].append(record)
            decision=None
            try:
                mode,action=parse_action(text)
                if mode==MODES[0]:
                    state=decision_state(envs[i].messages)
                    state_length=max(len(model.tokenizer.encode(state+"\nCandidate: "+c["text"],add_special_tokens=False)) for c in episodes[i]["candidates"])
                    bucket=next(b for b in (512,1024,2048) if b>=state_length)
                    with torch.no_grad(): logits=model.decision_logits([state],[episodes[i]["candidates"]],bucket=bucket)
                    probabilities=logits[0].softmax(-1)
                    choice=int(torch.multinomial(probabilities,1).item()) if sample else int(probabilities.argmax().item())
                    decision=episodes[i]["candidates"][choice]["value"]; forwards[i]+=1
                    traces[i].append({"kind":"decision","state":state,"bucket":bucket,"candidates":episodes[i]["candidates"],"choice":choice,"probabilities":probabilities.cpu().tolist()})
            except (ValueError,KeyError,StopIteration): pass
            envs[i].step(text,decision)
    synchronize(device); elapsed=time.perf_counter()-start
    results=[]
    for i,env in enumerate(envs):
        if not env.done: env.error="action or token budget exhausted"; env.done=True
        reward,info=env.reward(tokens[i])
        results.append({"id":episodes[i]["id"],"behavior":episodes[i]["behavior"],"answer":env.answer,"reward":reward,**info,"tokens":tokens[i],"forwards":forwards[i],"modes":env.modes,"lookups":env.lookups,"asked":env.asked,"trace":traces[i],"batch_seconds":elapsed,"batch_size":len(episodes),"synchronization_seconds":start-sync_start})
    return results

def action_logps(model,trace):
    result=[]
    for action in trace:
        if action["kind"]=="tokens": result.append(token_logps(model,action["prompt"],action["completion"]))
        elif action["kind"]=="decision":
            logits=model.decision_logits([action["state"]],[action["candidates"]],bucket=action.get("bucket",512))
            result.append(logits.log_softmax(-1)[0,action["choice"]].reshape(1))
    if not result: raise ValueError("rollout has no trainable actions")
    return torch.cat(result)

class SessionRuntime:
    """Interactive inference pauses for users; fixture auto-replies are only for evaluation."""
    def __init__(self,model,max_transitions=4,max_lookups=2,max_tokens=512):
        self.model=model;self.sessions={};self.limits=(max_transitions,max_lookups,max_tokens)

    def run(self,request,session_id,lookup=None):
        from .protocol import SYSTEM,encode_action
        if session_id in self.sessions:raise ValueError("session already exists")
        if "answer" in request:raise ValueError("public request must not contain a gold answer")
        self.sessions[session_id]={"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":request["prompt"]}],"candidates":request.get("candidates",[]),"tokens":0,"modes":[],"lookups":0,"actions":0,"paused":False,"finished":False}
        return self._advance(session_id,lookup)

    def resume(self,session_id,reply,lookup=None):
        session=self.sessions[session_id]
        if not session["paused"] or session["finished"]:raise ValueError("session is not awaiting clarification")
        session["messages"].append({"role":"user","content":reply});session["paused"]=False
        return self._advance(session_id,lookup)

    def _advance(self,session_id,lookup):
        s=self.sessions[session_id];device=next(self.model.parameters()).device
        self.model.eval()
        for _ in range(8-s["actions"]):
            remaining=self.limits[2]-s["tokens"]
            if remaining<=0:break
            prompt=self.model.prompt_ids(s["messages"])
            if len(prompt)>=2048:break
            ids=torch.tensor([prompt],device=device)
            with torch.no_grad():output=self.model.lm.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=min(remaining,2048-len(prompt)),eos_token_id=self.model.tokenizer.convert_tokens_to_ids("<|im_end|>"),pad_token_id=self.model.tokenizer.pad_token_id,use_cache=True)
            new=output[0,len(prompt):];s["tokens"]+=len(new);s["actions"]+=1
            text=self.model.tokenizer.decode(new,skip_special_tokens=False).removesuffix("<|im_end|>").strip()
            try:mode,action=parse_action(text)
            except ValueError as exc:return self._finish(s,"unresolved",error=str(exc))
            transitions=sum(a!=b for a,b in zip(s["modes"],s["modes"][1:]))+int(bool(s["modes"]) and s["modes"][-1]!=mode)
            if transitions>self.limits[0]:break
            s["modes"].append(mode);s["messages"].append({"role":"assistant","content":text})
            if action["action"]=="answer":return self._finish(s,"complete",answer=action["value"])
            if action["action"]=="decide":
                state=decision_state(s["messages"])
                length=max(len(self.model.tokenizer.encode(state+"\nCandidate: "+c["text"],add_special_tokens=False)) for c in s["candidates"])
                if length>2048:return self._finish(s,"unresolved",error="decision context budget exhausted")
                bucket=next(b for b in (512,1024,2048) if b>=length)
                with torch.no_grad():probs=self.model.decision_logits([state],[s["candidates"]],bucket=bucket)[0].softmax(-1)
                chosen=s["candidates"][int(probs.argmax().item())]
                return self._finish(s,"complete",answer=chosen["value"],candidate_id=chosen["id"],probabilities=probs.cpu().tolist())
            if action["action"]=="ask":
                s["paused"]=True;return {"status":"awaiting_user","question":action["question"],"session_id":session_id,"modes":s["modes"]}
            if action["action"]=="lookup":
                s["lookups"]+=1
                if s["lookups"]>self.limits[1]:break
                evidence=lookup(action["query"]) if lookup else "No lookup evidence available."
                s["messages"].append({"role":"user","content":"External evidence: "+str(evidence)})
            else:s["messages"].append({"role":"user","content":"Continue with a decision or final answer."})
        return self._finish(s,"unresolved",error="execution budget exhausted")

    @staticmethod
    def _finish(session,status,**fields):
        session["finished"]=True
        return {"status":status,"modes":session["modes"],"tokens":session["tokens"],**fields}
