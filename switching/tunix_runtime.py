"""Cached Tunix-native generation and exact mixed-action likelihoods."""
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from .episode_v2 import EpisodeEnvV2
from .protocol import MODES
from .trajectory_v2 import TrajectoryV2
from .tunix_optim import assert_finite


class TunixRuntime:
    def __init__(self,model,tokenizer,context=2048,limits=None):
        self.graph,self.actor,self.frozen=model.split();self.tokenizer=tokenizer;self.context=context
        self.mode_ids=[tokenizer.convert_tokens_to_ids(m) for m in MODES]
        self.eos=tokenizer.convert_tokens_to_ids("<|im_end|>");self.pad=tokenizer.pad_token_id
        self.temperature=1.;self.top_p=1.;self.top_k=None;self.suppress_jev=False;self.stop_at_jev=True
        self.limits=limits or {'max_transitions':4,'max_lookups':2,'max_actions':8}
        self._call=jax.jit(self._language)
        self._head=jax.jit(self._decision)

    def model(self,state,frozen=None):return nnx.merge(self.graph,state,self.frozen if frozen is None else frozen)

    def prompt_ids(self,messages):
        text="".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)+"<|im_start|>assistant\n"
        return self.tokenizer.encode(text,add_special_tokens=False)

    def _language(self,state,frozen,ids,positions,mask,cache,select):
        return self.model(state,frozen)(ids,positions,mask,cache,select_positions=select)

    def _decision(self,state,frozen,ids,lengths,valid):return self.model(state,frozen).candidate_logits(ids,lengths,valid)

    def candidate_batch(self,state_text,candidates):
        rows=[self.tokenizer.encode(state_text+"\nCandidate: "+c["text"],add_special_tokens=False) for c in candidates]
        if not rows or len(rows)<2:raise ValueError("at least two candidates required")
        width=next((b for b in (512,1024,2048) if b<=self.context and b>=max(map(len,rows))),None)
        if width is None:raise ValueError("candidate context capacity exhausted")
        count=next((c for c in (2,4,8,16) if c>=len(rows)),None)
        if count is None:raise ValueError("candidate count exceeds validated buckets")
        lengths=[len(r) for r in rows]+[1]*(count-len(rows))
        ids=[r+[self.pad]*(width-len(r)) for r in rows]+[[self.pad]*width for _ in range(count-len(rows))]
        valid=[[True]*len(rows)+[False]*(count-len(rows))]
        return jnp.array(ids,dtype=jnp.int32),jnp.array(lengths),jnp.array(valid),sum(map(len,rows))

    def decision_logps(self,state,text,candidates):
        ids,lengths,valid,cost=self.candidate_batch(text,candidates)
        logits=self._head(state,self.frozen,ids,lengths,valid)[0,:len(candidates)]
        assert_finite(logits,"decision-logits")
        return jax.nn.log_softmax(logits),cost

    def decision_probs(self,state,text,candidates):
        logps,cost=self.decision_logps(state,text,candidates)
        return jnp.exp(logps),cost

    def token_logps(self,state,prompt,completion):
        values=prompt+completion
        width=next((b for b in (512,1024,2048) if b>=len(values)),None)
        if width is None:raise ValueError("likelihood context capacity exhausted")
        ids=jnp.array([values+[self.pad]*(width-len(values))],dtype=jnp.int32)
        valid=jnp.arange(width)<len(values)
        mask=valid[None,None,:] & (jnp.arange(width)[:,None]>=jnp.arange(width)[None,:])[None,:,:]
        positions=jnp.arange(width)[None,:]
        projection=next(b for b in (32,64,128,256,512,1024,2048) if b>=len(completion))
        selected=jnp.minimum(len(prompt)-1+jnp.arange(projection),width-1)
        logits,_=self._call(state,self.frozen,ids,positions,mask,None,selected)
        return jax.nn.log_softmax(logits[0,:len(completion)].astype(jnp.float32))[jnp.arange(len(completion)),jnp.asarray(completion)]

    def event_logps(self,state,event):
        if event["kind"]=="tokens":return self.token_logps(state,event["prompt"],event["completion"])
        ids,lengths,valid,_=self.candidate_batch(event["state"],event["candidates"])
        logits=self._head(state,self.frozen,ids,lengths,valid)[0,:len(event["candidates"])]
        return jax.nn.log_softmax(logits)[event["choice"]].reshape(1)

    def packed_logps(self,state,packed):
        selected=jnp.asarray(packed["selected"])
        logits,_=self._call(state,self.frozen,jnp.asarray(packed["ids"]),jnp.asarray(packed["positions"]),jnp.asarray(packed["mask"]),None,selected)
        values=jnp.take_along_axis(jax.nn.log_softmax(logits.astype(jnp.float32)),jnp.asarray(packed["targets"])[...,None],axis=-1)[...,0]
        return jnp.where(jnp.asarray(packed["target_mask"]),values,0.)

    def decision_events_logps(self,state,events):
        inputs=[self.candidate_batch(e["state"],e["candidates"]) for e in events]
        shapes={(x[0].shape,x[2].shape) for x in inputs}
        if len(shapes)!=1:raise ValueError("decision microbatch must share count/length bucket")
        ids=jnp.concatenate([x[0] for x in inputs]);lengths=jnp.concatenate([x[1] for x in inputs]);valid=jnp.concatenate([x[2] for x in inputs])
        logits=self._head(state,self.frozen,ids,lengths,valid)
        return jax.nn.log_softmax(logits)[jnp.arange(len(events)),jnp.asarray([e["choice"] for e in events])]

    def generate(self,state,prompts,keys,sample=True,forced=None,carries=None):
        # The pinned Qwen cache writes all rows at end_index[0]. An exact-length
        # group is therefore required, even when padded allocation shapes match.
        groups={}
        for index,prompt in enumerate(prompts):
            carry=carries[index] if carries is not None else None
            actual=prompt+([forced] if forced is not None else [])
            prefix=len(carry['prefix']) if carry is not None and actual[:len(carry['prefix'])]==carry['prefix'] else 0
            groups.setdefault((len(prompt),prefix),[]).append(index)
        completions=[None]*len(prompts);logps=[None]*len(prompts)
        for indices in groups.values():
            positions=jnp.asarray(indices)
            local_carry=[carries[i] for i in indices] if carries is not None else None
            cs,ls,ks=self._generate_equal(state,[prompts[i] for i in indices],keys[positions],sample,forced,local_carry)
            if carries is not None:
                for local,index in enumerate(indices):carries[index]=local_carry[local]
            keys=keys.at[positions].set(ks)
            for offset,index in enumerate(indices):completions[index],logps[index]=cs[offset],ls[offset]
        return completions,logps,keys

    def _generate_equal(self,state,prompts,keys,sample=True,forced=None,carries=None):
        if forced is not None:prompts=[p+[forced] for p in prompts]
        lengths=np.array(list(map(len,prompts)),dtype=np.int32)
        if len(set(lengths.tolist()))!=1:raise ValueError("Tunix cache requires an exact-length group")
        if int(lengths.max())>=self.context:raise ValueError("context capacity exhausted")
        prefix=0
        if carries is not None and all(c is not None for c in carries):
            sizes=[len(c['prefix']) for c in carries]
            if len(set(sizes))==1 and all(p[:sizes[0]]==c['prefix'] for p,c in zip(prompts,carries)):prefix=sizes[0]
        remaining=int(lengths.max())-prefix
        width=next(b for b in (512,1024,2048) if b>=remaining)
        # Reuse only when the whole padded write fits. Near the boundary a fresh
        # prefill preserves context and avoids shifted dynamic_update_slice writes.
        if not remaining or prefix+width>self.context:prefix=0;width=next(b for b in (512,1024,2048) if b>=int(lengths.max()))
        batch=len(prompts);offsets=width-lengths
        if prefix:
            remaining=int(lengths.max())-prefix
            ids=jnp.asarray([p[prefix:]+[self.pad]*(width-remaining) for p in prompts],dtype=jnp.int32)
            positions=jnp.broadcast_to(prefix+jnp.arange(width)[None,:],ids.shape)
            mask=(jnp.arange(self.context)[None,None,:]<=positions[:,:,None]) & (jnp.arange(self.context)[None,None,:]<lengths[:,None,None])
            cache={name:{field:jnp.concatenate([c['cache'][name][field] for c in carries]) for field in ('k','v','end_index')} for name in carries[0]['cache']}
            logits,cache=self._call(state,self.frozen,ids,positions,mask,cache,jnp.asarray([remaining-1]))
            cache={name:{**c,'end_index':jnp.asarray(lengths,dtype=jnp.int32)} for name,c in cache.items()}
        else:
            ids=jnp.array([[self.pad]*int(width-len(p))+p for p in prompts],dtype=jnp.int32)
            valid=jnp.arange(width)[None,:]>=jnp.asarray(offsets)[:,None]
            positions=jnp.maximum(jnp.cumsum(valid,axis=1)-1,0)
            k=jnp.arange(self.context);q=jnp.arange(width)
            valid_keys=(k[None,:]>=jnp.asarray(offsets)[:,None])&(k[None,:]<width)
            mask=valid_keys[:,None,:]&(q[:,None]>=k[None,:])[None,:,:]
            mask=mask | ((~valid)[:,:,None]&(q[:,None]==k[None,:])[None,:,:])
            cache=self.cache(state,batch)
            logits,cache=self._call(state,self.frozen,ids,positions,mask,cache,jnp.array([width-1]))
            # Compact padded prefill slots so buckets never reduce usable context.
            gather=jnp.clip(jnp.arange(self.context)[None,:]+jnp.asarray(offsets)[:,None],0,self.context-1)
            live=jnp.arange(self.context)[None,:]<jnp.asarray(lengths)[:,None]
            cache={name:{"k":jnp.where(live[:,:,None,None],jnp.take_along_axis(c["k"],gather[:,:,None,None],axis=1),0),
                         "v":jnp.where(live[:,:,None,None],jnp.take_along_axis(c["v"],gather[:,:,None,None],axis=1),0),
                         "end_index":jnp.asarray(lengths,dtype=jnp.int32)} for name,c in cache.items()}
        completions=[[forced] if forced is not None else [] for _ in prompts];logps=[[] for _ in prompts]
        finished=np.zeros(batch,dtype=bool);step=0
        while not finished.all():
            scores=logits[:,0].astype(jnp.float32);assert_finite(scores,"rollout-logits")
            scores=scores/self.temperature
            if self.suppress_jev and step==0:scores=scores.at[:,self.mode_ids[0]].set(-1e9)
            if self.top_k is not None:
                cutoff=jax.lax.top_k(scores,self.top_k)[0][:,-1:];scores=jnp.where(scores>=cutoff,scores,-1e9)
            if self.top_p<1.:
                order=jnp.argsort(scores,axis=-1)[:,::-1];sorted_scores=jnp.take_along_axis(scores,order,axis=-1)
                before=jnp.cumsum(jax.nn.softmax(sorted_scores),axis=-1)-jax.nn.softmax(sorted_scores)
                sorted_scores=jnp.where(before<self.top_p,sorted_scores,-1e9)
                scores=jnp.take_along_axis(sorted_scores,jnp.argsort(order,axis=-1),axis=-1)
            distribution=jax.nn.log_softmax(scores);assert_finite(distribution,"rollout-distribution")
            split=jax.vmap(lambda k:jax.random.split(k,2))(keys);keys=split[:,0]
            chosen=jax.vmap(jax.random.categorical)(split[:,1],distribution) if sample else jnp.argmax(distribution,axis=-1)
            token=np.asarray(jax.device_get(chosen));chosen_logp=np.asarray(jax.device_get(distribution[jnp.arange(batch),chosen]))
            for i in range(batch):
                if finished[i]:continue
                completions[i].append(int(token[i]));logps[i].append(float(chosen_logp[i]))
                if token[i]==self.eos or (self.stop_at_jev and step==0 and forced is None and token[i]==self.mode_ids[0]) or lengths[i]+step+1>=self.context:finished[i]=True
            if finished.all():break
            pos=jnp.asarray(lengths+step)[:,None]
            mask=jnp.arange(self.context)[None,None,:]<=pos[:,:,None]
            logits,cache=self._call(state,self.frozen,jnp.asarray(token[:,None]),pos,mask,cache,jnp.array([0]))
            step+=1
        if carries is not None:
            for i in range(batch):
                sampled=completions[i][1:] if forced is not None else completions[i]
                written=prompts[i]+sampled[:-1]
                carries[i]={'prefix':written,'last_prefill_tokens':len(prompts[i])-prefix,'reused_prefix_tokens':prefix,
                            'context_exhausted':len(prompts[i])+len(sampled)>=self.context and sampled[-1]!=self.eos,
                            'cache':{name:{'k':c['k'][i:i+1],'v':c['v'][i:i+1],
                                           'end_index':jnp.asarray([len(written)],dtype=jnp.int32)} for name,c in cache.items()}}
        # Forced modes are controller actions, never falsely logged as sampled tokens.
        return completions,logps,keys

    def cache(self,state,batch):return self.model(state).base.init_cache(batch,self.context,self.model(state).base.config.dtype)

    def active_prompts(self,envs,active,forced_modes=None):
        """End only the episode whose public history cannot fit another token."""
        retained=[];prompts=[]
        for index in active:
            prompt=self.prompt_ids(envs[index].observations())
            forced=(forced_modes or {}).get(index)
            if len(prompt)+int(forced is not None)>=self.context:
                envs[index].finish_error('context capacity exhausted')
            else:retained.append(index);prompts.append(prompt)
        return retained,prompts

    def rollouts(self,state,episodes,key,version,sample=True,policy="learned",threshold=.8):
        envs=[EpisodeEnvV2(e,**self.limits) for e in episodes];traces=[TrajectoryV2(e["id"],version) for e in episodes]
        keys=jax.random.split(key,len(episodes));tokens=[0]*len(episodes);candidate_cost=[0]*len(episodes)
        carries=[None]*len(episodes)
        counters=[{"prefill_calls":0,"prefill_tokens":0,"reused_prefix_tokens":0,"decode_calls":0,"head_calls":0} for _ in episodes];jev_predictions=[[] for _ in episodes]
        started=time.perf_counter()
        for action_index in range(self.limits['max_actions']):
            active=[i for i,e in enumerate(envs) if not e.done]
            if not active:break
            for i in list(active):
                env=envs[i]
                locked_jev=policy=='initial_mode' and env.modes and env.modes[0]==MODES[0]
                if policy in {"always_jev","confidence"} or locked_jev:
                    try:probabilities,cost=self.decision_probs(state,env.state_text(),env.candidates)
                    except ValueError as exc:
                        if 'capacity exhausted' not in str(exc):raise
                        env.finish_error(str(exc));continue
                    candidate_cost[i]+=cost;counters[i]["head_calls"]+=1
                    jev_predictions[i].append([dict(c,probability=float(probabilities[j])) for j,c in enumerate(env.candidates)])
                    choices=[j for j,c in enumerate(env.candidates) if c["kind"]=="answer"]
                    best=max(choices,key=lambda j:float(probabilities[j])) if choices else None
                    selected=int(jnp.argmax(probabilities)) if policy=="always_jev" or locked_jev else best
                    if policy=="always_jev" or locked_jev or (best is not None and float(probabilities[best])>=threshold):
                        env.step(MODES[0],env.candidates[selected]["id"])
                        continue
            active=[i for i in active if not envs[i].done and not (policy=="always_jev") and
                    not (policy=='initial_mode' and envs[i].modes and envs[i].modes[0]==MODES[0])]
            if not active:continue
            forced=self.mode_ids[1] if policy=="always_direct" else self.mode_ids[2] if policy in {"always_cot","confidence"} else None
            forced_modes={i:self.mode_ids[MODES.index(envs[i].modes[0])] if policy=='initial_mode' and envs[i].modes else forced for i in active}
            active,prompts=self.active_prompts(envs,active,forced_modes)
            if not active:continue
            try:
                completions=[None]*len(active);logps=[None]*len(active);new_keys=keys[jnp.asarray(active)]
                locked={}
                for local,i in enumerate(active):
                    mode=forced
                    if policy=='initial_mode' and envs[i].modes:mode=self.mode_ids[MODES.index(envs[i].modes[0])]
                    locked.setdefault(mode,[]).append(local)
                for mode,indices in locked.items():
                    local_carry=[carries[active[j]] for j in indices]
                    cs,ls,ks=self.generate(state,[prompts[j] for j in indices],new_keys[jnp.asarray(indices)],sample,mode,local_carry)
                    for j,offset in enumerate(indices):carries[active[offset]]=local_carry[j]
                    new_keys=new_keys.at[jnp.asarray(indices)].set(ks)
                    for j,index in enumerate(indices):completions[index],logps[index]=cs[j],ls[j]
            except ValueError:
                # Capacity is checked per episode above; other failures are bugs,
                # not policy errors to hide in evaluation or reward records.
                raise
            for local,i in enumerate(active):
                completion=completions[local];keys=keys.at[i].set(new_keys[local])
                sampled_mode=forced is None and not (policy=='initial_mode' and envs[i].modes)
                generated=len(completion)-int(not sampled_mode);tokens[i]+=generated
                counters[i]["prefill_calls"]+=1;counters[i]["decode_calls"]+=max(0,generated-1)
                counters[i]['prefill_tokens']+=carries[i]['last_prefill_tokens'];counters[i]['reused_prefix_tokens']+=carries[i]['reused_prefix_tokens']
                text=self.tokenizer.decode(completion,skip_special_tokens=False).removesuffix("<|im_end|>").strip()
                if sampled_mode:traces[i].tokens(prompts[local],completion,logps[local],envs[i].observations())
                choice=None
                if completion and completion[0]==self.mode_ids[0]:
                    candidates=envs[i].candidates;state_text=envs[i].state_text()
                    try:choice_logps,cost=self.decision_logps(state,state_text,candidates)
                    except ValueError as exc:
                        if 'capacity exhausted' not in str(exc):raise
                        envs[i].finish_error(str(exc));continue
                    candidate_cost[i]+=cost;counters[i]["head_calls"]+=1
                    jev_predictions[i].append([dict(c,probability=float(jnp.exp(choice_logps[j]))) for j,c in enumerate(candidates)])
                    next_key,draw_key=jax.random.split(keys[i]);keys=keys.at[i].set(next_key)
                    selected=int(jax.random.categorical(draw_key,choice_logps)) if sample else int(jnp.argmax(choice_logps))
                    choice=candidates[selected]["id"]
                    traces[i].decision(state_text,candidates,selected,float(choice_logps[selected]))
                envs[i].step(text,choice)
                if envs[i].error and carries[i]['context_exhausted']:envs[i].finish_error('context capacity exhausted')
        jax.block_until_ready(keys);elapsed=time.perf_counter()-started
        results=[]
        for i,env in enumerate(envs):
            if not env.done:env.finish_error("action limit")
            reward,info=env.reward(tokens[i],candidate_cost[i]);traces[i].finalize(reward,info["reward_components"],env.error or "answer")
            results.append({"schema_version":2,"id":episodes[i]["id"],"group":episodes[i]["group"],"behavior":episodes[i]["behavior"],
                            "answer":env.answer,"reward":reward,**info,"tokens":tokens[i],"candidate_encoding_tokens":candidate_cost[i],
                            **counters[i],"modes":env.modes,"tool_calls":env.tool_calls,"batch_seconds":elapsed,"batch_size":len(episodes),
                            "actions":[{**event,"action":{k:v for k,v in event["action"].items() if k!="text"}} for event in env.events],
                            "observed_evidence":[m["content"] for m in env.messages[2:] if m["role"]=="user" and (m["content"].startswith("Observed ") or m["content"].startswith("Lookup evidence:"))],
                            "jev_predictions":jev_predictions[i],"trajectory":traces[i]})
        return results
