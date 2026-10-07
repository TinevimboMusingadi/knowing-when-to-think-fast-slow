"""Custom mixed-action learner on Tunix Qwen/NNX and standard Optax.

Upstream text-only GRPOLearner cannot represent categorical head actions. This
learner preserves those probabilities and uses explicit update acceptance.
"""
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .tunix_optim import optimizer,checked_update,advantages,grpo_loss,assert_finite,NumericalFailure,add_gradients
from .episode_v2 import EpisodeEnvV2
from .protocol import MODES
from .tunix_distributed import Replicas
from .packing_v2 import packs,batch_packs


class MixedLearner:
    def __init__(self,runtime,config,phase,output):
        self.runtime=runtime;self.config=config;self.phase=phase;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.actor=runtime.actor
        self.reference=jax.tree.map(lambda x:jnp.array(x,copy=True),self.actor)
        self.tx=optimizer(self.actor,config[f"{phase}_learning_rate"],config["warmup_steps"])
        self.optimizer_state=self.tx.init(self.actor)
        self.key=jax.random.PRNGKey(config["seed"]);self.accepted_updates=0;self.cursor=0
        self.warmed_times=[];self.cold_times=[]
        self.replicas=Replicas(runtime)

    def metric(self,event):
        with (self.output/"metrics.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps({**event,"timestamp":time.time(),"phase":self.phase})+"\n")

    def state(self):
        return {"actor":self.actor,"reference":self.reference,"optimizer":self.optimizer_state,"rng":self.key,
                "cursor":self.cursor,"accepted_updates":self.accepted_updates,
                "scheduler":{"count":self.accepted_updates,"learning_rate":self.config[f"{self.phase}_learning_rate"],"warmup_steps":self.config["warmup_steps"]}}

    def restore(self,state):
        self.actor=state["actor"];self.reference=state["reference"];self.optimizer_state=state["optimizer"]
        self.key=state["rng"];self.cursor=int(state["cursor"]);self.accepted_updates=int(state["accepted_updates"])

    def apply(self,gradients,reduce=None):
        try:actor,state,info=checked_update(self.actor,self.optimizer_state,gradients,self.tx,reduce)
        except NumericalFailure as exc:
            self.metric({"event":"update-refused","stage":exc.stage,"tensors":exc.tensors,"accepted_updates":self.accepted_updates})
            raise
        self.actor,self.optimizer_state=actor,state;self.accepted_updates+=1
        self.metric({"event":"update-accepted","accepted_updates":self.accepted_updates,**info})

    def supervised_examples(self,rows):
        examples=[]
        for row in rows:
            env=EpisodeEnvV2(row)
            for step in row["supervision"]:
                completion=step["text"]+("" if step["text"]==MODES[0] else "<|im_end|>")
                tokens=self.runtime.tokenizer.encode(completion,add_special_tokens=False)
                event={"kind":"tokens","prompt":self.runtime.prompt_ids(env.observations()),"completion":tokens}
                decision=None
                if "choice_id" in step:
                    candidates=env.candidates;index=next(i for i,c in enumerate(candidates) if c["id"]==step["choice_id"])
                    decision={"kind":"decision","state":env.state_text(),"candidates":candidates,"choice":index}
                examples.append((event,decision));env.step(step["text"],step.get("choice_id"))
        return examples

    def sft_step(self,rows):
        examples=self.supervised_examples(rows);start=time.perf_counter()
        inventory=batch_packs(packs([e for e,_ in examples],self.runtime.pad),self.runtime.pad,self.config.get('microbatch',1))
        useful=sum(p['useful_tokens'] for p in inventory);allocation=sum(p['allocation_tokens'] for p in inventory)
        candidate_tokens=sum(self.runtime.candidate_batch(d['state'],d['candidates'])[3] for _,d in examples if d is not None)
        # Accumulate per-event gradients to avoid retaining a whole episode batch graph.
        language_count=sum(len(e["completion"]) for e,_ in examples);decision_count=sum(d is not None for _,d in examples)
        world=len(self.replicas.devices)
        def work(runtime,actor,reference,batch,key):
            gradients=jax.tree.map(jnp.zeros_like,actor);total=0.
            microbatch=self.config.get("microbatch",1)
            for packed in batch_packs(packs([e for e,_ in batch],runtime.pad),runtime.pad,microbatch):
                def loss(params):
                    return -runtime.packed_logps(params,packed).sum()*.5*world/language_count
                value,grad=jax.value_and_grad(loss)(actor);assert_finite(value,"sft-loss")
                total+=float(value);gradients=add_gradients(gradients,grad)
            grouped={}
            for _,decision in batch:
                if decision is not None:
                    ids,_,valid,_=runtime.candidate_batch(decision["state"],decision["candidates"])
                    grouped.setdefault((ids.shape,valid.shape),[]).append(decision)
            for decisions in grouped.values():
                for offset in range(0,len(decisions),microbatch):
                    selected=decisions[offset:offset+microbatch]
                    loss=lambda params:-runtime.decision_events_logps(params,selected).sum()*.5*world/max(1,decision_count)
                    value,grad=jax.value_and_grad(loss)(actor);assert_finite(value,"sft-decision-loss")
                    total+=float(value);gradients=add_gradients(gradients,grad)
            return gradients,total
        self.key,draw=jax.random.split(self.key)
        results=self.replicas.map(work,self.actor,self.reference,[examples[i::world] for i in range(world)],jax.random.split(draw,world))
        gradients=self.replicas.mean([result[0] for result in results]);total=sum(result[1] for result in results)/world
        self.apply(gradients);self.cursor+=len(rows);seconds=time.perf_counter()-start
        self.metric({"event":"sft-step","loss":total,"episodes":len(rows),"seconds":seconds,"devices":world,
                    'useful_tokens':useful,'allocation_tokens':allocation,'padding_fraction':1-useful/allocation,
                    'useful_tokens_per_second':useful/seconds,'candidate_encoding_tokens':candidate_tokens,
                    'cost_upper_bound_per_step':seconds/3600*self.config.get('rate_upper_bound',4.8)*self.config.get('rate_margin',1.15)})
        return total

    def rl_step(self,rows):
        started=time.perf_counter();world=len(self.replicas.devices);self.key,draw=jax.random.split(self.key)
        def work(runtime,actor,reference,local_rows,key):
            all_samples=[];groups=[]
            for row in local_rows:
                key,draw=jax.random.split(key)
                samples=runtime.rollouts(actor,[row]*self.config["group_size"],draw,f"rl-{self.accepted_updates}",sample=True)
                adv=advantages([s["reward"] for s in samples]);groups.append(float(jnp.std(jnp.array([s["reward"] for s in samples])))>1e-8)
                all_samples.extend(zip(samples,adv))
            return self._group_gradients(runtime,actor,reference,all_samples,world/(len(rows)*self.config["group_size"])),all_samples,groups
        results=self.replicas.map(work,self.actor,self.reference,[rows[i::world] for i in range(world)],jax.random.split(draw,world))
        gradients=self.replicas.mean([r[0][0] for r in results]);total=sum(r[0][1] for r in results)/world
        tail_max=max(r[0][2] for r in results);all_samples=[s for r in results for s in r[1]];groups=[g for r in results for g in r[2]]
        self.apply(gradients);self.cursor+=len(rows)
        for sample,_ in all_samples:self.metric({"event":"trajectory","record":sample["trajectory"].public_record(),"behavior":sample["behavior"],"error":sample["error"]})
        seconds=time.perf_counter()-started
        self.metric({"event":"rl-step","loss":total,"reward":float(np.mean([s["reward"] for s,_ in all_samples])),"seconds":seconds,
                     "varied_reward_fraction":sum(groups)/max(1,len(groups)),"max_log_ratio":tail_max,"devices":world,
                     "generated_tokens":sum(s["tokens"] for s,_ in all_samples),"candidate_encoding_tokens":sum(s["candidate_encoding_tokens"] for s,_ in all_samples)})
        return {"seconds":seconds,"varied_reward_fraction":sum(groups)/max(1,len(groups))}

    def _group_gradients(self,runtime,actor,reference_state,all_samples,weight):
        gradients=jax.tree.map(jnp.zeros_like,actor);total=0.;tail_max=0.
        for sample,advantage in all_samples:
            trace=sample["trajectory"]
            if not trace.events:continue
            old=jnp.concatenate([jnp.asarray(event["old_logps"],dtype=jnp.float32) for event in trace.events])
            reference=jnp.concatenate([runtime.event_logps(reference_state,event) for event in trace.events])
            current=jnp.concatenate([runtime.event_logps(actor,event) for event in trace.events])
            for value,stage in ((old,"old-logps"),(reference,"reference-logps"),(current,"policy-logps")):assert_finite(value,stage)
            tail=float(jnp.maximum(jnp.max(jnp.abs(current-old)),jnp.max(jnp.abs(reference-current))))
            tail_max=max(tail_max,tail)
            if tail>20:raise NumericalFailure("likelihood-tail",[{"max_abs_log_ratio":tail}])
            def loss(params):
                new=jnp.concatenate([runtime.event_logps(params,event) for event in trace.events])
                return grpo_loss(new,old,reference,advantage,self.config["clip_epsilon"],self.config["kl_coefficient"])*weight
            value,grad=jax.value_and_grad(loss)(actor);assert_finite(value,"rl-loss")
            gradients=add_gradients(gradients,grad);total+=float(value)
        return gradients,total,tail_max

    def close(self):self.replicas.close()
