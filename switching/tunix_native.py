"""External Qwen baselines with native chat templates and the original LM head."""
import dataclasses
import json
import time
import jax
import jax.numpy as jnp
from flax import nnx
from .tunix_runtime import TunixRuntime
from .episode_v2 import EpisodeEnvV2
from .protocol import MODES

NATIVE_SYSTEM="""Solve the task using observed facts. You may request an available missing field or use an available lookup query. Respond with one JSON action: {"action":"answer","value":...}, {"action":"ask","field":...,"question":...}, or {"action":"lookup","query":...}. After a tool response continue the same task. Do not invent missing information."""


class NativeRuntime(TunixRuntime):
    def __init__(self,trained_runtime,tokenizer,thinking):
        module=trained_runtime.model(trained_runtime.actor);base=module.base
        base.embedder=base.embedder.original
        for layer in base.layers:
            for name in ("q_proj","k_proj","v_proj","o_proj"):setattr(layer.attn,name,getattr(layer.attn,name).original)
        base.config=dataclasses.replace(base.config,vocab_size=base.embedder.input_embedding.value.shape[0])
        self.graph,self.actor=nnx.split(base);self.frozen=nnx.State({});self.tokenizer=tokenizer
        self.context=trained_runtime.context;self.thinking=thinking;self.mode_ids=[-1,-1,-1]
        self.limits=dict(trained_runtime.limits)
        self.eos=tokenizer.convert_tokens_to_ids("<|im_end|>");self.pad=tokenizer.pad_token_id
        self.temperature=.6 if thinking else .7;self.top_p=.95 if thinking else .8;self.top_k=20
        self.suppress_jev=False;self.stop_at_jev=False;self._call=jax.jit(self._language);self._head=None

    def _language(self,state,frozen,ids,positions,mask,cache,select):
        base=self.model(state,frozen);hidden,cache=base(ids,positions,cache,mask,skip_lm_head=True)
        return base.compute_final_logits(hidden[:,select]),cache

    def prompt_ids(self,messages):
        return self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=self.thinking)

    def rollouts(self,state,episodes,key,version,sample=True,policy="native",threshold=.8):
        envs=[EpisodeEnvV2(e,**self.limits) for e in episodes];keys=jax.random.split(key,len(episodes))
        for env in envs:env.messages[0]["content"]=NATIVE_SYSTEM
        tokens=[0]*len(envs);calls=[0]*len(envs);carries=[None]*len(envs);prefill=[0]*len(envs);reused=[0]*len(envs);started=time.perf_counter()
        for _ in range(self.limits['max_actions']):
            active=[i for i,e in enumerate(envs) if not e.done]
            if not active:break
            active,prompts=self.active_prompts(envs,active)
            if not active:continue
            try:
                local_carry=[carries[i] for i in active]
                outputs,_,new_keys=self.generate(state,prompts,keys[jnp.asarray(active)],sample,carries=local_carry)
                for local,i in enumerate(active):carries[i]=local_carry[local];prefill[i]+=carries[i]['last_prefill_tokens'];reused[i]+=carries[i]['reused_prefix_tokens']
            except ValueError:raise
            for local,i in enumerate(active):
                keys=keys.at[i].set(new_keys[local]);tokens[i]+=len(outputs[local]);calls[i]+=1
                text=self.tokenizer.decode(outputs[local],skip_special_tokens=False).removesuffix("<|im_end|>").strip()
                native_text=text
                if "</think>" in text:text=text.split("</think>",1)[1].strip()
                try:
                    action=json.loads(text);envs[i].step((MODES[2] if self.thinking else MODES[1])+json.dumps(action))
                    # Mode annotations belong to the verifier, not native Qwen's history.
                    for message in reversed(envs[i].messages):
                        if message['role']=='assistant':message['content']=native_text;break
                except (ValueError,TypeError):envs[i].finish_error('context capacity exhausted' if carries[i]['context_exhausted'] else "native action JSON invalid")
        jax.block_until_ready(keys);elapsed=time.perf_counter()-started;results=[]
        for i,env in enumerate(envs):
            if not env.done:env.finish_error("action limit")
            reward,info=env.reward(tokens[i])
            results.append({"schema_version":2,"id":episodes[i]["id"],"group":episodes[i]["group"],"behavior":episodes[i]["behavior"],
                            "answer":env.answer,"reward":reward,**info,"tokens":tokens[i],"candidate_encoding_tokens":0,
                            "prefill_calls":calls[i],"prefill_tokens":prefill[i],"reused_prefix_tokens":reused[i],"decode_calls":max(0,tokens[i]-calls[i]),"head_calls":0,"tool_calls":env.tool_calls,
                            "modes":env.modes,"batch_seconds":elapsed,"batch_size":len(envs),"external_baseline":True,"jev_predictions":[]})
        return results

    def cache(self,state,batch):return self.model(state).init_cache(batch,self.context,self.model(state).config.dtype)
