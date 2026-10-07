"""Local production-path checks for caches, masks and four-device collectives."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .tunix_load import load_fixture
from .tunix_runtime import TunixRuntime
from .tunix_distributed import Replicas
from .tunix_optim import checked_update,optimizer,grpo_loss,advantages,assert_finite
from .packing_v2 import packs,batch_packs
from .experiment_v2 import revisions,sha


class DiagnosticTokenizer:
    """Fixed token IDs for a numerical fixture, never a learned demonstration."""
    pad_token_id=0
    def convert_tokens_to_ids(self,text):
        return {'<mode:jev>':64,'<mode:direct>':65,'<mode:cot>':66,'<|im_end|>':15}[text]
    def encode(self,text,**kwargs):return [2,3,4]+[5+(b%8) for b in hashlib.sha256(text.encode()).digest()[:3]]


def validate(folder,output,development=False):
    from .tunix_environment import require_unchanged
    root=Path(__file__).resolve().parents[1];cfg=json.loads((root/'configs/recovery.json').read_text());checks=[]
    report={'schema_version':2,'stage':'local-protocol','passed':False,'tunix_revision':cfg['tunix_revision'],
            'dataset_manifest_sha256':sha(root/'data/recovery-v2/manifest.json'),'revisions':revisions(root),'checks':checks}
    started=time.perf_counter();replicas=None
    try:
        from .tunix_environment import verify
        if development:report['production_acceptance']=False
        else:report['environment']=verify(root,jax.default_backend());report['production_acceptance']=True
        if len(jax.local_devices())!=4:raise ValueError('four local CPU devices required: set XLA_FLAGS before Python starts')
        model,_=load_fixture(folder);runtime=TunixRuntime(model,DiagnosticTokenizer(),512);actor=runtime.actor
        runtime.stop_at_jev=False
        prompt=[2,3,4]*169+[5,6]
        completion,old,_=runtime.generate(actor,[prompt],jax.random.split(jax.random.PRNGKey(42),1),sample=False)
        actual=runtime.token_logps(actor,prompt,completion[0])
        np.testing.assert_allclose(actual,old[0],atol=1e-5,rtol=1e-4)
        checks.append({'name':'cached-and-recomputed-likelihoods','passed':True,'tokens':len(completion[0]),'context_capacity':512})
        # Force an immediate JEV only in a diagnostic fixture, so a second action
        # can exercise prefix reuse without imposing an output allowance.
        fresh,_=load_fixture(folder);short=[2,3,4]*60
        h=fresh.hidden(jnp.asarray([short]),jnp.arange(len(short))[None,:],jnp.tril(jnp.ones((1,len(short),len(short)),dtype=bool)))[0][0,-1]
        fresh.base.embedder.rows.value=fresh.base.embedder.rows.value.at[0].set(h*100)
        reuse=TunixRuntime(fresh,DiagnosticTokenizer(),2048);carry=[None]
        first,_,_=reuse.generate(reuse.actor,[short],jax.random.split(jax.random.PRNGKey(2),1),sample=False,carries=carry)
        if first[0]!=[64]:raise ValueError('diagnostic JEV setup did not stop immediately')
        following=short+[64,15,2,3]
        second,prob,_=reuse.generate(reuse.actor,[following],jax.random.split(jax.random.PRNGKey(3),1),sample=False,carries=carry)
        uncached,expected_prob,_=reuse.generate(reuse.actor,[following],jax.random.split(jax.random.PRNGKey(3),1),sample=False)
        if second!=uncached or carry[0]['reused_prefix_tokens']!=len(short):raise ValueError('episode prefix reuse failed')
        np.testing.assert_allclose(prob[0],expected_prob[0],atol=1e-5,rtol=1e-4)
        checks.append({'name':'cross-action-cache-reuse-and-2048-capacity','passed':True,'prefill_tokens':carry[0]['last_prefill_tokens'],
                       'reused_prefix_tokens':carry[0]['reused_prefix_tokens'],'learned_transition_claim':False})
        events=[{'prompt':[2,3,4],'completion':[64,6,7]},{'prompt':[8,9,10,11],'completion':[65,12,13,14]}]
        packed=batch_packs(packs(events,runtime.pad),runtime.pad)[0]
        joint=runtime.packed_logps(actor,packed)[packed['target_mask']]
        individual=jnp.concatenate([runtime.token_logps(actor,e['prompt'],e['completion']) for e in sorted(events,key=lambda e:len(e['prompt'])+len(e['completion']),reverse=True)])
        np.testing.assert_allclose(joint,individual,atol=1e-5,rtol=1e-4)
        altered={**packed,'ids':packed['ids'].copy()};altered['ids'][0,7:13]=[13,14,2,3,4,5]
        # The longer example occupies positions 0..7; another block cannot affect it.
        preserved=runtime.packed_logps(actor,altered)[0,:4]
        np.testing.assert_allclose(preserved,joint[:4],atol=1e-5,rtol=1e-4)
        masked=jax.grad(lambda params:runtime.packed_logps(params,packed)[~packed['target_mask']].sum())(actor)
        if any(bool(jnp.any(v!=0)) for v in jax.tree.leaves(masked)):raise ValueError('padding contributed policy gradients')
        checks.append({'name':'packed-isolation-reset-positions-assistant-masking','passed':True,'prompt_labels':0,'padding_gradients':0})
        replicas=Replicas(runtime)
        gradients=[]
        for i,device in enumerate(replicas.devices):
            with jax.default_device(device):gradients.append(jax.tree.map(lambda x:jnp.ones_like(x)*(i+1),actor))
        reduced=replicas.mean(gradients)
        for value in jax.tree.leaves(reduced):np.testing.assert_array_equal(value,np.full(value.shape,2.5,dtype=np.float32))
        checks.append({'name':'four-device-fp32-gradient-average','passed':True,'device_count':4})
        # Diagnostic mixed probabilities use actual model likelihoods; no invented choice token.
        candidates=[{'id':'a','text':'5','kind':'answer','value':5},{'id':'b','text':'6','kind':'answer','value':6}]
        reference=actor;tx=optimizer(actor,5e-6,5);state=tx.init(actor)
        for update in range(3):
            token_event={'kind':'tokens','prompt':[2,3,4],'completion':[64,6,7]}
            decision_event={'kind':'decision','state':'Choose a value.','candidates':candidates,'choice':update%2}
            old=jnp.concatenate([runtime.event_logps(actor,e) for e in (token_event,decision_event)])
            ref=jnp.concatenate([runtime.event_logps(reference,e) for e in (token_event,decision_event)])
            def loss(params):
                new=jnp.concatenate([runtime.event_logps(params,e) for e in (token_event,decision_event)])
                return grpo_loss(new,old,ref,jnp.array(.5,dtype=jnp.float32))
            grad=jax.grad(loss)(actor);actor,state,_=checked_update(actor,state,grad,tx)
        assert_finite((actor,state),'mixed-diagnostic-state')
        if any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(reference),jax.tree.leaves(runtime.actor))):raise ValueError('reference mutated')
        checks.append({'name':'three-mixed-action-diagnostic-updates','passed':True,'accepted_updates':3,
                       'sampled_rollout_claim':False,'reference_unchanged':True})
        bf16_model,_=load_fixture(folder,bf16=True,remat=True);bf16_runtime=TunixRuntime(bf16_model,DiagnosticTokenizer(),512)
        def bf16_loss(params):
            return -bf16_runtime.token_logps(params,[2,3,4],[64,5,6]).mean()-bf16_runtime.event_logps(params,
                    {'kind':'decision','state':'Choose a value.','candidates':candidates,'choice':0}).mean()
        gradient=jax.grad(bf16_loss)(bf16_runtime.actor);assert_finite(gradient,'bf16-remat-gradient')
        if any(v.dtype!=jnp.float32 for v in jax.tree.leaves(gradient)):raise ValueError('BF16 forward reduced trainable gradient precision')
        tx=optimizer(bf16_runtime.actor,5e-6);checked_update(bf16_runtime.actor,tx.init(bf16_runtime.actor),gradient,tx)
        checks.append({'name':'bf16-frozen-forward-rematerialization-fp32-update','passed':True})
        from .tunix_train import MixedLearner
        from .data_v2 import make
        import tempfile
        learner=MixedLearner(runtime,{**cfg,'microbatch':1},'sft',tempfile.mkdtemp(prefix='kws-sft-diagnostic-'))
        try:
            before=learner.actor
            loss=learner.sft_step([make('train','fast',i) for i in range(32)])
            if learner.accepted_updates!=1 or not np.isfinite(loss):raise ValueError('distributed supervised update did not complete')
            if not any(bool(jnp.any(a!=b)) for a,b in zip(jax.tree.leaves(before),jax.tree.leaves(learner.actor))):raise ValueError('supervised step made no parameter change')
            checks.append({'name':'actual-four-device-supervised-learner-update','passed':True,'episodes':32,
                           'accepted_updates':1,'diagnostic_tokenizer':True,'trained_capability_claim':False})
        finally:learner.close()
        class CapacityEnv:
            def __init__(self,length):self.prompt=list(range(length));self.error=None
            def observations(self):return self.prompt
            def finish_error(self,error):self.error=error
        capacity=TunixRuntime.__new__(TunixRuntime);capacity.context=512;capacity.prompt_ids=lambda messages:messages
        envs=[CapacityEnv(512),CapacityEnv(8),CapacityEnv(511)]
        retained,prompts=capacity.active_prompts(envs,[0,1,2],{2:65})
        if retained!=[1] or prompts!=[envs[1].prompt] or envs[1].error is not None or not all(envs[i].error=='context capacity exhausted' for i in (0,2)):
            raise ValueError('context exhaustion contaminated another episode')
        checks.append({'name':'per-episode-context-exhaustion-isolation','passed':True,'retained_episodes':1,'exhausted_episodes':2})
        require_unchanged(root,report);report['passed']=True
    except Exception as exc:report['error']={'type':type(exc).__name__,'message':str(exc)};raise
    finally:
        if replicas:replicas.close()
        report['seconds']=time.perf_counter()-started;Path(output).write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fixture',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    jax.config.update('jax_default_matmul_precision','highest')
    print(json.dumps(validate(args.fixture,args.output),indent=2))
