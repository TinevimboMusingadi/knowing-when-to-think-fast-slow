"""Fail-closed local acceptance report; no paid resources and no skipped checks."""
import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from .tunix_load import load_fixture
from .tunix_model import Trainable,decision_head
from .tunix_optim import optimizer,checked_update,grpo_loss,advantages,assert_finite,NumericalFailure
from .checkpoint_v2 import CheckpointsV2


def validate(folder,output,development=False):
    from .tunix_environment import fingerprint,require_unchanged
    root=Path(__file__).resolve().parents[1];cfg=json.loads((root/'configs/recovery.json').read_text())
    folder=Path(folder);metadata=json.loads((folder/"fixture.json").read_text());checks=[];started=time.time()
    report={"schema_version":2,"stage":"local-numerical","passed":False,"checks":checks,"started_at":started,
            "model":metadata["model"],"tolerances":{"atol":1e-5,"rtol":1e-4,"full_model_probability_atol":1e-3},
            "devices":[str(d) for d in jax.devices()],"fixture_sha256":hashlib.sha256((folder/"fixture.json").read_bytes()).hexdigest(),
            'tunix_revision':cfg['tunix_revision'],**fingerprint(root)}
    def checked(name,function):
        result=function();checks.append({"name":name,"passed":True,"detail":result})
    try:
        from .tunix_environment import verify
        if development:report['production_acceptance']=False
        else:report['environment']=verify(Path(__file__).resolve().parents[1],jax.default_backend());report['production_acceptance']=True
        for name,digest in metadata["files"].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError(f"fixture checksum mismatch: {name}")
        model,_=load_fixture(folder);graph,actor,frozen=model.split()
        expected=np.load(folder/"expected.npz",allow_pickle=False)
        ids=jnp.asarray(expected["ids"],dtype=jnp.int32);width=ids.shape[1]
        positions=jnp.arange(width)[None,:];mask=jnp.tril(jnp.ones((1,width,width),dtype=bool))
        def call(params):return nnx.merge(graph,params,frozen)(ids,positions,mask)[0]
        logits=call(actor);hidden=model.hidden(ids,positions,mask)[0]
        def agree(name,actual,target,probability=False):
            actual=np.asarray(jax.device_get(actual));target=np.asarray(target)
            if probability and metadata["model"]!="tiny-reference":np.testing.assert_allclose(actual,target,atol=1e-3,rtol=0)
            else:np.testing.assert_allclose(actual,target,atol=1e-5,rtol=1e-4)
            return {"max_absolute_difference":float(np.max(np.abs(actual-target)))}
        checked("hidden-state-parity",lambda:agree("hidden",hidden,expected["hidden"]))
        checked("language-logit-parity",lambda:agree("logits",logits,expected["logits"]))
        logps=jax.nn.log_softmax(logits[0,1:4])[jnp.arange(3),jnp.array([3,4,5])]
        checked("selected-token-log-probability",lambda:agree("logps",logps,expected["selected_logps"]))
        cids=jnp.asarray(expected["candidate_ids"],dtype=jnp.int32);lengths=jnp.asarray(expected["lengths"]);valid=jnp.asarray(expected["valid"])
        candidate=model.candidate_logits(cids,lengths,valid)
        checked("candidate-probability-parity",lambda:agree("probabilities",jax.nn.softmax(candidate),expected["candidate_probabilities"],True))
        permutation=jnp.array([2,0,1,3]);permuted=model.candidate_logits(cids[permutation],lengths[permutation],valid[:,permutation])
        checked("candidate-permutation",lambda:agree("permuted",permuted,candidate[:,permutation]))
        def isolated():
            causal=jnp.tril(jnp.ones((len(cids),cids.shape[1],cids.shape[1]),dtype=bool))&(jnp.arange(cids.shape[1])[None,None,:]<lengths[:,None,None])
            h=model.hidden(cids,jnp.broadcast_to(jnp.arange(cids.shape[1]),cids.shape),causal)[0]
            changed=model.hidden(cids.at[0,2].set(11),jnp.broadcast_to(jnp.arange(cids.shape[1]),cids.shape),causal)[0]
            return agree("isolation",h[1:],changed[1:])
        checked("candidate-backbone-isolation",isolated)
        def candidate_gradients():
            causal=jnp.tril(jnp.ones((len(cids),cids.shape[1],cids.shape[1]),dtype=bool))&(jnp.arange(cids.shape[1])[None,None,:]<lengths[:,None,None])
            pos=jnp.broadcast_to(jnp.arange(cids.shape[1]),cids.shape)
            def reference_loss(params):
                module=nnx.merge(graph,params,frozen)
                hidden,_=module.hidden(cids,pos,causal)
                reps=hidden[jnp.arange(len(cids)),jnp.maximum(lengths-1,0)][None,:]
                scores=decision_head({n:p.value for n,p in module.head.items()},reps,valid)
                return -jax.nn.log_softmax(scores[0,:3])[0]
            target=jax.grad(reference_loss)(actor);maximum=0.
            for chunk in (1,2):
                def scanned_loss(params):
                    module=nnx.merge(graph,params,frozen);module.candidate_microbatch=chunk
                    return -jax.nn.log_softmax(module.candidate_logits(cids,lengths,valid)[0,:3])[0]
                actual=jax.grad(scanned_loss)(actor)
                for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(target)):
                    maximum=max(maximum,agree('candidate-gradient',a,b)['max_absolute_difference'])
            return {'encoding_microbatches':[1,2],'maximum_absolute_gradient_difference':maximum,'reference':'independent batched backbone branches'}
        checked('candidate-scan-gradient-parity',candidate_gradients)
        original=np.load(folder/"trainables.npz",allow_pickle=False)
        def roundtrip():
            restored={}
            for name in original.files:
                if name.endswith("rows"):value=model.base.embedder.rows.value
                elif name.startswith("head."):value=model.head[name.removeprefix("head.")].value
                else:
                    import re
                    match=re.search(r"model.layers.(\d+).self_attn.(\w+).lora_([AB])",name)
                    layer=getattr(model.base.layers[int(match[1])].attn,match[2]);value=(layer.a.value if match[3]=="A" else layer.b.value).T
                restored[name]=value;agree(name,value,original[name])
            if any(x.dtype!=jnp.float32 for x in jax.tree.leaves(actor)):raise ValueError("trainables must be FP32")
            if sum(x.size for x in jax.tree.leaves(actor))!=sum(x.size for x in restored.values()):raise ValueError("trainable inventory mismatch")
            return {"trainable_tensors":len(restored),"dtype":"float32"}
        checked("roundtrip-and-inventory",roundtrip)
        def update_reference():
            params={"matrix":jnp.array([[1.,-2.,.3],[.5,.8,-.4]]),"rows":jnp.array([[.1,.2,.3]])};tx=optimizer(params,.001,5);state=tx.init(params)
            targets=np.load(folder/"optimizer-reference.npz")["updates"]
            for i in range(100):
                gradients={"matrix":jnp.array([[.2,-.4,.1],[.7,.8,-.1]])*(1+(i%7)/10),"rows":jnp.array([[.3,-.2,.4]])}
                params,state,_=checked_update(params,state,gradients,tx)
                agree("optimizer",jnp.concatenate((params["matrix"].ravel(),params["rows"].ravel())),targets[i])
            return {"updates":100,"reference":"PyTorch AdamW, FP32"}
        checked("100-optimizer-updates-against-pytorch",update_reference)
        def masks_and_loss():
            # Tokens and categorical actions are separate sampled probabilities.
            token=jnp.array([-.2,-.4]);decision=jnp.array([-.8]);old=jnp.concatenate((token,decision))
            val=grpo_loss(old,old,old,jnp.array(.7))
            agree("joint",val,-.7)
            agree("zero-advantage",advantages([1.,1.,1.,1.]),np.zeros(4))
            return {"sampled_tokens":2,"categorical_actions":1,"zero_group_advantage":True}
        checked("mixed-action-loss-and-zero-variance",masks_and_loss)
        tx=optimizer(actor,1e-4,5);opt=tx.init(actor)
        def diagnostic(params):
            module=nnx.merge(graph,params,frozen)
            lm=module(ids,positions,mask)[0]
            head=module.candidate_logits(cids,lengths,valid)[0,:3]
            return -jax.nn.log_softmax(lm[0,1])[metadata["vocabulary_boundary"]]-jax.nn.log_softmax(head)[0]
        grad=jax.grad(diagnostic)(actor);assert_finite(grad,"diagnostic-gradient")
        updated,new_opt,_=checked_update(actor,opt,grad,tx)
        def changes():
            changed={"adapters":False,"head":False,"rows":False}
            for (path,a),b in zip(jax.tree_util.tree_flatten_with_path(actor)[0],jax.tree.leaves(updated)):
                if bool(jnp.any(a!=b)):
                    name=jax.tree_util.keystr(path);component="head" if "head" in name else "rows" if "rows" in name else "adapters";changed[component]=True
            if not all(changed.values()):raise ValueError(f"missing diagnostic parameter update: {changed}")
            return changed
        checked("all-custom-components-update",changes)
        def checkpoint():
            with tempfile.TemporaryDirectory() as root:
                manager=CheckpointsV2(root)
                state={"actor":updated,"reference":actor,"optimizer":new_opt,"rng":jax.random.PRNGKey(42),"cursor":32,"accepted_updates":1,
                       "scheduler":{"count":1,"learning_rate":1e-4,"warmup_steps":5}}
                path=manager.save("diagnostic",state,{"schema_version":2,"backend":"tunix","model":metadata["model"]})
                restored=manager.load(path,state,{"model":metadata["model"]});manager.close()
                next_a,next_o,_=checked_update(updated,new_opt,grad,tx)
                next_b,next_ob,_=checked_update(restored["actor"],restored["optimizer"],grad,tx)
                for a,b in zip(jax.tree.leaves((next_a,next_o)),jax.tree.leaves((next_b,next_ob))):np.testing.assert_array_equal(a,b)
                for a,b in zip(jax.tree.leaves(actor),jax.tree.leaves(restored["reference"])):np.testing.assert_array_equal(a,b)
            return {"next_update_identical":True,"reference_unchanged":True}
        checked("orbax-restore-next-update",checkpoint)
        def failures():
            bad=jax.tree.map(jnp.zeros_like,actor)
            leaves,structure=jax.tree.flatten(bad);leaves[0]=jnp.full_like(leaves[0],jnp.nan);bad=jax.tree.unflatten(structure,leaves)
            try:checked_update(actor,opt,bad,tx)
            except NumericalFailure as exc:
                if exc.stage!="local-gradients":raise
                return {"stage":exc.stage,"diagnostics_present":bool(exc.tensors)}
            raise AssertionError("NaN update was accepted")
        checked("nonfinite-update-refused",failures)
        require_unchanged(root,report);report["passed"]=True
        # Tiny checks cannot satisfy the separate full-model launch gate.
        report["full_model_probability_parity"]=metadata["model"]!="tiny-reference"
    except Exception as exc:
        report["error"]={"type":type(exc).__name__,"message":str(exc)}
        raise
    finally:
        report["seconds"]=time.time()-started
        target=Path(output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fixture",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    print(json.dumps(validate(a.fixture,a.output),indent=2))
