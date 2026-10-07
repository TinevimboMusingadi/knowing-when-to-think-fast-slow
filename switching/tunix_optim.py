"""Explicit gradient-reduction and Optax acceptance boundaries, all FP32."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import lru_cache


@jax.jit
def add_gradients(left,right):return jax.tree.map(jnp.add,left,right)


@jax.jit
def all_finite(tree):
    leaves=[x for x in jax.tree.leaves(tree) if hasattr(x,'dtype')]
    return jnp.stack([jnp.isfinite(x).all() for x in leaves]).all() if leaves else jnp.array(True)


@jax.jit
def all_nonnegative(tree):return jnp.stack([(x>=0).all() for x in jax.tree.leaves(tree)]).all()


@jax.jit
def gradient_norm(tree):return optax.global_norm(tree)


@jax.jit
def clip_gradients(tree,norm):return jax.tree.map(lambda x:x*jnp.minimum(1.,1./(norm+1e-6)),tree)


@lru_cache(maxsize=8)
def update_kernel(tx):
    @jax.jit
    def apply(params,state,gradients):
        updates,new_state=tx.update(gradients,state,params)
        return optax.apply_updates(params,updates),new_state
    return apply


def cast_fp32(tree):return jax.tree.map(lambda x:x.astype(jnp.float32) if hasattr(x,"dtype") and jnp.issubdtype(x.dtype,jnp.floating) else x,tree)


def failures(tree,require_nonnegative=False):
    result=[]
    for path,value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not hasattr(value,"dtype"):continue
        array=np.asarray(jax.device_get(value));finite=np.isfinite(array)
        negative=int((array<0).sum()) if require_nonnegative else 0
        if not finite.all() or negative:
            good=array[finite]
            result.append({"tensor":jax.tree_util.keystr(path),"shape":list(array.shape),"dtype":str(array.dtype),
                           "nonfinite":int((~finite).sum()),"negative":negative,
                           "finite_min":float(good.min()) if good.size else None,"finite_max":float(good.max()) if good.size else None})
    return result


def assert_finite(tree,stage):
    valid=all_finite(tree)
    if not bool(jax.device_get(valid)):raise NumericalFailure(stage,failures(tree))


class NumericalFailure(RuntimeError):
    def __init__(self,stage,tensors):
        self.stage,self.tensors=stage,tensors
        super().__init__(f"update refused at {stage}: {tensors}")


def optimizer(params,learning_rate,warmup_steps=5):
    # NNX state keys explicitly identify new rows and normalization/bias tensors.
    def decay(path,value):
        name=jax.tree_util.keystr(path).lower()
        return value.ndim>=2 and not any(term in name for term in ("rows","norm","bias"))
    mask=jax.tree_util.tree_map_with_path(decay,params)
    schedule=lambda count:learning_rate*jnp.minimum(1.,(count+1)/warmup_steps)
    return optax.adamw(schedule,b1=.9,b2=.999,eps=1e-8,weight_decay=.01,mask=mask)


def checked_update(params,state,gradients,tx,reduce=None):
    gradients=cast_fp32(gradients);assert_finite(gradients,"local-gradients")
    reduced=gradients if reduce is None else reduce(gradients)
    assert_finite(reduced,"reduced-gradients")
    norm=gradient_norm(reduced)
    if not np.isfinite(float(norm)):raise NumericalFailure("gradient-norm",[{"norm":str(float(norm))}])
    clipped=clip_gradients(reduced,norm)
    new_params,new_state=update_kernel(tx)(params,state,clipped)
    assert_finite(new_params,"parameters");assert_finite(new_state,"optimizer-state")
    for item in jax.tree.leaves(new_state,is_leaf=lambda x:hasattr(x,"nu")):
        if hasattr(item,"nu"):
            valid=all_nonnegative(item.nu)
            if not bool(jax.device_get(valid)):raise NumericalFailure("second-moments",failures(item.nu,require_nonnegative=True))
    return new_params,new_state,{"gradient_norm":float(norm),"accepted":True}


def advantages(rewards):
    values=jnp.asarray(rewards,dtype=jnp.float32)
    return jax.lax.stop_gradient((values-values.mean())/(values.std()+1e-6))


def grpo_loss(new,old,reference,advantage,epsilon=.2,beta=.02):
    # Inputs are checked outside tracing; extreme tails are refused rather than hidden.
    new=new.astype(jnp.float32);old=jax.lax.stop_gradient(old.astype(jnp.float32));reference=jax.lax.stop_gradient(reference.astype(jnp.float32))
    ratio=jnp.exp(new-old)
    policy=jnp.minimum(ratio*advantage,jnp.clip(ratio,1-epsilon,1+epsilon)*advantage)
    delta=reference-new;kl=jnp.expm1(delta)-delta
    return jnp.mean(-policy+beta*kl)
