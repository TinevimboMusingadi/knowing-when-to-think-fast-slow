"""Tunix Qwen with explicit FP32 PEFT state and the preserved typed head.

Import only in the isolated JAX environment. Frozen base and actor/reference
states are separated by NNX variable type rather than host-side weight swapping.
"""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


class Trainable(nnx.Param):
    pass


class ProjectionLoRA(nnx.Module):
    def __init__(self, original, a, b, scale, output_projection=False):
        self.original=original
        self.a=Trainable(jnp.asarray(a.T,dtype=jnp.float32))
        self.b=Trainable(jnp.asarray(b.T,dtype=jnp.float32))
        self.scale=float(scale);self.output_projection=output_projection
        self.shape=original.shape
        self.output_shape=original.w.value.shape[-1:] if output_projection else original.w.value.shape[1:]

    def __call__(self,x):
        base=self.original(x)
        flat=x.reshape(*x.shape[:2],-1) if self.output_projection else x
        delta=(flat.astype(jnp.float32)@self.a.value@self.b.value)*self.scale
        return base+delta.reshape(*base.shape).astype(base.dtype)


class ModeEmbedder(nnx.Module):
    def __init__(self, original, rows, boundary):
        self.original=original;self.rows=Trainable(jnp.asarray(rows,dtype=jnp.float32));self.boundary=int(boundary)
        self.dtype=original.dtype

    def encode(self,ids):
        old=self.original.encode(jnp.minimum(ids,self.boundary-1))
        new=self.rows.value[jnp.clip(ids-self.boundary,0,len(self.rows.value)-1)].astype(old.dtype)
        return jnp.where((ids>=self.boundary)[...,None],new,old)

    def decode(self,hidden):
        old=self.original.decode(hidden)[...,:self.boundary].astype(jnp.float32)
        new=hidden.astype(jnp.float32)@self.rows.value.T
        return jnp.concatenate((old,new),axis=-1)


def decision_head(weights,representations,valid):
    """Numerically equivalent PyTorch post-norm TransformerEncoder, dropout=0."""
    def linear(x,prefix):return x@weights[prefix+".weight"].T+weights[prefix+".bias"]
    def norm(x,prefix):
        mean=jnp.mean(x,axis=-1,keepdims=True);variance=jnp.mean((x-mean)**2,axis=-1,keepdims=True)
        return (x-mean)*jax.lax.rsqrt(variance+1e-5)*weights[prefix+".weight"]+weights[prefix+".bias"]
    x=linear(representations.astype(jnp.float32),"project")
    batch,count,width=x.shape;heads=4;head_dim=width//heads
    for index in range(2):
        prefix=f"attention.layers.{index}"
        q,k,v=jnp.split(x@weights[prefix+".self_attn.in_proj_weight"].T+weights[prefix+".self_attn.in_proj_bias"],3,axis=-1)
        q,k,v=[value.reshape(batch,count,heads,head_dim).transpose(0,2,1,3) for value in (q,k,v)]
        score=jnp.einsum("bhid,bhjd->bhij",q,k)/jnp.sqrt(float(head_dim))
        score=jnp.where(valid[:,None,None,:],score,jnp.finfo(jnp.float32).min)
        attention=jax.nn.softmax(score,axis=-1)
        attended=jnp.einsum("bhij,bhjd->bhid",attention,v).transpose(0,2,1,3).reshape(batch,count,width)
        x=norm(x+linear(attended,prefix+".self_attn.out_proj"),prefix+".norm1")
        x=norm(x+linear(jax.nn.relu(linear(x,prefix+".linear1")),prefix+".linear2"),prefix+".norm2")
    logits=linear(x,"score").squeeze(-1)
    return jnp.where(valid,logits,-1e9)


class TunixSwitchModel(nnx.Module):
    def __init__(self,base,arrays,boundary,rank=16,alpha=32):
        self.base=base;self.boundary=int(boundary)
        self.candidate_microbatch=1
        row_name=next(n for n in arrays if n.endswith("rows"))
        self.base.embedder=ModeEmbedder(self.base.embedder,arrays[row_name],boundary)
        self.base.config.vocab_size=boundary+3
        self.head=nnx.Dict({n.removeprefix("head."):Trainable(jnp.asarray(v,dtype=jnp.float32)) for n,v in arrays.items() if n.startswith("head.")})
        consumed={row_name}|{n for n in arrays if n.startswith("head.")}
        for index,layer in enumerate(self.base.layers):
            for projection in ("q_proj","k_proj","v_proj","o_proj"):
                suffix=f"model.layers.{index}.self_attn.{projection}.lora_A.default.weight"
                matches=[n for n in arrays if n.endswith(suffix)]
                if len(matches)!=1:raise ValueError(f"adapter architecture mismatch: {suffix}")
                an=matches[0];bn=an.replace("lora_A","lora_B")
                if arrays[an].shape[0]!=rank or arrays[bn].shape[1]!=rank:raise ValueError("adapter rank mismatch")
                original=getattr(layer.attn,projection)
                setattr(layer.attn,projection,ProjectionLoRA(original,arrays[an],arrays[bn],alpha/rank,projection=="o_proj"))
                consumed.update((an,bn))
        if consumed!=set(arrays):raise ValueError("unmapped trainable tensors")

    def hidden(self,ids,positions,mask,cache=None,segments=None):
        return self.base(input_tokens=ids,positions=positions,cache=cache,attention_mask=mask,segment_ids=segments,skip_lm_head=True)

    def __call__(self,ids,positions,mask,cache=None,segments=None,select_positions=None):
        hidden,cache=self.hidden(ids,positions,mask,cache,segments)
        if select_positions is not None:
            hidden=hidden[:,select_positions] if select_positions.ndim==1 else hidden[jnp.arange(hidden.shape[0])[:,None],select_positions]
        if self.base.config.use_tied_embedding:logits=self.base.embedder.decode(hidden)
        else:
            old=self.base.lm_head(hidden)[...,:self.boundary].astype(jnp.float32)
            new=hidden.astype(jnp.float32)@self.base.embedder.rows.value.T
            logits=jnp.concatenate((old,new),axis=-1)
        return logits.astype(jnp.float32),cache

    def candidate_logits(self,ids,lengths,candidate_valid):
        # Every candidate is an independent batch row, with matching reset positions.
        count=candidate_valid.shape[1];batch=candidate_valid.shape[0];width=ids.shape[1]
        positions=jnp.broadcast_to(jnp.arange(width)[None,:],ids.shape)
        valid=jnp.arange(width)[None,:]<lengths[:,None]
        mask=valid[:,None,:] & (jnp.arange(width)[:,None]>=jnp.arange(width)[None,:])[None,:,:]
        chunk=min(self.candidate_microbatch,len(ids))
        if len(ids)%chunk:raise ValueError('candidate count must divide its encoding microbatch')
        # A scan compiles one backbone body. Python unrolling duplicated all 28
        # layers per candidate and created a large host compilation workload.
        # Parameters are explicit invariant carry, not embedded weight literals.
        graph,state=nnx.split(self.base)
        def encode(state,inputs):
            tokens,pos,attention,size=inputs
            base=nnx.merge(graph,state)
            hidden,_=base(input_tokens=tokens,positions=pos,cache=None,attention_mask=attention,skip_lm_head=True)
            return state,hidden[jnp.arange(chunk),jnp.maximum(size-1,0)]
        inputs=(ids.reshape(-1,chunk,width),positions.reshape(-1,chunk,width),
                mask.reshape(-1,chunk,width,width),lengths.reshape(-1,chunk))
        _,representations=jax.lax.scan(jax.checkpoint(encode),state,inputs)
        reps=representations.reshape(batch,count,-1)
        weights={n:p.value for n,p in self.head.items()}
        return decision_head(weights,reps,candidate_valid)

    def split(self):
        return nnx.split(self,Trainable,...)


def load_portable(folder):
    import hashlib
    folder=Path(folder);manifest=json.loads((folder/"manifest.json").read_text())
    if hashlib.sha256((folder/"trainables.npz").read_bytes()).hexdigest()!=manifest["npz_sha256"]:raise ValueError("portable checksum mismatch")
    with np.load(folder/"trainables.npz",allow_pickle=False) as archive:arrays={n:archive[n].copy() for n in archive.files}
    return arrays,manifest
