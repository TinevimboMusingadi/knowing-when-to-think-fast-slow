"""Explicit HF frozen-weight mapping; trainables are converted independently."""
import dataclasses
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from flax import nnx
from tunix.models.qwen3.model import Qwen3,ModelConfig,RematConfig
from .tunix_model import TunixSwitchModel,load_portable


def verify_base(folder):
    from .verified_download import digest
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/recovery.json').read_text())
    folder=Path(folder)
    if {p.name for p in folder.glob('*.safetensors')}!=set(cfg['base_weight_hashes']):raise ValueError('unexpected or missing base shard')
    for name,expected in cfg['base_weight_hashes'].items():
        if digest(folder/name)!=expected:raise ValueError('base weights differ from pinned Hugging Face revision')
    return cfg['base_model_revision']


def config_from_hf(cfg,dtype=jnp.float32,remat=False):
    return ModelConfig(num_layers=cfg["num_hidden_layers"],vocab_size=cfg["vocab_size"],embed_dim=cfg["hidden_size"],
                       hidden_dim=cfg["intermediate_size"],num_heads=cfg["num_attention_heads"],head_dim=cfg["head_dim"],
                       num_kv_heads=cfg["num_key_value_heads"],rope_theta=cfg["rope_theta"],norm_eps=cfg["rms_norm_eps"],
                       use_tied_embedding=cfg["tie_word_embeddings"],dtype=dtype,param_dtype=dtype,
                       remat_config=RematConfig.DECODER if remat else RematConfig.NONE)


def assign_frozen(base,tensors):
    consumed=set()
    def put(variable,key,transpose=False):
        value=tensors[key].T if transpose else tensors[key]
        if value.size!=variable.value.size:raise ValueError(f"frozen shape mismatch: {key}")
        variable.value=jnp.asarray(value.reshape(variable.value.shape),dtype=variable.value.dtype);consumed.add(key)
    put(base.embedder.input_embedding,"model.embed_tokens.weight");put(base.final_norm.w,"model.norm.weight")
    for i,layer in enumerate(base.layers):
        prefix=f"model.layers.{i}."
        for name in ("input_layernorm","post_attention_layernorm"):put(getattr(layer,name).w,prefix+name+".weight")
        for name in ("q_proj","k_proj","v_proj","o_proj"):put(getattr(layer.attn,name).w,prefix+"self_attn."+name+".weight",True)
        for name in ("q_norm","k_norm"):put(getattr(layer.attn,name).w,prefix+"self_attn."+name+".weight")
        for name in ("gate_proj","up_proj","down_proj"):put(getattr(layer.mlp,name).kernel,prefix+"mlp."+name+".weight",True)
    if not base.config.use_tied_embedding:put(base.lm_head.w,"lm_head.weight",True)
    unexpected=set(tensors)-consumed
    # Tied HF exports may duplicate lm_head; verify equality before excluding it.
    if unexpected=={"lm_head.weight"} and base.config.use_tied_embedding:
        if not np.array_equal(tensors["lm_head.weight"],tensors["model.embed_tokens.weight"]):raise ValueError("base language-head tying mismatch")
        unexpected=set()
    if unexpected:raise ValueError(f"unmapped frozen tensors: {sorted(unexpected)}")


def load_fixture(folder,bf16=False,remat=False):
    folder=Path(folder);metadata=json.loads((folder/"fixture.json").read_text())
    # The HF fixture config was extended by three modes after construction.
    cfg=dict(metadata["hf_config"]);cfg["vocab_size"]=np.load(folder/"frozen.npz")["model.embed_tokens.weight"].shape[0]
    base=Qwen3(config_from_hf(cfg,jnp.bfloat16 if bf16 else jnp.float32,remat=remat),rngs=nnx.Rngs(42))
    with np.load(folder/"frozen.npz",allow_pickle=False) as values:assign_frozen(base,{n:values[n] for n in values.files})
    with np.load(folder/"trainables.npz",allow_pickle=False) as values:arrays={n:values[n] for n in values.files}
    return TunixSwitchModel(base,arrays,metadata["vocabulary_boundary"],metadata["rank"],metadata["alpha"]),metadata


def load_full(base_folder,portable,tokenizer_folder,bf16=True,storage_bf16=False):
    from transformers import AutoTokenizer
    from safetensors import safe_open
    base_folder=Path(base_folder);cfg=json.loads((base_folder/"config.json").read_text())
    verify_base(base_folder)
    if cfg["num_hidden_layers"]!=28 or cfg["hidden_size"]!=2048:raise ValueError("Qwen3-1.7B architecture required")
    config=config_from_hf(cfg,jnp.bfloat16 if bf16 else jnp.float32,remat=bf16)
    if storage_bf16:config.param_dtype=jnp.bfloat16
    # Allocate each checkpoint tensor once; do not initialize another full model.
    base=nnx.eval_shape(lambda:Qwen3(config,rngs=nnx.Rngs(42)))
    readers={};locations={}
    for path in sorted(base_folder.glob("*.safetensors")):
        reader=safe_open(str(path),framework='flax');reader.__enter__();readers[str(path)]=reader
        for name in reader.keys():
            if name in locations:raise ValueError("duplicate base tensor")
            locations[name]=reader
    class LazyWeights:
        def __getitem__(self,name):return locations[name].get_tensor(name)
        def __iter__(self):return iter(locations)
    try:assign_frozen(base,LazyWeights())
    finally:
        for reader in readers.values():reader.__exit__(None,None,None)
    arrays,manifest=load_portable(portable)
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_folder,local_files_only=True)
    for name,identifier in manifest["mode_tokens"].items():
        if tokenizer.convert_tokens_to_ids(name)!=identifier:raise ValueError("tokenizer mode identity mismatch")
    if len(tokenizer)!=manifest["vocabulary_boundary"]+3:raise ValueError("tokenizer boundary mismatch")
    model=TunixSwitchModel(base,arrays,manifest["vocabulary_boundary"])
    return model,tokenizer,manifest
