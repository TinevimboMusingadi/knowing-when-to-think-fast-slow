"""FP32 full-model reference with frozen BF16 storage and bounded cast memory."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from switching.model import SwitchModel
from switching.conversion import verify_source


def fp32_frozen_hooks(model):
    base=model.lm.get_base_model();blocks=[base.model.embed_tokens.original,*base.model.layers,base.model.norm,base.lm_head.original]
    handles=[]
    for block in blocks:
        originals={id(p):p.dtype for p in block.parameters() if not p.requires_grad}
        def before(module,inputs,originals=originals):
            for p in module.parameters():
                if id(p) in originals:p.data=p.data.float()
        def after(module,inputs,output,originals=originals):
            for p in module.parameters():
                if id(p) in originals:p.data=p.data.to(originals[id(p)])
        handles.extend((block.register_forward_pre_hook(before),block.register_forward_hook(after)))
    return handles


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--checkpoint',default='runs/recovery-v2/source-sft')
    parser.add_argument('--output',default='runs/recovery-v2/full-reference');args=parser.parse_args()
    verify_source(args.checkpoint)
    # Pure-Python checksum validation without importing the JAX backend here.
    from switching.verified_download import digest
    cfg=json.loads(Path('configs/recovery.json').read_text())
    for name,expected in cfg['base_weight_hashes'].items():
        if digest(Path(args.base)/name)!=expected:raise ValueError('base weights do not match the fixed model revision')
    state=torch.load(Path(args.checkpoint)/'state.pt',map_location='cpu',weights_only=False)
    model=SwitchModel.load(args.base,dtype=torch.bfloat16,local_files_only=True)
    for p in model.parameters():
        if p.requires_grad:p.data=p.data.float()
    model.restore_trainable(state['trainable']);del state
    model.eval();torch.set_num_threads(4);handles=fp32_frozen_hooks(model)
    boundary=model.lm.get_base_model().model.embed_tokens.old_size
    ids=torch.tensor([[2,boundary,3,4,5,6,7,8]])
    cids=torch.tensor([[2,3,4,5,0,0,0,0],[2,3,4,6,7,0,0,0],[2,3,4,8,9,10,0,0],[0]*8]);lengths=torch.tensor([4,5,6,1]);valid=torch.tensor([[True,True,True,False]])
    with torch.no_grad():
        logits=model.lm(input_ids=ids,use_cache=False).logits.float()
        mask=torch.arange(8)[None,:]<lengths[:,None]
        hidden=model.lm.get_base_model().model(input_ids=cids,attention_mask=mask,use_cache=False).last_hidden_state
        reps=hidden[torch.arange(4),lengths-1];decisions=model.head(reps[None,:],valid)
    for handle in handles:handle.remove()
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    np.savez(folder/'expected.npz',ids=ids.numpy(),candidate_ids=cids.numpy(),lengths=lengths.numpy(),valid=valid.numpy(),
             language_probabilities=logits.softmax(-1).numpy(),selected_logps=logits[0,1:4].log_softmax(-1)[torch.arange(3),torch.tensor([3,4,5])].numpy(),
             candidate_probabilities=decisions.softmax(-1).numpy())
    (folder/'manifest.json').write_text(json.dumps({'model':'Qwen/Qwen3-1.7B','source_step':150,'computation_dtype':'float32',
         'frozen_storage':'bfloat16; per-block exact cast to float32 for computation','vocabulary_boundary':boundary,
         'files':{'expected.npz':hashlib.sha256((folder/'expected.npz').read_bytes()).hexdigest()}},indent=2))


if __name__=='__main__':main()
