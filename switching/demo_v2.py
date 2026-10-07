"""Replay a labeled saved example, or run a verified v2 checkpoint on one episode."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--replay');p.add_argument('--base');p.add_argument('--checkpoint')
    p.add_argument('--checkpoint-gcs');p.add_argument('--tokenizer',default='runs/recovery-v2/source-sft/tokenizer')
    p.add_argument('--episode');p.add_argument('--output',default='runs/recovery-v2/live-demo.json');args=p.parse_args()
    if args.replay:
        record=json.loads(Path(args.replay).read_text())
        print(json.dumps({'saved_replay':True,'new_inference':False,'record':record},indent=2));return
    if not args.base or not args.episode or not (args.checkpoint or args.checkpoint_gcs):p.error('live demo requires base weights, a verified checkpoint, and an episode')
    import jax
    from .tunix_load import load_full
    from .tunix_runtime import TunixRuntime
    from .tunix_train import MixedLearner
    from .checkpoint_v2 import CheckpointsV2
    from .verified_download import restore
    from .tunix_evaluate import public_result
    root=Path(__file__).resolve().parents[1];cfg=json.loads((root/'configs/recovery.json').read_text())
    folder=Path(args.checkpoint or root/'runs/recovery-v2/restored-demo')
    if args.checkpoint_gcs:restore(args.checkpoint_gcs,folder)
    metadata=json.loads((folder/'manifest.json').read_text())
    model,tokenizer,_=load_full(args.base,root/'runs/recovery-v2/converted',args.tokenizer)
    runtime=TunixRuntime(model,tokenizer);learner=MixedLearner(runtime,cfg,metadata['phase'],root/'runs/recovery-v2/demo-template')
    manager=CheckpointsV2(root/'runs/recovery-v2/demo-checkpoints')
    try:
        state=manager.load(folder,learner.state(),{'backend':'tunix','model':cfg['model'],'tunix_revision':cfg['tunix_revision']})
        episode=json.loads(Path(args.episode).read_text())
        result=runtime.rollouts(state['actor'],[episode],jax.random.PRNGKey(cfg['seed']),'live-demo',sample=True)[0]
        record={'saved_replay':False,'new_inference':True,'checkpoint':str(folder),'result':public_result(result)}
        Path(args.output).write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
    finally:learner.close();manager.close()


if __name__=='__main__':main()
