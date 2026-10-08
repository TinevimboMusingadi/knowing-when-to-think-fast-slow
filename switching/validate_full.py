"""Full Qwen probability parity; fixed tolerance, verified SFT source, FP32 math."""
import argparse
import gc
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from .tunix_load import load_full
from .experiment_v2 import revisions,sha


def validate(base,reference,output):
    root=Path(__file__).resolve().parents[1];folder=Path(reference);metadata=json.loads((folder/'manifest.json').read_text())
    cfg=json.loads((root/'configs/recovery.json').read_text());started=time.time();checks=[]
    report={'schema_version':2,'stage':'full-model-parity','model':cfg['model'],'passed':False,'full_model_probability_parity':False,
            'tunix_revision':cfg['tunix_revision'],'revisions':revisions(root),'dataset_manifest_sha256':sha(root/'data/recovery-v2/manifest.json'),
            'probability_atol':1e-3,'tolerance_changed':False,'compute_dtype':'float32','checks':checks}
    def stage(name):
        report['execution_stage']=name
        report['seconds']=time.time()-started
        Path(output).write_text(json.dumps(report,indent=2))
        print(json.dumps({'event':'full-reference-stage','stage':name,'seconds':report['seconds']}),flush=True)
    try:
        from .tunix_environment import verify
        jax.config.update('jax_cpu_enable_async_dispatch',False)
        stage('verify-environment')
        report['environment']=verify(root,jax.default_backend());report['production_acceptance']=True
        if metadata['source_step']!=150 or metadata['computation_dtype']!='float32':raise ValueError('incorrect full reference')
        if sha(folder/'expected.npz')!=metadata['files']['expected.npz']:raise ValueError('full reference checksum changed')
        stage('load-full-backbone')
        model,_,manifest=load_full(base,root/'runs/recovery-v2/converted',root/'runs/recovery-v2/source-sft/tokenizer',bf16=False,storage_bf16=True)
        expected=np.load(folder/'expected.npz');ids=jnp.asarray(expected['ids']);width=ids.shape[1]
        stage('language-forward')
        logits,_=model(ids,jnp.arange(width)[None,:],jnp.tril(jnp.ones((1,width,width),dtype=bool)))
        language_probabilities=np.asarray(jax.device_get(jax.nn.softmax(logits))).copy()
        selected=jax.nn.log_softmax(logits[0,1:4])[jnp.arange(3),jnp.array([3,4,5])]
        selected_probabilities=np.asarray(jax.device_get(jnp.exp(selected))).copy()
        del logits,selected
        gc.collect();jax.clear_caches()
        # This FP32 conversion reference keeps BF16 weights in host memory and
        # casts each block for computation. Compiling the whole candidate scan
        # can hoist every invariant FP32 cast at once, exceeding a small CPU VM.
        # Interpret the identical scan body here; compiled scan/gradient checks
        # run separately on the small model, then BF16 execution on the TPU.
        report['candidate_execution']='eager-scan-body-for-bounded-fp32-reference-memory'
        report['cpu_async_dispatch']=False
        stage('candidate-forward')
        with jax.disable_jit():
            candidate_logits=model.candidate_logits(jnp.asarray(expected['candidate_ids']),jnp.asarray(expected['lengths']),jnp.asarray(expected['valid']))
        candidate_probabilities=np.asarray(jax.device_get(jax.nn.softmax(candidate_logits))).copy()
        for name,actual,target in (('language_probabilities',language_probabilities,expected['language_probabilities']),
                    ('candidate_probabilities',candidate_probabilities,expected['candidate_probabilities'])):
            difference=float(np.max(np.abs(actual-target)))
            np.testing.assert_allclose(actual,target,atol=1e-3,rtol=0)
            checks.append({'name':name,'passed':True,'maximum_absolute_difference':difference})
        # Compare selected probabilities under the predeclared full-model tolerance.
        np.testing.assert_allclose(selected_probabilities,np.exp(expected['selected_logps']),atol=1e-3,rtol=0)
        checks.append({'name':'selected-token-probabilities','passed':True})
        from .tunix_environment import require_unchanged
        require_unchanged(root,report)
        report.update(passed=True,full_model_probability_parity=True,source_step=manifest['source_step'],mode_tokens=manifest['mode_tokens'])
        report['execution_stage']='complete'
    except Exception as exc:report['error']={'type':type(exc).__name__,'message':str(exc)};raise
    finally:
        report['seconds']=time.time()-started;Path(output).write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--reference',default='runs/recovery-v2/full-reference')
    parser.add_argument('--output',default='runs/recovery-v2/full-model-parity.json');args=parser.parse_args()
    jax.config.update('jax_default_matmul_precision','highest')
    print(json.dumps(validate(args.base,args.reference,args.output),indent=2))
