"""Owned one-slice experiment: pilot, corrective SFT, mixed-action RL, evaluation.

Launch only through experiment_v2 after all local gates have passed. This worker
does not create or delete accelerators; two independent cloud watchdogs do that.
"""
import argparse
import hashlib
import json
import math
import signal
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .recovery_control import StageControl,StageLimit,balanced_schedule
from .experiment_v2 import launch_gates,freeze_evaluation,sha
from .experiment_v2 import revisions
from .tunix_load import load_full
from .tunix_runtime import TunixRuntime
from .tunix_train import MixedLearner
from .tunix_distributed import Replicas
from .tunix_evaluate import Evaluator,compare
from .tunix_native import NativeRuntime
from .checkpoint_v2 import CheckpointsV2
from .tunix_optim import NumericalFailure,assert_finite


def publish_file(session,path,key):
    from google.cloud import storage
    bucket,prefix=session['gcs_prefix'].removeprefix('gs://').split('/',1)
    blob=storage.Client(project=session['project']).bucket(bucket).blob(f'{prefix}/{key}')
    blob.metadata={'sha256':sha(path)};blob.upload_from_filename(str(path),checksum='auto',timeout=60)
    blob.reload(timeout=60)
    if blob.size!=Path(path).stat().st_size or blob.metadata.get('sha256')!=sha(path):raise RuntimeError('artifact upload verification failed')


def memory_report(devices):
    result=[]
    for device in devices:
        stats=device.memory_stats() or {}
        used=stats.get('peak_bytes_in_use',stats.get('bytes_in_use'));limit=stats.get('bytes_limit')
        if used is None or limit is None or limit<=0:raise RuntimeError('device memory headroom could not be measured')
        result.append({'device':str(device),'peak_bytes':int(used),'limit_bytes':int(limit),'headroom':1-used/limit})
    return result


def tree_checksum(tree):
    digest=hashlib.sha256()
    for path,value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        digest.update(jax.tree_util.keystr(path).encode());digest.update(str(value.dtype).encode())
        digest.update(np.asarray(jax.device_get(value)).tobytes())
    return digest.hexdigest()


class Experiment:
    def __init__(self,session,root):
        self.root=Path(root);self.session=session;self.folder=self.root/'runs/recovery-v2'
        self.cfg=json.loads((self.root/'configs/recovery.json').read_text())
        self.control=StageControl(self.cfg,session,self.folder,lambda path,key:publish_file(session,path,key))
        self.data=self.root/'data/recovery-v2';self.checkpoints=CheckpointsV2(self.folder/'checkpoints',session['gcs_prefix']+'/checkpoints')
        self.train=balanced_schedule(self.rows('train'));self.val=balanced_schedule(self.rows('val'))
        self.runtime=None;self.learner=None;self.selected={};self.reports={};self.closed=False
        for name in (signal.SIGTERM,signal.SIGINT):signal.signal(name,lambda *_:setattr(self.control,'interrupted',True))

    def rows(self,split):return [json.loads(line) for line in (self.data/f'{split}.jsonl').read_text().splitlines()]

    def metadata(self,phase):
        return {'schema_version':2,'backend':'tunix','phase':phase,'model':self.cfg['model'],
                'model_revision':self.cfg['base_model_revision'],'tunix_revision':self.cfg['tunix_revision'],
                'tokenizer_mode_ids':self.runtime.mode_ids,'dataset_manifest_sha256':sha(self.data/'manifest.json'),
                'tokenizer_revision':{p.name:sha(p) for p in (self.folder/'source-sft/tokenizer').iterdir() if p.is_file()},
                'implementation_revisions':revisions(self.root),'source_manifest_sha256':json.loads((self.folder/'converted/manifest.json').read_text())['source_manifest_sha256'],
                'protocol':'episode-v2/trajectory-v2','run_id':self.session['run_id']}

    def save(self,tag):
        self.control.check();folder=self.checkpoints.save(tag,self.learner.state(),self.metadata(self.learner.phase))
        self.control.record('checkpoint',{'tag':tag,'path':str(folder),'manifest_sha256':sha(folder/'manifest.json')})
        self.control.sync();return folder

    def restore_selected(self,name):
        folder=self.selected[name]
        state=self.checkpoints.load(folder,self.learner.state(),{'backend':'tunix','model':self.cfg['model'],'phase':self.learner.phase})
        self.learner.restore(state);self.runtime.actor=self.learner.actor

    def load(self):
        if not launch_gates(self.root,self.cfg,self.data,self.folder)['passed']:raise RuntimeError('local acceptance reports changed in payload')
        if jax.default_backend()!='tpu' or len(jax.local_devices())!=4:raise RuntimeError('approved four-device TPU slice not detected')
        self.control.sync();self.control.check()
        from huggingface_hub import snapshot_download
        started=time.perf_counter()
        base=snapshot_download(self.cfg['model'],revision=self.cfg['base_model_revision'],
                               allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
        self.control.check()
        model,tokenizer,manifest=load_full(base,self.folder/'converted',self.folder/'source-sft/tokenizer')
        self.runtime=TunixRuntime(model,tokenizer,max(self.cfg['buckets']),
                                 {k:self.cfg[k] for k in ('max_transitions','max_lookups','max_actions')})
        self.base_hash=tree_checksum(self.runtime.frozen)
        self.control.record('restore',{'seconds':time.perf_counter()-started,'source_step':manifest['source_step'],
                            'devices':[str(d) for d in jax.local_devices()],'frozen_sha256':self.base_hash})

    def pilot(self):
        from .validate_tunix import validate
        self.control.check();validate(self.folder/'tiny-reference',self.folder/'tpu-numerical.json')
        settings=[]
        # Each setting uses disposable optimizer/policy state, never the SFT source.
        for microbatch in (1,2,4):
            self.control.check();cfg={**self.cfg,'microbatch':microbatch};learner=MixedLearner(self.runtime,cfg,'sft',self.folder/f'pilot-microbatch-{microbatch}')
            times=[]
            try:
                for iteration in range(3):
                    self.control.check(max(times,default=0));started=time.perf_counter();learner.sft_step(self.train[:32]);times.append(time.perf_counter()-started)
                memory=memory_report(jax.local_devices())
                stable=all(r['headroom']>=self.cfg['min_memory_headroom'] for r in memory)
                settings.append({'microbatch':microbatch,'cold_step_seconds':times[0],'warmed_seconds':times[1:],
                                 'memory':memory,'accepted':stable})
                self.control.record('pilot-setting',settings[-1]);self.control.sync()
            finally:learner.close()
            if not stable:break
        viable=[s for s in settings if s['accepted']]
        if not viable:raise StageLimit('replication did not retain measured memory headroom; no hardware upgrade')
        best=min(viable,key=lambda s:np.median(s['warmed_seconds']));self.cfg['microbatch']=best['microbatch']
        self.learner=MixedLearner(self.runtime,self.cfg,'rl',self.folder/'pilot-mixed')
        updates=[]
        for index in range(3):
            self.control.check();updates.append(self.learner.rl_step(self.train[index*4:index*4+4]))
            if index==0:
                folder=self.save('pilot-disposable')
                restored=self.checkpoints.load(folder,self.learner.state(),{'phase':'rl','run_id':self.session['run_id']})
                self.learner.restore(restored)
        assert_finite(self.learner.state(),'pilot-state')
        if tree_checksum(self.runtime.frozen)!=self.base_hash:raise RuntimeError('frozen backbone changed')
        self.reports['pilot']={'passed':True,'settings':settings,'selected_microbatch':best['microbatch'],
                    'mixed_accepted_updates':3,'save_reload_continuation':True,'frozen_unchanged':True,'rl_timings':updates}
        (self.folder/'pilot.json').write_text(json.dumps(self.reports['pilot'],indent=2))
        self.learner.close();self.learner=None
        # The real phase always begins at converted SFT; disposable pilot state is discarded.
        return best,updates

    def validation(self,name):
        evaluator=Evaluator(self.learner.replicas,self.control,self.folder/'validation',self.cfg['seed'])
        started=time.perf_counter();records,report=evaluator.run(name,self.val,self.learner.actor)
        elapsed=time.perf_counter()-started
        report['seconds']=elapsed;report['compute_cost_proxy']=sum(r['cost_proxy'] for r in records)/len(records)
        self.control.record('validation',report);return report,elapsed

    @staticmethod
    def ranking(report,step):return (report['grounded_accuracy'],-report['compute_cost_proxy'],-step)

    def sft(self,pilot):
        self.control.enter('sft');self.learner=MixedLearner(self.runtime,self.cfg,'sft',self.folder/'sft')
        # Exposed validation can be examined before corrective updates; sealed test cannot.
        initial,val_seconds=self.validation('old-sft')
        batch=self.cfg['effective_sft_batch'];times=list(pilot['warmed_seconds']);steps=math.ceil(len(self.train)/batch)
        self.selected['sft']=self.save('best-sft');best=initial
        self.reports['sft']={'old_validation':initial,'selected_validation':best,'protocol_gate_passed':best['protocol_validity']>=.95,
                            'checkpoint':str(self.selected['sft']),'one_pass_completed':False}
        self.control.project(times,steps,compilation=pilot['cold_step_seconds'],validation=3*val_seconds)
        for offset in range(0,len(self.train),batch):
            self.control.project(times,steps-self.learner.accepted_updates,validation=((steps-self.learner.accepted_updates+15)//16)*val_seconds)
            started=time.perf_counter();self.learner.sft_step(self.train[offset:offset+batch]);times.append(time.perf_counter()-started)
            if self.learner.accepted_updates%self.cfg['checkpoint_interval']==0:self.save('latest')
            if self.learner.accepted_updates%16==0 or offset+batch>=len(self.train):
                report,val_seconds=self.validation(f'sft-{self.learner.accepted_updates:04d}')
                if self.ranking(report,self.learner.accepted_updates)>self.ranking(best,int(json.loads((self.selected['sft']/'manifest.json').read_text())['accepted_updates'])):
                    best=report;self.selected['sft']=self.save('best-sft')
                self.reports['sft'].update(selected_validation=best,protocol_gate_passed=best['protocol_validity']>=.95,checkpoint=str(self.selected['sft']))
        self.save('latest');self.restore_selected('sft')
        self.reports['sft']={'old_validation':initial,'selected_validation':best,'protocol_gate_passed':best['protocol_validity']>=.95,
                             'checkpoint':str(self.selected['sft']),'one_pass_completed':True}
        self.learner.close();self.learner=None;return best

    def rl(self,pilot,baseline):
        if baseline['protocol_validity']<.95:
            self.reports['rl']={'status':'skipped','reason':'corrective SFT protocol validity below 95%'};return
        self.control.enter('rl');self.learner=MixedLearner(self.runtime,self.cfg,'rl',self.folder/'rl')
        # Fresh optimizer; the selected actor is frozen as exactly matching reference state.
        self.runtime.actor=self.learner.actor;best=None;stale=0;timings=[u['seconds'] for u in pilot];varied=[]
        affordable=None
        for count in range(self.cfg['target_rl_updates'],self.cfg['minimum_rl_updates']-1,-1):
            projection=self.control.budget.project('rl',timings,count,validation=math.ceil(count/10)*baseline['seconds'])
            if projection['allowed']:affordable=count;break
        if affordable is None:raise StageLimit('the minimum 20-update RL milestone does not fit its allowance')
        self.control.project(timings,affordable,validation=math.ceil(affordable/10)*baseline['seconds'])
        for update in range(affordable):
            self.control.project(timings,affordable-update,validation=((affordable-update+9)//10)*baseline['seconds'])
            offset=update*self.cfg['prompts_per_update'];rows=[self.train[(offset+i)%len(self.train)] for i in range(self.cfg['prompts_per_update'])]
            info=self.learner.rl_step(rows);timings.append(info['seconds']);varied.append(info['varied_reward_fraction']);self.control.sync()
            if update==1 and sum(varied[:2])/2<.2:
                self.reports['rl']={'status':'stopped-for-task-review','accepted_updates':2,'varied_reward_fraction':varied};break
            if self.learner.accepted_updates%self.cfg['checkpoint_interval']==0:self.save('latest')
            if self.learner.accepted_updates%10==0 or update+1==affordable:
                report,_=self.validation(f'rl-{self.learner.accepted_updates:04d}')
                best_step=int(json.loads((self.selected['rl']/'manifest.json').read_text())['accepted_updates']) if 'rl' in self.selected else 0
                if best is None or self.ranking(report,self.learner.accepted_updates)>self.ranking(best,best_step):
                    best=report;self.selected['rl']=self.save('best-rl');stale=0
                else:stale+=1
                if stale>=3:break
        self.save('latest')
        self.reports.setdefault('rl',{'status':'finished','accepted_updates':self.learner.accepted_updates,
                    'stability_milestone':self.learner.accepted_updates>=self.cfg['minimum_rl_updates'],
                    'selected_validation':best,'affordable_target':affordable,'improvement_established':False})
        self.learner.close();self.learner=None

    def evaluation(self,stage_entered=False):
        if not stage_entered:self.control.enter('evaluation')
        # Read trainable state using a matching SFT optimizer template, then keep actor only.
        template=MixedLearner(self.runtime,self.cfg,'sft',self.folder/'eval-template')
        sft=self.checkpoints.load(self.selected['sft'],template.state(),{'phase':'sft'})['actor'];template.close()
        actors={'sft':sft};hashes={'sft':sha(self.selected['sft']/'manifest.json')}
        if 'rl' in self.selected:
            template=MixedLearner(self.runtime,self.cfg,'rl',self.folder/'eval-rl-template')
            actors['rl']=self.checkpoints.load(self.selected['rl'],template.state(),{'phase':'rl'})['actor'];template.close()
            hashes['rl']=sha(self.selected['rl']/'manifest.json')
        native_tokenizer=__import__('transformers').AutoTokenizer.from_pretrained(self.cfg['model'],revision=self.cfg['base_model_revision'],local_files_only=True)
        specs={'sft_switching':(self.runtime,actors['sft'],'learned')}
        if 'rl' in actors:specs['rl_switching']=(self.runtime,actors['rl'],'learned')
        for name,policy in (('always_direct','always_direct'),('always_jev','always_jev'),('always_cot','always_cot'),('confidence','confidence')):specs[name]=(self.runtime,sft,policy)
        native_direct=NativeRuntime(self.runtime,native_tokenizer,False);native_cot=NativeRuntime(self.runtime,native_tokenizer,True)
        specs.update(native_without_thinking=(native_direct,native_direct.actor,'native'),native_with_thinking=(native_cot,native_cot.actor,'native'))
        timings={};threshold=.8
        for name,(runtime,actor,policy) in specs.items():
            replicas=Replicas(runtime)
            try:
                evaluator=Evaluator(replicas,self.control,self.folder/'evaluation-profiling',self.cfg['seed'])
                evaluator.warm(name,self.val,actor,policy);timings[name]=evaluator.timings[name]
            finally:replicas.close()
        # Tune every predeclared threshold on validation, before opening test outcomes.
        replicas=Replicas(self.runtime);ranks=[]
        try:
            evaluator=Evaluator(replicas,self.control,self.folder/'thresholds',self.cfg['seed'])
            # Threshold tuning cannot consume the minimum primary-comparison reserve.
            test_seconds=sum(max(v) for v in timings.values())*math.ceil(60/len(replicas.devices))+600
            self.control.project(timings['confidence'],math.ceil(len(self.val)/len(replicas.devices))*len(self.cfg['confidence_thresholds']),
                                 compilation=0,validation=test_seconds)
            for value in self.cfg['confidence_thresholds']:
                records,report=evaluator.run(f'confidence-{value}',self.val,sft,'confidence',value)
                ranks.append((report['grounded_accuracy'],-sum(r['cost_proxy'] for r in records)/len(records),-value,value))
            threshold=max(ranks)[-1]
        finally:replicas.close()
        # Reading test IDs is allowed here; predictions start only after settings are sealed.
        test=self.rows('test');plan=freeze_evaluation(test,timings,self.control.budget,self.folder/'test-plan.json',hashes,
                 {**self.cfg,'selected_confidence_threshold':threshold,'evaluation_parallelism':len(jax.local_devices())})
        publish_file(self.session,self.folder/'test-plan.json','test-plan.json')
        selected=[r for r in test if r['id'] in set(plan['ids'])];results={}
        for name,(runtime,actor,policy) in specs.items():
            replicas=Replicas(runtime)
            try:
                evaluator=Evaluator(replicas,self.control,self.folder/'test',self.cfg['seed'])
                results[name],_=evaluator.run(name,selected,actor,policy,threshold)
            finally:replicas.close()
        self.reports['evaluation']=compare(results,selected,self.folder/'comparison.json')
        # Keep every recorded test attempt. Any demonstrated transition must occur naturally.
        self.demonstrations(results,selected)
        self.ablations(results,selected,actors.get('rl',sft),timings.get('rl_switching',timings['sft_switching']))

    def ablations(self,results,episodes,actor,times):
        from copy import deepcopy
        from .evaluation_v2 import summary
        replicas=Replicas(self.runtime);reports={};evaluator=Evaluator(replicas,self.control,self.folder/'ablations',self.cfg['seed'])
        try:
            for name in ('no_jev','initial_mode'):
                projected=self.control.budget.project('evaluation',times,math.ceil(len(episodes)/len(replicas.devices)),compilation=120)
                if not projected['allowed']:reports[name]={'status':'skipped','reason':'remaining evaluation allowance'};break
                _,reports[name]=evaluator.run(name,episodes,actor,name)
            subset=balanced_schedule(episodes)[:24]
            projected=self.control.budget.project('evaluation',times,3*math.ceil(len(subset)/len(replicas.devices)),compilation=120)
            if projected['allowed']:
                attempts=[]
                for index in range(3):
                    variant=deepcopy(subset)
                    rng=np.random.default_rng(self.cfg['seed']+index)
                    for row in variant:rng.shuffle(row['visible']['candidates'])
                    records,_=evaluator.run(f'permutation-{index}',variant,actor);attempts.append({r['id']:r for r in records})
                stable=sum(len({json.dumps(a[row['id']]['answer'],sort_keys=True) for a in attempts})==1 for row in subset)
                reports['candidate_permutations']={'count':len(subset),'permutations':3,'answer_stability':stable/len(subset)}
            else:reports['candidate_permutations']={'status':'skipped','reason':'remaining evaluation allowance'}
        finally:replicas.close()
        # Context counterfactuals are already paired in the primary suite; no extra run.
        final=results.get('rl_switching',results['sft_switching']);by_id={r['id']:r for r in final};groups={}
        for row in episodes:groups.setdefault(row['group'],[]).append(row)
        pairs=[]
        for group,rows in groups.items():
            if len(rows)!=2 or bool(rows[0]['oracle'].get('required_fields'))==bool(rows[1]['oracle'].get('required_fields')):continue
            pairs.append({'group':group,'variants':[{'id':r['id'],'evidence_missing':bool(r['oracle'].get('required_fields')),
                          'modes':by_id[r['id']]['modes'],'tool_calls':by_id[r['id']]['tool_calls'],
                          'grounded_correct':bool(by_id[r['id']]['correct'] and by_id[r['id']]['grounded'])} for r in rows]})
        reports['context_counterfactuals']={'pairs':pairs,'status':'measured-from-primary-comparison'}
        (self.folder/'ablations.json').write_text(json.dumps(reports,indent=2));self.reports['ablations']=reports

    def demonstrations(self,results,episodes):
        rows=results.get('rl_switching',results['sft_switching']);pool={e['id']:e for e in balanced_schedule(episodes)[:24]}
        attempts=[r for r in rows if r['id'] in pool];patterns={'jev_to_cot':('<mode:jev>','<mode:cot>'),'cot_to_jev':('<mode:cot>','<mode:jev>')}
        selected={}
        for name,pair in patterns.items():
            selected[name]=next((r['id'] for r in attempts if r['correct'] and r['grounded'] and pair in list(zip(r.get('modes',[]),r.get('modes',[])[1:]))),None)
        for behavior in ('clarification','lookup'):selected[behavior]=next((r['id'] for r in attempts if r['behavior']==behavior and r['acquisition_success'] and r['correct'] and r['grounded']),None)
        (self.folder/'demonstrations.json').write_text(json.dumps({'schema_version':2,'saved_replays':True,
                'selection_note':'selected successes are examples; all attempts and failures retained; absent transitions reported as null',
                'pool_ids':list(pool),'selected_success_ids':selected,'attempts':attempts},indent=2))

    def export(self):
        record={'schema_version':2,'backend':'tunix','run_id':self.session['run_id'],'reports':self.reports,
                'checkpoints':{tag:{'directory':str(folder),'manifest_sha256':sha(folder/'manifest.json'),
                                      'gcs':self.session['gcs_prefix']+'/checkpoints/'+folder.name} for tag,folder in self.selected.items()},
                'cost_upper_bound_before_verified_deletion':self.control.budget.total(),
                'billing_reconciled':False,'single_training_seed':True}
        target=self.folder/'result.json';target.write_text(json.dumps(record,indent=2));publish_file(self.session,target,'result.json')
        self.control.sync()

    def run(self):
        try:
            self.load();sft_pilot,rl_pilot=self.pilot();best=self.sft(sft_pilot);self.rl(rl_pilot,best);self.evaluation()
        except (StageLimit,NumericalFailure) as exc:
            self.reports['stop']={'type':type(exc).__name__,'reason':str(exc),'stage':self.control.attempt['stage']}
            # Invalid new state is never published. The preceding accepted state is recoverable.
            if self.learner is not None and isinstance(exc,StageLimit):
                self.checkpoints.save('latest',self.learner.state(),self.metadata(self.learner.phase))
            self.control.record('stopped',self.reports['stop'])
            if isinstance(exc,StageLimit) and 'sft' in self.selected and self.control.attempt['stage'] in {'sft','rl'}:
                if self.learner is not None:self.learner.close();self.learner=None
                self.reports.setdefault('rl',{'status':'skipped','reason':'training-stage allowance'})
                try:
                    self.control.enter_evaluation_after_budget_stop();self.evaluation(stage_entered=True)
                except StageLimit as evaluation_stop:
                    self.reports['evaluation']={'complete':False,'reason':str(evaluation_stop)}
                    self.control.record('evaluation-incomplete',self.reports['evaluation'])
        except Exception as exc:
            self.reports['failure']={'type':type(exc).__name__,'reason':str(exc),'stage':self.control.attempt['stage']}
            self.control.record('failed',self.reports['failure']);raise
        finally:
            if self.learner is not None:self.learner.close()
            self.checkpoints.close();self.export()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    jax.config.update('jax_compilation_cache_dir',str(root/'.cache/jax'))
    jax.config.update('jax_default_matmul_precision','highest')
    Experiment(json.loads(Path(args.session).read_text()),root).run()


if __name__=='__main__':main()
