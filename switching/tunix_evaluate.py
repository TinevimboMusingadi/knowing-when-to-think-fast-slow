"""Synchronized comparisons with sealed settings and complete attempt records."""
import json
import time
from pathlib import Path
import jax
import numpy as np
from .evaluation_v2 import summary,paired_interval
from .protocol import equivalent


def public_result(record):
    return {k:v for k,v in record.items() if k!='trajectory'}


class Evaluator:
    def __init__(self,replicas,control,folder,seed):
        self.replicas,self.control,self.folder,self.seed=replicas,control,Path(folder),seed
        self.folder.mkdir(parents=True,exist_ok=True)
        self.timings={};self.compilation={}

    def run(self,name,rows,actor,policy='learned',threshold=.8,retain=True):
        records=[];world=len(self.replicas.devices);folder=self.folder/name;folder.mkdir(exist_ok=True)
        target=folder/'records.jsonl'
        if retain and target.exists():raise ValueError('evaluation attempts may not overwrite existing records')
        for offset in range(0,len(rows),world):
            recent=[r['warmed_seconds'] for r in records[-world*2:]]
            self.control.check(max(recent,default=0))
            selected=rows[offset:offset+world]
            key=jax.random.PRNGKey(self.seed+offset)
            def work(runtime,params,reference,batch,draw):
                if not batch:return []
                old=runtime.suppress_jev
                try:
                    runtime.suppress_jev=policy=='no_jev'
                    result=runtime.rollouts(params,batch,draw,name,sample=True,policy=policy,threshold=threshold)
                finally:runtime.suppress_jev=old
                jax.block_until_ready(params)
                return [public_result(r) for r in result]
            started=time.perf_counter()
            result=self.replicas.map(work,actor,actor,[[r] for r in selected]+[[]]*(world-len(selected)),jax.random.split(key,world))
            wall=time.perf_counter()-started
            for group in result:
                for record in group:
                    # One episode per replica: this latency is not a batch-time division.
                    record.update(policy=name,warmed_seconds=record['batch_seconds'],collection_wall_seconds=wall)
                    records.append(record)
                    if retain:
                        with target.open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
            self.control.sync()
        report=summary(records);report['complete']=len(records)==len(rows)
        report['requested_count']=len(rows)
        if retain:(folder/'summary.json').write_text(json.dumps(report,indent=2))
        return records,report

    def warm(self,name,rows,actor,policy='learned',threshold=.8):
        sample=rows[:len(self.replicas.devices)]
        cold=time.perf_counter();self.run(name+'-compile',sample,actor,policy,threshold)
        self.compilation[name]=time.perf_counter()-cold
        records,_=self.run(name+'-warm',sample,actor,policy,threshold)
        self.timings[name]=[r['collection_wall_seconds'] for r in records]
        self.control.record('inference-timing',{'policy':name,'cold_first_collection_seconds':self.compilation[name],
                'warmed_individual_seconds':self.timings[name],
                'compilation_note':'first collection includes compilation and inference; not isolated compiler time'})


def detailed_summary(records,episodes):
    from collections import Counter
    result=summary(records);by_id={e['id']:e for e in episodes}
    modes=Counter();transitions=Counter();unnecessary=0;success={'lookup':0,'clarification':0};briers=[];calibration=[]
    for record in records:
        modes.update(record.get('modes',[]));ms=record.get('modes',[])
        transitions.update(f'{a} -> {b}' for a,b in zip(ms,ms[1:]) if a!=b)
        row=by_id[record['id']]
        if not row['oracle'].get('required_fields'):unnecessary+=record.get('tool_calls',0)
        if record['behavior'] in success:success[record['behavior']]+=bool(record['acquisition_success'] and record['correct'] and record['grounded'])
        for prediction in record.get('jev_predictions',[]):
            answers=[c for c in prediction if c['kind']=='answer']
            total=sum(c['probability'] for c in answers)
            if not answers or total<=0:continue
            p=np.array([c['probability']/total for c in answers]);y=np.array([equivalent(c['value'],row['oracle']['answer']) for c in answers],dtype=float)
            if y.sum()!=1:continue
            briers.append(float(((p-y)**2).sum()));index=int(p.argmax());calibration.append((float(p[index]),bool(y[index])))
    bins=[]
    for lower in np.arange(0,1,.1):
        pairs=[(p,y) for p,y in calibration if lower<=p<lower+.1 or lower>=.9 and p==1]
        if pairs:bins.append({'lower':float(lower),'count':len(pairs),'confidence':float(np.mean([p for p,_ in pairs])),
                              'accuracy':float(np.mean([y for _,y in pairs]))})
    result.update(mode_frequencies=dict(modes),transitions=dict(transitions),unnecessary_tool_calls=unnecessary,
                  acquisition_correct_grounded=success,jev_answer_conditional_brier=float(np.mean(briers)) if briers else None,
                  jev_calibration_bins=bins,
                  calibration_definition='Conditional probability over answer candidates; acquisition/defer candidates excluded. Every scored call retained.')
    return result


def compare(records_by_policy,episodes,output):
    reports={name:detailed_summary(rows,episodes) for name,rows in records_by_policy.items()}
    differences={name:{'grounded_difference_vs_sft':reports[name]['grounded_accuracy']-reports['sft_switching']['grounded_accuracy'],
                       'paired_source_group_bootstrap_95':paired_interval(records_by_policy['sft_switching'],rows)}
                 for name,rows in records_by_policy.items() if name!='sft_switching'}
    result={'schema_version':2,'training_seeds':1,'exploratory':True,'policies':reports,'differences':differences}
    Path(output).write_text(json.dumps(result,indent=2));return result
