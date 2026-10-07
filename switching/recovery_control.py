"""Stage deadlines and projections, independent of either numerical backend."""
import json
import time
from pathlib import Path
from .recovery_budget import RecoveryBudget


class StageLimit(RuntimeError):
    pass


def device_memory_exhausted(error):
    """Only a device allocation failure may end the increasing batch probe."""
    module=type(error).__module__
    message=str(error).lower()
    return (module.startswith(('jax.', 'jaxlib.')) and
            'resource_exhausted' in message and
            any(word in message for word in ('memory', 'allocation', 'allocating')))


class StageControl:
    def __init__(self,config,session,folder,publish=None,now=time.time):
        self.config,self.session,self.folder=config,session,Path(folder)
        self.publish,self.now=publish,now
        self.budget=RecoveryBudget(config,self.folder/'budget.json',now=now)
        self.attempt=next(a for a in self.budget.ledger['attempts'] if a['run_id']==session['run_id'])
        self.interrupted=False

    def sync(self):
        self.budget.save()
        if self.publish:self.publish(self.folder/'budget.json','budget.json')

    def check(self,seconds=0):
        deadline=min(self.attempt['worker_deadline'],self.session['deadline']-300)
        if self.interrupted or self.now()+seconds>=deadline:
            raise StageLimit('stage deadline reached; retain the last accepted checkpoint')

    def enter(self,stage):
        self.check()
        self.attempt=self.budget.advance(self.session['run_id'],stage)
        self.sync()

    def enter_evaluation_after_budget_stop(self):
        # The training-stage limit must not consume the protected evaluation
        # allocation. Interruptions and the overall deadline still stop all work.
        if self.attempt['stage'] not in {'sft','rl'}:
            raise StageLimit('no training-stage evaluation recovery is available')
        if self.interrupted or self.now()>=self.session['deadline']-300:
            raise StageLimit('experiment interrupted or overall deadline reached')
        self.attempt=self.budget.advance(self.session['run_id'],'evaluation')
        self.sync();self.check()

    def project(self,times,steps,compilation=0,validation=0):
        self.check()
        result=self.budget.project(self.attempt['stage'],times,steps,compilation,validation)
        self.record('projection',result)
        if not result['allowed']:raise StageLimit('measured work does not fit the stage allowance')
        self.check(result['seconds']-300)
        return result

    def record(self,name,detail):
        event={'schema_version':2,'event':name,'stage':self.attempt['stage'],
               'timestamp':self.now(),'detail':detail,'cost_upper_bound':self.budget.total()}
        self.folder.mkdir(parents=True,exist_ok=True)
        with (self.folder/'operations.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(event)+'\n')


def balanced_schedule(rows):
    """Cycle behaviors, preserving each source group and every episode exactly once."""
    groups={}
    for row in rows:groups.setdefault(row['behavior'],[]).append(row)
    keys=sorted(groups);result=[]
    for index in range(max(map(len,groups.values()))):
        for key in keys:
            if index<len(groups[key]):result.append(groups[key][index])
    return result
