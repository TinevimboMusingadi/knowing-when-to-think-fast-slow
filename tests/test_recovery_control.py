import json
import tempfile
import unittest
from pathlib import Path
from switching.recovery_control import StageControl,StageLimit,balanced_schedule,device_memory_exhausted
from switching.recovery_budget import RecoveryBudget


class RecoveryControlTests(unittest.TestCase):
    def test_only_device_oom_can_retain_a_smaller_pilot_batch(self):
        device_error=type('JaxRuntimeError',(RuntimeError,),{'__module__':'jaxlib._jax'})
        self.assertTrue(device_memory_exhausted(device_error('RESOURCE_EXHAUSTED: Out of memory allocating buffer')))
        self.assertFalse(device_memory_exhausted(device_error('INVALID_ARGUMENT: incorrect tensor shape')))
        self.assertFalse(device_memory_exhausted(RuntimeError('RESOURCE_EXHAUSTED: memory')))
        self.assertFalse(device_memory_exhausted(FloatingPointError('nonfinite gradient')))

    def test_worker_deadline_is_earlier_than_watchdog_and_evaluation_is_reserved(self):
        cfg=json.loads(Path('configs/recovery.json').read_text());clock=[1000.]
        with tempfile.TemporaryDirectory() as folder:
            budget=RecoveryBudget(cfg,Path(folder)/'budget.json',now=lambda:clock[0]);attempt=budget.begin('pilot','owned')
            control=StageControl(cfg,{'run_id':'owned','deadline':50000.},folder,now=lambda:clock[0])
            self.assertEqual(attempt['deadline']-attempt['worker_deadline'],300.)
            clock[0]=attempt['worker_deadline']-1
            with self.assertRaises(StageLimit):control.check(2)
            self.assertEqual(control.budget.remaining('evaluation'),16.)

    def test_deadline_refresh_is_published_before_next_stage_work(self):
        cfg=json.loads(Path('configs/recovery.json').read_text());clock=[1000.];published=[]
        with tempfile.TemporaryDirectory() as folder:
            RecoveryBudget(cfg,Path(folder)/'budget.json',now=lambda:clock[0]).begin('pilot','owned')
            control=StageControl(cfg,{'run_id':'owned','deadline':50000.},folder,
                                 publish=lambda path,key:published.append(json.loads(path.read_text())),now=lambda:clock[0])
            clock[0]+=100;control.enter('sft')
            self.assertEqual(published[-1]['attempts'][0]['stage'],'sft')
            self.assertEqual(len(published[-1]['attempts'][0]['segments']),2)
            control.interrupted=True
            with self.assertRaises(StageLimit):control.project([1],1)

    def test_behavior_schedule_keeps_all_rows_once(self):
        rows=[{'id':f'{b}{i}','behavior':b} for b in ('a','b','c') for i in range(3)]
        result=balanced_schedule(rows)
        self.assertEqual([r['behavior'] for r in result[:3]],['a','b','c'])
        self.assertEqual(sorted(r['id'] for r in result),sorted(r['id'] for r in rows))

    def test_training_deadline_can_enter_reserved_evaluation_but_not_after_interruption(self):
        cfg=json.loads(Path('configs/recovery.json').read_text());clock=[1000.]
        with tempfile.TemporaryDirectory() as folder:
            RecoveryBudget(cfg,Path(folder)/'budget.json',now=lambda:clock[0]).begin('sft','owned')
            control=StageControl(cfg,{'run_id':'owned','deadline':50000.},folder,now=lambda:clock[0])
            clock[0]=control.attempt['worker_deadline']+1
            with self.assertRaises(StageLimit):control.check()
            control.enter_evaluation_after_budget_stop()
            self.assertEqual(control.attempt['stage'],'evaluation');control.check()
        with tempfile.TemporaryDirectory() as folder:
            RecoveryBudget(cfg,Path(folder)/'budget.json',now=lambda:clock[0]).begin('rl','owned')
            control=StageControl(cfg,{'run_id':'owned','deadline':50000.},folder,now=lambda:clock[0]);control.interrupted=True
            with self.assertRaises(StageLimit):control.enter_evaluation_after_budget_stop()
            self.assertEqual(control.attempt['stage'],'rl')
