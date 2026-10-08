"""Budget pressure must not displace mixed-action pilot acceptance checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np


class PilotOrchestrationTests(unittest.TestCase):
    def test_core_checks_finish_when_larger_batch_projection_is_denied(self):
        try:
            from switching import tunix_experiment as engine
            from switching import validate_tunix
        except ImportError:
            self.skipTest('run this orchestration check in the locked Tunix environment')
        events=[]
        class Learner:
            phase='rl'
            def __init__(self,*args):events.append('mixed-created')
            def rl_step(self,rows):events.append(('mixed-update',list(rows)));return {'seconds':1.,'varied_reward_fraction':1.}
            def state(self):return {'actor':np.array([1.],dtype=np.float32)}
            def restore(self,state):events.append('restored')
            def close(self):events.append('mixed-closed')
        with tempfile.TemporaryDirectory() as temporary:
            experiment=engine.Experiment.__new__(engine.Experiment)
            experiment.folder=Path(temporary);experiment.cfg={};experiment.train=list(range(12))
            experiment.runtime=SimpleNamespace(frozen={'weight':np.array([1.],dtype=np.float32)})
            experiment.base_hash=engine.tree_checksum(experiment.runtime.frozen)
            experiment.session={'run_id':'fixture'};experiment.reports={};experiment.learner=None
            def project(*args,**kwargs):
                events.append('optimization-projection');return {'allowed':False,'cost_upper_bound':2.,'stage_remaining':1.}
            experiment.control=SimpleNamespace(check=lambda *args:None,sync=lambda:None,record=lambda *args:None,
                                               budget=SimpleNamespace(project=project))
            def setting(microbatch):
                events.append(('supervised-setting',microbatch))
                return {'accepted':True,'microbatch':microbatch,'cold_step_seconds':60.,'warmed_seconds':[1.,2.]}
            experiment.pilot_setting=setting
            experiment.save=lambda tag: experiment.folder/tag
            experiment.checkpoints=SimpleNamespace(load=lambda folder,state,metadata:state)
            with patch.object(validate_tunix,'validate',return_value={'passed':True}),patch.object(engine,'MixedLearner',Learner):
                best,updates=experiment.pilot()
            self.assertEqual(len(updates),3);self.assertEqual(best['microbatch'],1)
            self.assertEqual([event for event in events if isinstance(event,tuple) and event[0]=='supervised-setting'],
                             [('supervised-setting',1)])
            self.assertLess(events.index('restored'),events.index(('mixed-update',[4,5,6,7])))
            self.assertLess(events.index(('mixed-update',[8,9,10,11])),events.index('optimization-projection'))
            report=json.loads((experiment.folder/'pilot.json').read_text())
            self.assertTrue(report['passed']);self.assertEqual(report['untested_microbatches'],[2,4])
            self.assertFalse(report['optimization_complete']);self.assertTrue((experiment.folder/'pilot-core.json').exists())
