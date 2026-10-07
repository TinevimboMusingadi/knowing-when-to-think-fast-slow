import json
import tempfile
import unittest
from pathlib import Path
from switching.experiment_v2 import launch_gates,freeze_evaluation
from switching.recovery_budget import RecoveryBudget
from switching.data_v2 import make,BEHAVIORS
from switching.tunix_environment import fingerprint,require_unchanged


class ExperimentV2Tests(unittest.TestCase):
    def config(self):return json.loads((Path(__file__).parents[1]/"configs/recovery.json").read_text())
    def test_missing_or_stale_reports_cannot_authorize_spending(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"manifest.json").write_text("{}")
            result=launch_gates(root,self.config(),root,root)
            self.assertFalse(result["paid_launch_allowed"]);self.assertGreater(len(result["issues"]),3)
    def test_balanced_pair_selection_is_frozen_before_results(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=self.config();budget=RecoveryBudget(cfg,Path(folder)/"budget.json")
            rows=[make("test",b,i) for b in BEHAVIORS for i in range(24)]
            names=["native_without_thinking","native_with_thinking","sft_switching","always_direct","always_jev","always_cot","confidence"]
            result=freeze_evaluation(rows,{p:[1,2,3] for p in names},budget,Path(folder)/"plan.json",{"sft":"abc"},cfg)
            self.assertEqual(result["count_per_policy"],120);self.assertFalse(result["read_test_results"])
            groups={r["group"] for r in rows if r["id"] in result["ids"]};self.assertEqual(len(groups),60)
            with self.assertRaises(ValueError):freeze_evaluation(rows,{p:[1,2,3] for p in names},budget,Path(folder)/"plan.json",{"sft":"changed"},cfg)

    def test_validation_cannot_stamp_new_code_as_the_tested_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'switching').mkdir();(root/'scripts').mkdir()
            data=root/'data/recovery-v2';data.mkdir(parents=True);(data/'manifest.json').write_text('{}')
            code=root/'switching/learner.py';code.write_text('original')
            report=fingerprint(root);require_unchanged(root,report)
            code.write_text('changed during validation')
            with self.assertRaisesRegex(ValueError,'revisions'):require_unchanged(root,report)
