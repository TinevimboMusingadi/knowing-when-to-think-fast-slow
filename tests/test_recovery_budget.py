import json
import tempfile
import unittest
from pathlib import Path
from switching.recovery_budget import RecoveryBudget,validate_config


class RecoveryBudgetTests(unittest.TestCase):
    def setUp(self):self.config=json.loads(Path("configs/recovery.json").read_text())
    def test_new_cap_includes_prior_spend_and_evaluation_reserve(self):
        with tempfile.TemporaryDirectory() as folder:
            b=RecoveryBudget(self.config,Path(folder)/"ledger.json",now=lambda:1000.)
            self.assertAlmostEqual(b.remaining("rl"),self.config['budget']['rl'])
            item=b.begin("pilot","owned")
            allowance=self.config["budget"]["pilot"]
            self.assertAlmostEqual(item["deadline"]-1000,allowance/5.52*3600)
            with self.assertRaises(RuntimeError):b.begin("rl","other")
            with self.assertRaises(RuntimeError):b.finish("owned",item["deadline"],False)
            b.finish("owned",item["deadline"],True)
            self.assertAlmostEqual(b.total(),self.config["prior_spend_upper_bound"]+allowance)
            self.assertAlmostEqual(b.remaining("evaluation"),16)
    def test_overallocation_and_bad_checkpoint_are_rejected(self):
        bad={**self.config,"hard_ceiling":70}
        with self.assertRaises(ValueError):validate_config(bad)
        bad={**self.config,"source_checkpoint":"gs://bucket/20261007-084201/checkpoint"}
        with self.assertRaises(ValueError):validate_config(bad)
    def test_projection_counts_cold_and_cleanup_time(self):
        with tempfile.TemporaryDirectory() as folder:
            b=RecoveryBudget(self.config,Path(folder)/"ledger.json")
            p=b.project("pilot",[10,20,30],5,compilation=60,validation=90)
            self.assertEqual(p["seconds"],600);self.assertTrue(p["allowed"])
    def test_one_resource_charges_each_stage_without_losing_idle_time(self):
        with tempfile.TemporaryDirectory() as folder:
            current=[1000.];b=RecoveryBudget(self.config,Path(folder)/"ledger.json",now=lambda:current[0])
            b.begin("pilot","one");current[0]+=600;b.advance("one","sft");current[0]+=1200
            b.finish("one",current[0],True)
            self.assertAlmostEqual(b.stage_spend("pilot"),.92);self.assertAlmostEqual(b.stage_spend("sft"),1.84)
            self.assertAlmostEqual(b.total(),64.87)
