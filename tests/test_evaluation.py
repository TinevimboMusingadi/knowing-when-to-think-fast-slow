import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from switching.compare import main as compare_main
from switching.compare import choose_threshold,comparison_report
from switching.evaluate import summary

class EvaluationTests(unittest.TestCase):
    def test_interrupted_comparison_persists_completed_policy_records(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.head=torch.nn.Linear(1,1)
            def trainable_state(self):return {}
            def restore_trainable(self,state):pass
        targets=[{"id":"f","behavior":"fast"},{"id":"l","behavior":"lookup"}]
        values=[{"id":r["id"],"behavior":r["behavior"],"correct":True,"grounded":True,"tokens":1,"forwards":1,"batch_seconds":1,"batch_size":1,"trace":[],"modes":[],"error":None} for r in targets]
        with tempfile.TemporaryDirectory() as folder:
            argv=["compare","--sft","placeholder","--output",folder,"--device","cpu","--offline"]
            with patch("sys.argv",argv),patch("switching.compare.load_rows",return_value=targets),patch("switching.compare.SwitchModel.load",return_value=Model()),patch("switching.compare.Checkpoints"),patch("switching.compare.native_base",return_value=values[:1]),patch("switching.compare.rollout_group",side_effect=[values,values,KeyboardInterrupt("test stop")]):
                with self.assertRaisesRegex(RuntimeError,"test stop"):compare_main()
            report=json.loads((Path(folder)/"comparison.json").read_text())
            self.assertFalse(report["complete"]);self.assertEqual(report["base_qwen"]["count"],1)
            self.assertEqual(report["sft_switch"]["count"],0);self.assertEqual(report["stop_reason"],"test stop")
            self.assertEqual(len((Path(folder)/"base_qwen.jsonl").read_text().splitlines()),1)
    def test_pairing_requires_base_records_only_where_applicable(self):
        targets=[{"id":"fast","behavior":"fast"},{"id":"ask","behavior":"clarification"}]
        values=[{"id":r["id"],"behavior":r["behavior"],"correct":True,"grounded":True,"tokens":1,"forwards":1,"batch_seconds":1,"batch_size":1,"trace":[],"modes":[],"error":None} for r in targets]
        report=comparison_report(targets,{"base_qwen":[],"sft_switch":values})
        self.assertFalse(report["complete"]);self.assertEqual(report["completed_paired_episodes"],1)
        report=comparison_report(targets,{"base_qwen":values[:1],"sft_switch":values})
        self.assertTrue(report["complete"]);self.assertEqual(report["base_qwen"]["applicable_target"],1)
        with self.assertRaises(ValueError):comparison_report(targets,{"sft_switch":values+values})
    def test_acquisition_requires_action_and_grounded_correct_answer(self):
        row={"correct":True,"grounded":True,"behavior":"lookup","lookups":0,"modes":["direct","jev"],"tokens":2,"forwards":3,"batch_seconds":1,"batch_size":1,"error":None,"trace":[{"kind":"baseline_decision","probabilities":[.9,.1],"gold_index":0}]}
        report=summary([row]);self.assertEqual(report["acquisition_success"]["lookup"],0)
        self.assertEqual(report["mean_transitions"],1);self.assertAlmostEqual(report["brier_score"],.02)
        row["lookups"]=1
        self.assertEqual(summary([row])["acquisition_success"]["lookup"],1)

    def test_threshold_uses_paired_validation_records(self):
        f=[{"id":"a","correct":True,"grounded":True,"tokens":0,"trace":[{"kind":"baseline_decision","probabilities":[.9,.1]}]}]
        s=[{"id":"a","correct":True,"grounded":True,"tokens":50}]
        chosen=choose_threshold(f,s);self.assertEqual(chosen["validation_mean_tokens"],0)
        s[0]["id"]="b"
        with self.assertRaises(ValueError):choose_threshold(f,s)

    def test_summary_reports_failures_and_intervals(self):
        row={"correct":False,"grounded":True,"behavior":"fast","tokens":2,"forwards":3,"batch_seconds":1,"batch_size":1,"error":"bad action","trace":[]}
        report=summary([row]);self.assertEqual(report["accuracy"],0);self.assertEqual(report["errors"]["bad action"],1)

if __name__=="__main__":unittest.main()
