import unittest
from switching.compare import choose_threshold
from switching.evaluate import summary

class EvaluationTests(unittest.TestCase):
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
