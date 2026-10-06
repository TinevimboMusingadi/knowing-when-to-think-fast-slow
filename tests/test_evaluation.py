import unittest
from switching.compare import choose_threshold
from switching.evaluate import summary

class EvaluationTests(unittest.TestCase):
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
