import unittest
from switching.evaluation_v2 import wilson,select,paired_interval
from switching.data_v2 import make,BEHAVIORS


class EvaluationV2Tests(unittest.TestCase):
    def test_perfect_small_result_is_not_certain(self):
        low,high=wilson(4,4);self.assertLess(low,.6);self.assertAlmostEqual(high,1)
    def test_fixed_balanced_selection_and_strict_pairing(self):
        rows=[make("test",b,i) for b in BEHAVIORS for i in range(24)]
        chosen=select(rows,120);self.assertEqual(len(chosen),120)
        self.assertEqual([r["id"] for r in chosen],[r["id"] for r in select(rows,120)])
        with self.assertRaises(ValueError):paired_interval([{"id":"a"}],[{"id":"b"}])
