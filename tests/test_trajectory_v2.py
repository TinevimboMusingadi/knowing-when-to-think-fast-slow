import unittest
from switching.trajectory_v2 import TrajectoryV2


class TrajectoryV2Tests(unittest.TestCase):
    def test_decision_is_categorical_not_a_fake_language_token(self):
        t=TrajectoryV2("sample","sft-v2")
        t.tokens([1,2],[3,4],[-.1,-.2],"public context")
        t.decision("state",[{"id":"a"},{"id":"b"}],1,-.3)
        self.assertEqual([e["kind"] for e in t.events],["tokens","decision"])
        self.assertNotIn("completion",t.events[1])
        self.assertNotIn("prompt",str(t.public_record()))
    def test_likelihoods_cannot_hide_nan(self):
        t=TrajectoryV2("sample","sft")
        with self.assertRaises(ValueError):t.tokens([1],[2],[float("nan")],"observation")
