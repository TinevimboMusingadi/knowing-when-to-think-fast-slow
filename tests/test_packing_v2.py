import unittest
import numpy as np
from switching.packing_v2 import packs,batch_packs


class PackingV2Tests(unittest.TestCase):
    def test_targets_exclude_prompts_and_history_with_reset_positions(self):
        items=packs([{"prompt":[1,2],"completion":[3,4]},{"prompt":[5],"completion":[6]}],0,(8,))
        row=items[0]
        np.testing.assert_array_equal(row["selected"],[1,2,4]);np.testing.assert_array_equal(row["targets"],[3,4,6])
        np.testing.assert_array_equal(row["positions"][0],[0,1,2,3,0,1,0,0])
        self.assertFalse(row["mask"][0,4,0]);self.assertTrue(row["mask"][0,5,4]);self.assertFalse(row["mask"][0,4,5])
        self.assertFalse(row["mask"][0,1,7]);self.assertEqual(row["useful_tokens"],6)
    def test_overlong_target_is_rejected_not_truncated(self):
        with self.assertRaises(ValueError):packs([{"prompt":[1]*5,"completion":[2]*4}],0,(8,))
    def test_microbatch_padding_does_not_become_a_target(self):
        items=packs([{"prompt":[1]*5,"completion":[2]*2},{"prompt":[3]*5,"completion":[4]}],0,(8,))
        batches=batch_packs(items,0,2);self.assertEqual(len(batches),1)
        self.assertEqual(int(batches[0]["target_mask"].sum()),3)
        np.testing.assert_array_equal(batches[0]["targets"][batches[0]["target_mask"]],[2,2,4])
