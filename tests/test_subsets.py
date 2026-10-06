import collections
import unittest
from switching.subsets import BEHAVIORS,balanced_rows

class SubsetTests(unittest.TestCase):
    def test_balanced_selection_is_order_independent_and_interleaved(self):
        rows=[{"id":f"{behavior}-{i}","behavior":behavior} for behavior in BEHAVIORS for i in range(20)]
        selected=balanced_rows(rows)
        self.assertEqual(len(selected),60)
        self.assertEqual(collections.Counter(r["behavior"] for r in selected),dict.fromkeys(BEHAVIORS,10))
        self.assertEqual(selected,balanced_rows(list(reversed(rows))))
        self.assertEqual([r["behavior"] for r in selected[:6]],list(BEHAVIORS))
        self.assertNotEqual(selected,balanced_rows(rows,seed=7))

    def test_refuses_duplicates_and_unbalanced_input(self):
        with self.assertRaises(ValueError):balanced_rows([{"id":"a","behavior":"fast"}]*2)
        with self.assertRaises(ValueError):balanced_rows([{"id":"a","behavior":"fast"}])

if __name__=="__main__":unittest.main()
