import unittest
import random
from switching.recover import verify_dataset
from switching.storage import rng_tuple

class RecoveryTests(unittest.TestCase):
    def test_xla_list_converted_python_rng_restores_exactly(self):
        generator=random.Random(42);state=generator.getstate();expected=generator.random()
        converted=[state[0],list(state[1]),state[2]]
        restored=random.Random();restored.setstate(rng_tuple(converted))
        self.assertEqual(restored.random(),expected)
    def test_recovery_requires_all_dataset_checksums(self):
        current={"splits":{s:{"sha256":s} for s in ("train","val","test")}}
        verify_dataset(current,current)
        altered={"splits":{s:{"sha256":s+"x"} for s in ("train","val","test")}}
        with self.assertRaises(ValueError):verify_dataset(current,altered)

if __name__=="__main__":unittest.main()
