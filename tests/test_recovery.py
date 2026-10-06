import unittest
from switching.recover import verify_dataset

class RecoveryTests(unittest.TestCase):
    def test_recovery_requires_all_dataset_checksums(self):
        current={"splits":{s:{"sha256":s} for s in ("train","val","test")}}
        verify_dataset(current,current)
        altered={"splits":{s:{"sha256":s+"x"} for s in ("train","val","test")}}
        with self.assertRaises(ValueError):verify_dataset(current,altered)

if __name__=="__main__":unittest.main()
