import unittest
import numpy as np
import torch
from switching.conversion import portable_state,project_adapter


class ConversionTests(unittest.TestCase):
    def test_projection_matches_peft_scaling_and_axes(self):
        rng=np.random.default_rng(42)
        for input_shape,output_shape in (((12,),(3,4)),((3,4),(8,))):
            a=rng.normal(size=(2,np.prod(input_shape))).astype(np.float32);b=rng.normal(size=(np.prod(output_shape),2)).astype(np.float32)
            ja,jb=project_adapter(a,b,input_shape,output_shape)
            x=rng.normal(size=(5,np.prod(input_shape))).astype(np.float32)
            expected=(x@a.T@b.T)*2
            actual=(x@ja.reshape(-1,2)@jb.reshape(2,-1))*2
            np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-4)
    def test_invalid_and_nonfinite_source_is_rejected(self):
        with self.assertRaises(ValueError):portable_state({"x":torch.ones(1)})
        with self.assertRaises(ValueError):portable_state({"head.weight":torch.tensor([float("nan")])})
        with self.assertRaises(ValueError):project_adapter(np.ones((2,3)),np.ones((4,2)),(7,),(4,))
