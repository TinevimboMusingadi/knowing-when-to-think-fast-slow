import unittest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from switching.decoding import static_generate

class DecodingTests(unittest.TestCase):
    def test_static_greedy_matches_dynamic_cache_for_padded_batch(self):
        torch.manual_seed(4)
        config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        config._attn_implementation="eager"
        model=Qwen3ForCausalLM(config).eval();prompts=[[2,3,4],[5,6]]
        actual=static_generate(model,prompts,5,0,63,capacity=16,prefill_buckets=(8,))
        ids=torch.tensor([[2,3,4],[0,5,6]])
        expected=model.generate(input_ids=ids,attention_mask=ids.ne(0),max_new_tokens=5,do_sample=False,pad_token_id=0,eos_token_id=63)[:,3:]
        torch.testing.assert_close(actual,expected)

    def test_static_jev_stops_first_token(self):
        torch.manual_seed(4)
        config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        model=Qwen3ForCausalLM(config).eval();prompts=[[2,3,4]]
        first=int(static_generate(model,prompts,1,0,63,capacity=16,prefill_buckets=(8,))[0,0])
        output=static_generate(model,prompts,5,0,63,first,capacity=16,prefill_buckets=(8,))
        self.assertEqual(output.shape,(1,1))

if __name__=="__main__":unittest.main()
