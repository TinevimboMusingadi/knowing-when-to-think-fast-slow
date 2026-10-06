import unittest
import torch
from switching.protocol import MODES,encode_action
from switching.runtime import SessionRuntime

class Tokenizer:
    pad_token_id=0
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]
    def decode(self,ids,skip_special_tokens=False):return "".join(chr(int(c)) for c in ids)
    def convert_tokens_to_ids(self,text):return 1

class ScriptedModel(torch.nn.Module):
    def __init__(self,completions):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.tokenizer=Tokenizer()
        self.completions=iter(completions);self.lm=self
    def eval(self):return self
    def prompt_ids(self,messages):return [2,3]
    def generate(self,input_ids,**kwargs):
        new=torch.tensor([self.tokenizer.encode(next(self.completions))])
        return torch.cat((input_ids,new),dim=1)
    def decision_logits(self,states,candidates,bucket):
        return torch.tensor([[4.0,0.0]])

class RuntimeTests(unittest.TestCase):
    def test_intermediate_jev_continues_and_clarification_resumes(self):
        model=ScriptedModel([MODES[0],encode_action(MODES[2],"reason",text="I need the record value."),encode_action(MODES[1],"ask",question="What is the value?"),MODES[0]])
        runtime=SessionRuntime(model)
        candidates=[{"id":"a","value":7,"text":"7"},{"id":"b","value":8,"text":"8"}]
        stage={"candidates":[{"id":"reason","value":"reason","text":"Reason further"},{"id":"direct","value":"direct","text":"Answer now"}]}
        result=runtime.run({"prompt":"Find the undisclosed record value.","candidates":candidates,"decision_stages":[stage]},"one")
        self.assertEqual(result["status"],"awaiting_user")
        result=runtime.resume("one","The record value is 7.")
        self.assertEqual(result["answer"],7)
        self.assertEqual(result["modes"],[MODES[0],MODES[2],MODES[1],MODES[0]])

    def test_public_requests_reject_gold_stage_labels(self):
        runtime=SessionRuntime(ScriptedModel([]))
        with self.assertRaises(ValueError):runtime.run({"prompt":"p","decision_stages":[{"expected":"reason"}]},"one")

if __name__=="__main__":unittest.main()
