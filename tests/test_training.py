import tempfile
import unittest
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from switching.model import SwitchModel,DecisionHead
from switching.batching import pack
from switching.storage import Checkpoints
from switching.train import grpo_loss,advantages

class TinyTokenizer:
    pad_token_id=0
    def add_special_tokens(self,config):return 3
    def encode(self,text,add_special_tokens=False):return [1+ord(c)%60 for c in text]

def tiny_model():
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    config._attn_implementation="eager"
    return SwitchModel(Qwen3ForCausalLM(config),TinyTokenizer(),rank=2,alpha=4)

class TrainingTests(unittest.TestCase):
    def test_head_permutation(self):
        torch.manual_seed(1);head=DecisionHead(32).eval();x=torch.randn(2,4,32);order=[2,0,3,1]
        torch.testing.assert_close(head(x)[:,order],head(x[:,order,:]),atol=1e-5,rtol=1e-5)

    def test_packing_and_prompt_mask(self):
        batch,consumed,useful=pack([([1,2,3],[-100,2,3]),([4,5],[-100,5])],8,1,0,"cpu")
        self.assertEqual(consumed,2);self.assertEqual(useful,5)
        self.assertEqual(batch["position_ids"].tolist(),[[0,1,2,0,1,0,0,0]])
        self.assertLess(batch["attention_mask"][0,0,4,1].item(),-1e9)
        self.assertEqual(batch["labels"][0,3].item(),-100)

    def test_real_sft_and_mode_embedding_updates(self):
        torch.manual_seed(42);model=tiny_model();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        before=model.trainable_state();ids=torch.tensor([[2,64,3,4,5]])
        loss=model.lm(input_ids=ids,labels=ids,use_cache=False).loss;loss.backward()
        self.assertTrue(torch.isfinite(loss));optimizer.step()
        after=model.trainable_state();self.assertTrue(any(not torch.equal(before[k],after[k]) for k in before))
        row_name=next(k for k in before if k.endswith("rows"));self.assertFalse(torch.equal(before[row_name],after[row_name]))

    def test_real_generation_with_extended_vocabulary(self):
        model=tiny_model().eval()
        generated=model.lm.generate(input_ids=torch.tensor([[1,64,2]]),max_new_tokens=2,pad_token_id=0,eos_token_id=63)
        self.assertGreater(generated.shape[1],3)

    def test_reference_preserves_both_heads_and_embeddings(self):
        model=tiny_model();reference=model.trainable_state()
        with torch.no_grad():model.head.score.weight.add_(2)
        policy=model.trainable_state()
        with model.reference(reference):
            for k,v in model.trainable_state().items():torch.testing.assert_close(v,reference[k])
        for k,v in model.trainable_state().items():torch.testing.assert_close(v,policy[k])

    def test_decision_branches_invariant_and_head_updates(self):
        model=tiny_model().eval();c=[{"id":str(i),"text":str(i),"value":i} for i in range(3)]
        logits=model.decision_logits(["state"],[c],bucket=32)
        swapped=model.decision_logits(["state"],[[c[2],c[0],c[1]]],bucket=32)
        torch.testing.assert_close(logits[:,[2,0,1]],swapped,atol=1e-5,rtol=1e-5)
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([1]));loss.backward()
        self.assertGreater(model.head.score.weight.grad.abs().sum().item(),0)

    def test_grpo_changes_real_parameters(self):
        logits=torch.nn.Parameter(torch.tensor([0.,0.]));opt=torch.optim.SGD([logits],lr=.1)
        old=logits.detach().log_softmax(-1);new=logits.log_softmax(-1)
        adv=advantages([1.,-1.]);loss=grpo_loss(new,old,old,adv);loss.backward();opt.step()
        self.assertGreater(logits[0].item(),logits[1].item())

    def test_checkpoint_restore_and_corruption(self):
        model=tiny_model();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
        with tempfile.TemporaryDirectory() as root:
            manager=Checkpoints(root);state=model.trainable_state();path=manager.save("latest",model,opt,sched,{"step":1})
            with torch.no_grad():model.head.score.weight.add_(4)
            manager.load(path,model,opt,sched)
            for k,v in model.trainable_state().items():torch.testing.assert_close(v,state[k])
            (path/"state.pt").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):manager.load(path,model)
            manager.close()

if __name__=="__main__":unittest.main()
