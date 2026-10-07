import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from switching.model import SwitchModel,DecisionHead
from switching.batching import pack
from switching.storage import Checkpoints,checksum
from switching.train import grpo_loss,advantages,memory_headroom,is_memory_exhaustion,project_training_seconds,ensure_all_gradients,gradient_norm
from switching.optim import DeviceAdamW
from switching.inspect_checkpoint import inspect

class TinyTokenizer:
    pad_token_id=0
    def add_special_tokens(self,config):return 3
    def encode(self,text,add_special_tokens=False):return [1+ord(c)%60 for c in text]

def tiny_model():
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    config._attn_implementation="eager"
    return SwitchModel(Qwen3ForCausalLM(config),TinyTokenizer(),rank=2,alpha=4)

class TrainingTests(unittest.TestCase):
    def test_selected_projection_matches_full_logps_and_gradients(self):
        from switching.runtime import token_logps
        model=tiny_model().eval()
        for prompt,completion in (([2,64,3],[4,63]),([2]*490,[3]*20)):
            values=prompt+completion;ids=torch.tensor([values+[0]*(512-len(values))])
            mask=torch.arange(512)[None,:]<len(values)
            model.zero_grad(set_to_none=True)
            full=model.lm(input_ids=ids,attention_mask=mask,use_cache=False).logits[0].float()
            expected=full[len(prompt)-1:len(values)-1].log_softmax(-1).gather(1,torch.tensor(completion)[:,None]).squeeze(1)
            expected.sum().backward()
            gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            widths=[]
            handle=model.lm.get_output_embeddings().register_forward_pre_hook(lambda module,args:widths.append(args[0].shape[1]))
            try:actual=token_logps(model,prompt,completion)
            finally:handle.remove()
            torch.testing.assert_close(actual,expected.detach(),atol=1e-6,rtol=1e-6)
            actual.sum().backward()
            self.assertEqual(widths,[32])
            for n,p in model.named_parameters():
                if n in gradients:torch.testing.assert_close(p.grad,gradients[n],atol=2e-5,rtol=1e-5)

    def test_real_cpu_worker_runs_sft_then_dual_head_rl(self):
        import json
        from switching.train import worker
        model=tiny_model();candidates=[{"id":"a","text":"1","value":1},{"id":"b","text":"2","value":2}]
        generation=[([2,64,3,4],[-100,-100,3,4])];decisions=[("One value",candidates,0)]
        samples=[{"reward":1. if i%2==0 else -1.,"tokens":3,"error":None,"trace":[{"kind":"tokens","prompt":[2,64,3],"completion":[4+i%2,63]},{"kind":"decision","state":"One value","candidates":candidates,"choice":i%2,"bucket":32}]} for i in range(4)]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/"data.jsonl";data.write_text(json.dumps({"id":"one","behavior":"fast"})+"\n")
            config={"seed":42,"model":"tiny-test","lora_rank":2,"lora_alpha":4,"learning_rate":.001,"rl_learning_rate":.0001,"budget":{"sft":20,"rl":15},"buckets":[32],"effective_batch":1,"checkpoint_interval":50,"max_rl_steps":1,"group_size":4,"max_generated_tokens":512,"max_transitions":4,"max_lookups":2,"clip_epsilon":.2,"kl_coefficient":.02}
            path=root/"config.json";path.write_text(json.dumps(config))
            args=SimpleNamespace(config=str(path),device="cpu",phase="sft",output=str(root/"sft"),resume=None,restore_optimizer=False,data=str(data),validation=str(data),hourly_rate=1.,started=None,gcs=None,microbatch=1,max_steps=1)
            with patch("switching.train.SwitchModel.load",return_value=model),patch("switching.train.supervised_items",return_value=(generation,decisions)),patch("switching.train.rollout_group",return_value=samples):
                worker(0,args)
                checkpoint=root/"sft/checkpoints"/json.loads((root/"sft/checkpoints/best-sft-rank0.json").read_text())["directory"]
                args.phase="rl";args.output=str(root/"rl");args.resume=str(checkpoint)
                worker(0,args)
            metrics=[json.loads(line) for line in (root/"rl/metrics.jsonl").read_text().splitlines()]
            self.assertEqual(metrics[0]["step"],1);self.assertGreater(metrics[0]["gradient_norm"],0.)
            self.assertTrue((root/"rl/checkpoints/latest-rank0.json").exists())
            self.assertTrue((root/"rl/rl-episodes-rank0.jsonl").exists())
            events=[json.loads(line) for line in (root/"rl/rl-progress-rank0.jsonl").read_text().splitlines()]
            phases=[event for event in events if event["event"]=="phase-complete"]
            self.assertEqual([event["phase"] for event in phases],["policy-scoring","reference-scoring"]+["backward"]*4+["optimizer"])
            self.assertTrue(all(event["seconds"]>=0 for event in phases))
            self.assertEqual([event["sample"] for event in phases if event["phase"]=="backward"],list(range(4)))
    def test_gradient_inspection_does_not_poison_healthy_gradients(self):
        model=torch.nn.Linear(2,1);model.weight.grad=torch.tensor([[3.,4.]]);model.bias.grad=torch.tensor([0.])
        self.assertEqual(float(gradient_norm(model)),5.)
        model.bias.grad.fill_(float('nan'));before=model.weight.grad.clone()
        self.assertFalse(torch.isfinite(gradient_norm(model)))
        torch.testing.assert_close(model.weight.grad,before)
    def test_unused_head_keeps_identical_replica_gradient_lists(self):
        model=tiny_model();ids=torch.tensor([[2,64,3,4]])
        model.lm(input_ids=ids,labels=ids,use_cache=False).loss.backward()
        self.assertTrue(any(p.grad is None for p in model.head.parameters()))
        before={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        ensure_all_gradients(model)
        self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(torch.count_nonzero(p.grad)==0 for p in model.head.parameters()))
        for n,p in model.named_parameters():
            if n in before:torch.testing.assert_close(p.grad,before[n])
    def test_projection_separates_compilation_and_reserves_future_cost(self):
        self.assertEqual(project_training_seconds([10,11,10,10,10],100,[30,50]),1475)
        with self.assertRaises(ValueError):project_training_seconds([],100,[50])

    def test_device_warmup_matches_reference_and_legacy_resume(self):
        a=torch.nn.Parameter(torch.tensor([1.0]));b=torch.nn.Parameter(a.detach().clone())
        custom=DeviceAdamW([a],lr=.01,warmup_steps=10);reference=torch.optim.AdamW([b],lr=.01)
        for step in range(1,5):
            a.grad=torch.tensor([.4]);b.grad=a.grad.clone();reference.param_groups[0]["lr"]=.01*step/10
            custom.step();reference.step();torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        legacy=custom.state_dict()
        for group in legacy["param_groups"]:group.pop("base_lr");group.pop("warmup_steps")
        restored=DeviceAdamW([a],lr=.01,warmup_steps=10);restored.load_state_dict(legacy)
        self.assertEqual(restored.param_groups[0]["base_lr"],.01)
        self.assertEqual(restored.param_groups[0]["warmup_steps"],10)

    def test_device_adamw_matches_reference_and_restores(self):
        a=torch.nn.Parameter(torch.tensor([1.0,-2.0]));b=torch.nn.Parameter(a.detach().clone())
        custom=DeviceAdamW([a],lr=.01);reference=torch.optim.AdamW([b],lr=.01)
        for index in range(5):
            a.grad=torch.tensor([.2+index*.1,-.4]);b.grad=a.grad.clone()
            custom.step();reference.step()
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        restored=DeviceAdamW([a],lr=.01);restored.load_state_dict(custom.state_dict())
        self.assertEqual(restored.state[a]["step"].item(),5)
        mixed=torch.nn.Parameter(a.detach().to(torch.bfloat16))
        mixed_optimizer=DeviceAdamW([mixed]);mixed_optimizer.load_state_dict(custom.state_dict())
        self.assertEqual(mixed_optimizer.state[mixed]["exp_avg"].dtype,torch.float32)

    def test_pjrt_memory_counters(self):
        self.assertAlmostEqual(memory_headroom({"bytes_used":40,"bytes_limit":100,"peak_bytes_used":50}),.5)
        self.assertAlmostEqual(memory_headroom({"kb_free":80,"kb_total":100}),.8)
        with self.assertRaises(ValueError):memory_headroom({})
        self.assertTrue(is_memory_exhaustion(ValueError("XLA:TPU compile permanent error. Ran out of memory in memory space hbm.")))
        self.assertFalse(is_memory_exhaustion(ValueError("invalid attention mask")))

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

    def test_padded_base_vocabulary_routes_actual_new_token_ids(self):
        class PaddedTokenizer(TinyTokenizer):
            def __len__(self):return 60
        base=Qwen3ForCausalLM(Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8))
        model=SwitchModel(base,PaddedTokenizer(),rank=2,alpha=4)
        ids=torch.tensor([[60,61,62,2]])
        output=model.lm(input_ids=ids,labels=ids,use_cache=False)
        self.assertEqual(output.logits.shape[-1],63)
        output.loss.backward()
        rows=next(p for n,p in model.named_parameters() if n.endswith("rows"))
        self.assertTrue(torch.all(rows.grad.abs().sum(-1)>0))

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
        adv=advantages([1.,-1.]);loss=sum(grpo_loss(new[i:i+1],old[i:i+1],old[i:i+1],adv[i]) for i in range(2))/2;loss.backward();opt.step()
        self.assertGreater(logits[0].item(),logits[1].item())

    def test_old_grpo_overflows_and_repaired_tails_have_finite_gradients(self):
        new=torch.tensor([-1000.],requires_grad=True);old=new.detach().clone();reference=torch.zeros(1)
        delta=reference-new
        original=(delta.exp()-delta-1).mean()
        self.assertFalse(torch.isfinite(original))
        loss=grpo_loss(new,old,reference,0.)
        loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(new.grad).all())
        with self.assertRaises(ValueError):grpo_loss(torch.tensor([float('nan')]),old,reference,0.)

    def test_grpo_matches_unbounded_formula_in_regular_range(self):
        new=torch.tensor([-1.1,-2.2],requires_grad=True);old=torch.tensor([-1.,-2.]);ref=torch.tensor([-1.2,-2.1])
        ratio=(new.double()-old.double()).exp();delta=ref.double()-new.double()
        expected=(-torch.minimum(ratio*.7,ratio.clamp(.8,1.2)*.7)+.02*(torch.expm1(delta)-delta)).mean()
        actual=grpo_loss(new,old,ref,.7)
        torch.testing.assert_close(actual.double(),expected,atol=1e-7,rtol=1e-6)
        expected_grad=torch.autograd.grad(expected,new,retain_graph=True)[0]
        actual_grad=torch.autograd.grad(actual,new)[0]
        torch.testing.assert_close(actual_grad,expected_grad,atol=1e-7,rtol=1e-6)

    def test_grpo_updates_qwen_adapter_and_decision_head_with_frozen_reference(self):
        from switching.runtime import action_logps
        torch.manual_seed(42);model=tiny_model().eval();reference=model.trainable_state()
        candidates=[{"id":"a","text":"1","value":1},{"id":"b","text":"2","value":2}]
        traces=[[{"kind":"tokens","prompt":[2,64,3],"completion":[4+i,63]},{"kind":"decision","state":"A value is 1.","candidates":candidates,"choice":i,"bucket":32}] for i in range(2)]
        with torch.no_grad():old=[action_logps(model,t).detach() for t in traces]
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        for trace,previous,advantage in zip(traces,old,[1.,-1.]):
            loss=grpo_loss(action_logps(model,trace),previous,previous,advantage)/2
            loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step();current=model.trainable_state()
        self.assertTrue(any(not torch.equal(current[k],v) for k,v in reference.items() if "lora_B" in k))
        self.assertTrue(any(not torch.equal(current[k],v) for k,v in reference.items() if k.startswith("head.")))
        with model.reference(reference):
            for k,v in model.trainable_state().items():torch.testing.assert_close(v,reference[k])
        for k,v in model.trainable_state().items():torch.testing.assert_close(v,current[k])

    def test_checkpoint_restore_and_corruption(self):
        model=tiny_model();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
        with tempfile.TemporaryDirectory() as root:
            manager=Checkpoints(root);state=model.trainable_state();path=manager.save("latest",model,opt,sched,{"step":1})
            self.assertFalse(inspect(path)["parameter_updates_verified"])
            with torch.no_grad():model.head.score.weight.add_(4)
            manager.load(path,model,opt,sched)
            for k,v in model.trainable_state().items():torch.testing.assert_close(v,state[k])
            import json
            payload=torch.load(path/"state.pt",weights_only=False);payload["xla_rng"]=123
            torch.save(payload,path/"state.pt")
            manifest=json.loads((path/"manifest.json").read_text());manifest["files"]["state.pt"]=checksum(path/"state.pt")
            (path/"manifest.json").write_text(json.dumps(manifest));(path/"COMPLETE").write_text(checksum(path/"manifest.json"))
            manager.load(path,model)  # TPU RNG metadata must not require XLA on CPU.
            (path/"state.pt").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):manager.load(path,model)
            manager.close()

class WorkerFailureTests(unittest.TestCase):
    def test_worker_exception_is_saved_before_parent_wait(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from switching.train import worker
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            with patch("switching.train._worker_impl",side_effect=RuntimeError("replica failure")):
                with self.assertRaisesRegex(RuntimeError,"replica failure"):
                    worker(2,SimpleNamespace(output=folder))
            self.assertIn("replica failure",(Path(folder)/"worker-error-2.txt").read_text())

if __name__=="__main__":unittest.main()
