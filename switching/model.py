"""Shared Qwen backbone, trainable mode rows, and permutation-equivariant decision head."""
import contextlib
import torch
from torch import nn
from .protocol import MODES

class ExtendedEmbedding(nn.Module):
    def __init__(self, original, count, old_size=None):
        super().__init__()
        self.original = original
        self.old_size = original.num_embeddings if old_size is None else old_size
        self.rows = nn.Parameter(original.weight.detach()[:self.old_size].mean(0).repeat(count, 1))
        self.num_embeddings = self.old_size + count
        self.embedding_dim = original.embedding_dim

    def forward(self, ids):
        old = self.original(ids.clamp(max=self.old_size - 1))
        new = nn.functional.embedding((ids - self.old_size).clamp(min=0, max=len(self.rows)-1), self.rows)
        return torch.where((ids >= self.old_size).unsqueeze(-1), new, old)

class ExtendedOutput(nn.Module):
    def __init__(self, original, embedding):
        super().__init__(); self.original = original; self.embedding = embedding

    def forward(self, hidden):
        return torch.cat((self.original(hidden)[...,:self.embedding.old_size], nn.functional.linear(hidden, self.embedding.rows)), -1)

class DecisionHead(nn.Module):
    def __init__(self, hidden_size, width=128):
        super().__init__()
        self.project = nn.Linear(hidden_size, width)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0, batch_first=True)
        self.attention = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.score = nn.Linear(width, 1)

    def forward(self, representations, valid=None):
        hidden = self.project(representations)
        hidden = self.attention(hidden, src_key_padding_mask=None if valid is None else ~valid)
        logits = self.score(hidden).squeeze(-1)
        return logits if valid is None else logits.masked_fill(~valid, -1e9)

class SwitchModel(nn.Module):
    def __init__(self, language_model, tokenizer, rank=16, alpha=32, use_lora=True):
        super().__init__()
        self.tokenizer = tokenizer
        old_size=len(tokenizer) if hasattr(tokenizer,"__len__") else language_model.get_input_embeddings().num_embeddings
        added = tokenizer.add_special_tokens({"additional_special_tokens": list(MODES)})
        if added != len(MODES):
            raise ValueError("load the unextended base tokenizer; checkpoints restore mode rows separately")
        for param in language_model.parameters(): param.requires_grad_(False)
        embedding = ExtendedEmbedding(language_model.get_input_embeddings(), added,old_size)
        language_model.set_input_embeddings(embedding)
        language_model.set_output_embeddings(ExtendedOutput(language_model.get_output_embeddings(), embedding))
        language_model.config.vocab_size = old_size + added
        if use_lora:
            from peft import LoraConfig, get_peft_model
            language_model = get_peft_model(language_model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
            embedding.rows.requires_grad_(True)
        self.lm = language_model
        self.head = DecisionHead(language_model.config.hidden_size)
        self.config = {"rank": rank, "alpha": alpha}

    @classmethod
    def load(cls, model_id, rank=16, alpha=32, dtype=torch.float32,local_files_only=False):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id,local_files_only=local_files_only)
        if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="eager",local_files_only=local_files_only)
        return cls(base, tokenizer, rank, alpha)

    def prompt_ids(self, messages):
        # Preserve Qwen's chat boundaries without forcing its built-in think prefix.
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return self.tokenizer.encode(text + "<|im_start|>assistant\n", add_special_tokens=False)

    def decision_logits(self, states, candidate_sets, device=None, bucket=512):
        device = device or next(self.parameters()).device
        counts = {len(c) for c in candidate_sets}
        if len(counts) != 1 or not counts or min(counts) < 2:
            raise ValueError("decision batches require a common candidate count >= 2")
        count = counts.pop(); rows = []
        for state, candidates in zip(states, candidate_sets):
            if len({c["id"] for c in candidates}) != count:
                raise ValueError("candidate IDs must be unique")
            for candidate in candidates:
                ids = self.tokenizer.encode(state + "\nCandidate: " + candidate["text"], add_special_tokens=False)
                if not ids or len(ids) > bucket: raise ValueError("decision input exceeds bucket")
                rows.append(ids)
        lengths = torch.tensor([len(row) for row in rows], device=device)
        ids = torch.tensor([r + [self.tokenizer.pad_token_id] * (bucket-len(r)) for r in rows], device=device)
        mask = torch.arange(bucket, device=device)[None, :] < lengths[:, None]
        # Each state/candidate branch is an independent batch row. All branches use
        # identical positions and cannot attend to other candidates, in one forward.
        backbone = self.lm.get_base_model().model if hasattr(self.lm, "get_base_model") else self.lm.model
        outputs = backbone(input_ids=ids, attention_mask=mask, position_ids=torch.arange(bucket, device=device)[None, :].expand(len(rows), -1), use_cache=False)
        reps = outputs.last_hidden_state[torch.arange(len(rows), device=device), lengths - 1]
        return self.head(reps.reshape(len(states), count, -1)).float()

    def trainable_state(self):
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def restore_trainable(self, state):
        params = dict(self.named_parameters())
        expected = {n for n, p in params.items() if p.requires_grad}
        if set(state) != expected: raise ValueError("checkpoint trainable architecture mismatch")
        with torch.no_grad():
            for name, value in state.items(): params[name].copy_(value.to(params[name].device))

    @contextlib.contextmanager
    def reference(self, state):
        current = self.trainable_state()
        self.restore_trainable(state)
        try: yield
        finally: self.restore_trainable(current)
