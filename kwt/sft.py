"""Short supervised warm-start so every mode format is in the policy before RL.

The first GRPO run collapsed onto <mode:direct>: the base model almost never
emitted a valid mode tag, so the cheapest valid format was the only one that
ever scored. Here the verified recovery-v2 supervision (which mode sequence
solved each episode) is rendered into the text format and imitated briefly.

    python -m kwt.sft --out out/sft [--steps 256]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from kwt import format as fmt
from kwt import tasks

EOS = "<|im_end|>"


def _reason_text(step: dict) -> str | None:
  body = step["text"].split(">", 1)[1]
  try:
    return json.loads(body).get("text")
  except (json.JSONDecodeError, AttributeError):
    return None


def target(episode: dict, row: dict) -> str:
  """Render the episode's verified mode sequence as a kwt trace."""
  options = json.loads(row["options"])
  gold = row["answer"]
  final = (fmt.JEV_PREFIX + fmt.LETTERS[options.index(gold)]) if gold in options \
      else f"<answer>{gold}</answer>"
  reasons = [t for t in (_reason_text(s) for s in episode["supervision"]
                         if s["text"].startswith("<mode:cot>")) if t]
  if reasons:
    head = "<mode:cot>\n" + "\n".join(reasons) + "\n"
    return head + final
  if final.startswith("<mode:jev>"):
    return final
  return "<mode:direct>\n" + final


def examples(split: str = "train") -> list[dict]:
  rows = {r["id"]: r for r in tasks.load(split)}
  out = []
  with open(tasks.DATA / f"{split}.jsonl", encoding="utf-8") as handle:
    for line in handle:
      ep = json.loads(line)
      if ep["id"] in rows:
        row = rows[ep["id"]]
        text = target(ep, row)
        if fmt.parse(text, len(json.loads(row["options"]))).valid:
          out.append({"prompt": row["prompts"], "completion": text + EOS,
                      "behavior": row["behavior"]})
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--steps", type=int, default=256)
  ap.add_argument("--batch", type=int, default=4)
  ap.add_argument("--lr", type=float, default=1e-4)
  ap.add_argument("--max-len", type=int, default=512)
  ap.add_argument("--random-weights", action="store_true", help="local plumbing test")
  ap.add_argument("--examples", help="v2: jsonl from kwt.data.sft_data (prompt, completion, "
                  "tag_chars); default is the v1 recovery-v2 data")
  ap.add_argument("--tag-weight", type=float, default=5.0,
                  help="v2: loss weight on the genre+mode tag tokens (one decision token "
                  "among ~40 otherwise gets little gradient)")
  ap.add_argument("--model", help="v2: name=hf_id, e.g. qwen3-4b=Qwen/Qwen3-4B (bf16 compute)")
  ap.add_argument("--completion-window", type=int, default=0,
                  help="v2: compute vocab logits only for this many positions from the "
                  "start of the completion (f32 logits over every prompt position OOMed "
                  "Qwen3-4B on one chip); 0 = all positions (v1)")
  ap.add_argument("overrides", nargs="*")
  args = ap.parse_args()

  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top
  from flax import nnx  # pylint: disable=g-import-not-at-top
  from tunix.generate import utils as gen_utils  # pylint: disable=g-import-not-at-top

  from kwt import demo, grpo  # pylint: disable=g-import-not-at-top

  if args.random_weights:
    grpo._random_weights_patch()  # pylint: disable=protected-access
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  if args.examples:
    with open(args.examples, encoding="utf-8") as handle:
      data = [json.loads(line) for line in handle]
  else:
    data = examples()
  print(f"[sft] {len(data)} examples", flush=True)
  overrides = list(args.overrides)
  if args.model:
    name, hf_id = args.model.split("=")
    overrides = grpo.model_overrides(name, hf_id) + ["model_config.dtype=bfloat16"] + overrides
  model, tokenizer = demo.load_model(None, overrides)
  pad = tokenizer.pad_id()

  def encode(ex):
    p, c = tokenizer.encode(ex["prompt"]), tokenizer.encode(ex["completion"])
    ids = (p + c)[: args.max_len]
    weights = [1.0] * len(c)
    if ex.get("tag_chars"):
      n_tag = len(tokenizer.encode(ex["completion"][:ex["tag_chars"]]))
      weights[:n_tag] = [args.tag_weight] * min(n_tag, len(c))
    loss = ([0.0] * len(p) + weights)[: args.max_len]
    n = len(ids)
    return (ids + [pad] * (args.max_len - n), [1] * n + [0] * (args.max_len - n),
            loss + [0] * (args.max_len - n), len(p) - 1)

  if args.examples:  # truncation would cut off the answer: skip over-long examples
    before = len(data)
    window = args.completion_window or args.max_len
    data = [ex for ex in data
            if len(tokenizer.encode(ex["prompt"] + ex["completion"])) <= args.max_len
            and len(tokenizer.encode(ex["completion"])) <= window]
    print(f"[sft] kept {len(data)}/{before} examples within {args.max_len} tokens", flush=True)
  encoded = [encode(ex) for ex in data]
  warmup = min(10, max(1, args.steps // 4))
  schedule = optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup, max(args.steps, warmup + 1),
                                                args.lr * 0.1)
  opt = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.0),
                                         optax.adamw(schedule)), wrt=nnx.LoRAParam)

  W = args.completion_window

  def loss_fn(model, tokens, mask, loss_mask, start):
    inp, m = tokens[:, :-1], mask[:, :-1]
    positions = gen_utils.build_positions_from_mask(m)
    attn = gen_utils.make_causal_attn_mask(m, inp.shape[1])
    if W:
      hidden, _ = model(inp, positions, None, attn, skip_lm_head=True)  # [B, L-1, D]
      span = start[:, None] + jnp.arange(W)[None, :]  # positions predicting the completion
      valid = span <= inp.shape[1] - 1
      span = jnp.minimum(span, inp.shape[1] - 1)
      logits = model.compute_final_logits(
          jnp.take_along_axis(hidden, span[..., None], axis=1))  # [B, W, V]
      tgt = jnp.take_along_axis(tokens[:, 1:], span, axis=1)
      w = jnp.take_along_axis(loss_mask[:, 1:], span, axis=1).astype(jnp.float32) * valid
    else:
      logits, _ = model(inp, positions, None, attn)
      tgt = tokens[:, 1:]
      w = loss_mask[:, 1:].astype(jnp.float32)
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    nll = -jnp.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
    return (nll * w).sum() / jnp.maximum(w.sum(), 1.0)

  @nnx.jit
  def step(model, opt, tokens, mask, loss_mask, start):
    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam))(
        model, tokens, mask, loss_mask, start)
    opt.update(model, grads)
    return loss

  rng = np.random.default_rng(0)
  order = rng.permutation(len(encoded))
  started, losses = time.time(), []
  for i in range(args.steps):
    idx = [order[(i * args.batch + j) % len(order)] for j in range(args.batch)]
    batch = [jnp.asarray([encoded[k][f] for k in idx], dt)
             for f, dt in ((0, jnp.int32), (1, jnp.bool_), (2, jnp.float32), (3, jnp.int32))]
    losses.append(float(step(model, opt, *batch)))
    if i % 10 == 0 or i == args.steps - 1:
      print(f"[sft] step {i} loss {np.mean(losses[-10:]):.4f} "
            f"({time.time() - started:.0f}s)", flush=True)
  n = grpo.save_lora(model, str(out / "lora.npz"))
  (out / "sft.json").write_text(json.dumps({"steps": args.steps, "losses": losses}))
  print(f"[sft] saved {n} LoRA tensors", flush=True)


if __name__ == "__main__":
  main()
