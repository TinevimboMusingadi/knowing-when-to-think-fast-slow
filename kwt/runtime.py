"""Native mode switching at inference time.

The mode token the model emits changes what the runtime executes next:

  <mode:direct>  short generation (answer cap), no reasoning budget
  <mode:cot>     long generation; a later <mode:jev> hands off to a decision
  <mode:jev>     no generation at all: one forward step, logits restricted to
                 the option letters, argmax + calibrated probabilities

Everything is batched; prompts are grouped by the mode each one picked.
"""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np

from kwt import format as fmt

ROUTE_STEPS = 8
DIRECT_STEPS = 48
COT_STEPS = 512
MAX_PROMPT = 1024


@dataclasses.dataclass
class Result:
  trace: str
  modes: list[str]
  answer: str | None
  jev_probs: dict[str, float] | None
  generated_tokens: int


class SwitchingRuntime:

  def __init__(self, model, tokenizer, cache_size: int = MAX_PROMPT + COT_STEPS + 64):
    from tunix.common import configs  # pylint: disable=g-import-not-at-top
    from tunix.generate import sampler as sampler_lib  # pylint: disable=g-import-not-at-top

    self.model = model
    cfg = model.config
    self.sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=configs.CacheConfig(
            cache_size=cache_size,
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        ),
    )
    self.tok = self.sampler.tokenizer
    self.letter_ids = []
    for letter in fmt.LETTERS:
      ids = self.tok.encode(letter)
      if len(ids) != 1:
        raise ValueError(f"option letter {letter!r} is not a single token: {ids}")
      self.letter_ids.append(ids[0])

  # -- primitives ----------------------------------------------------------

  def _generate(self, prompts: list[str], steps: int) -> tuple[list[str], list[int]]:
    if not prompts:
      return [], []
    out = self.sampler(
        input_strings=prompts,
        max_generation_steps=steps,
        max_prompt_length=MAX_PROMPT,
        temperature=0.0,
        eos_tokens=[self.tok.eos_id()],
    )
    return list(out.text), [len(t) for t in out.tokens]

  def _jev(self, prefixes: list[str], num_options: list[int]) -> list[dict[str, float]]:
    """One forward step; softmax over the shown option letters only."""
    if not prefixes:
      return []
    from tunix.generate import utils as gen_utils  # pylint: disable=g-import-not-at-top

    # A plain forward pass (the sampler cannot return Qwen3 logits). Left-pad
    # to a 256 bucket so the last column is every row's next-token position.
    ids = [self.tok.encode(p) for p in prefixes]
    length = -(-max(len(i) for i in ids) // 256) * 256
    pad = self.tok.pad_id()
    tokens = jnp.asarray([[pad] * (length - len(i)) + i for i in ids], jnp.int32)
    mask = jnp.asarray([[0] * (length - len(i)) + [1] * len(i) for i in ids], jnp.bool_)
    logits, _ = self.model(
        tokens, gen_utils.build_positions_from_mask(mask), None,
        gen_utils.make_causal_attn_mask(mask, length))
    logits = np.asarray(jax.device_get(logits[:, -1, :]), np.float32)
    results = []
    for row, n in zip(logits, num_options):
      scores = row[self.letter_ids[:n]]
      probs = np.exp(scores - scores.max())
      probs /= probs.sum()
      results.append({fmt.LETTERS[i]: float(p) for i, p in enumerate(probs)})
    return results

  # -- full episode ----------------------------------------------------------

  def run(self, questions: list[str], options: list[list[str]],
          force: str | None = None) -> list[Result]:
    """Run each question through native switching (or a forced first mode)."""
    prompts = [fmt.chat_prompt(q) for q in questions]
    n = len(prompts)
    used = [0] * n
    if force is None:
      heads, _ = self._generate(prompts, ROUTE_STEPS)
      modes = []
      for head in heads:
        parsed = fmt.parse(head, 0).modes
        modes.append(parsed[0] if parsed and head.lstrip().startswith("<mode:") else None)
      # Only the mode tag itself counts; the rest of the probe is regenerated.
      used = [len(self.tok.encode(f"<mode:{m}>\n")) if m else ROUTE_STEPS for m in modes]
    else:
      modes = [force] * n

    results: list[Result | None] = [None] * n
    jev_idx, jev_prefix = [], []

    for mode, steps in (("direct", DIRECT_STEPS), ("cot", COT_STEPS)):
      idx = [i for i in range(n) if modes[i] == mode]
      texts, counts = self._generate([prompts[i] + f"<mode:{mode}>\n" for i in idx], steps)
      for i, text, c in zip(idx, texts, counts):
        trace = f"<mode:{mode}>\n" + text
        used[i] += c
        if mode == "cot" and "<mode:jev>" in text and options[i]:
          # Mid-trace switch: stop reasoning, hand off to a JEV decision.
          head = trace[: trace.index("<mode:jev>")]
          jev_idx.append(i)
          jev_prefix.append((head, prompts[i] + head + fmt.JEV_PREFIX))
          continue
        p = fmt.parse(trace, len(options[i]))
        results[i] = Result(trace, list(p.modes), p.answer, None, used[i])

    for i in range(n):
      if modes[i] == "jev" and options[i]:
        jev_idx.append(i)
        jev_prefix.append(("", prompts[i] + fmt.JEV_PREFIX))
      elif modes[i] not in ("direct", "cot") and results[i] is None:
        results[i] = Result("", [m for m in [modes[i]] if m], None, None, used[i])

    probs = self._jev([p for _, p in jev_prefix], [len(options[i]) for i in jev_idx])
    for i, (head, _), pr in zip(jev_idx, jev_prefix, probs):
      letter = max(pr, key=pr.get)
      trace = head + fmt.JEV_PREFIX + letter
      p = fmt.parse(trace, len(options[i]))
      results[i] = Result(trace, list(p.modes), p.answer, pr, used[i] + 1)
    return results  # pyrefly: ignore[bad-return]


def load_options(row: dict) -> list[str]:
  return json.loads(row["options"])
