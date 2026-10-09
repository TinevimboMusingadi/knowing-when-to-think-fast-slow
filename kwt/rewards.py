"""GRPO reward functions, loaded by the Tunix CLI from `reward_functions`.

Tunix treats every public function defined here as a reward and sums them.
There is deliberately no reward for picking a particular mode: routing has to
come out of correctness versus token cost.
"""

import json

from kwt import format as fmt

TOKEN_COST = 0.002  # per ~token (4 chars); 150 tokens of CoT costs 0.3
MAX_COST = 0.6  # never let cost outweigh a correct answer


def _parsed(completions, options):
  return [
      fmt.parse(c, len(json.loads(str(o)))) for c, o in zip(completions, options)
  ]


def correctness(prompts, completions, answer, options, **kwargs):
  del prompts, kwargs
  return [
      1.0 if fmt.is_correct(p, str(a), json.loads(str(o))) else 0.0
      for p, a, o in zip(_parsed(completions, options), answer, options)
  ]


def well_formed(prompts, completions, options, **kwargs):
  del prompts, kwargs
  return [0.1 if p.valid else -0.2 for p in _parsed(completions, options)]


def compute_cost(prompts, completions, **kwargs):
  del prompts, kwargs
  return [-min(MAX_COST, TOKEN_COST * len(c) / 4) for c in completions]
