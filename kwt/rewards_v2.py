"""v2 GRPO rewards. Tunix sums every public function here.

Changes from v1 (kwt/rewards.py), each from a v1 failure:
  compute_cost   charged only on correct answers: v1 charged wrong answers too,
                 so short-and-wrong always beat long-and-wrong; 4x cheaper.
  well_formed    a forced mode that is not obeyed is the worst outcome (-1).
  faithful       false `expr = value` lines cost reward (v1 run 3 wrote
                 `(7212*7213)+59=79981` and still got full credit).
  genre          small bonus for the right genre tag (ablated in E5).
  route_regret   suggested-vs-actual mode via measured utilities (see
                 kwt/data/labels.py); BETA is annealed to 0 by the trainer.

Dataset columns used: answer, options (JSON list), control ("auto" or a mode),
genre, and utilities (JSON {mode: U}) for route_regret.
"""

from __future__ import annotations

import json
import os

from kwt import checks
from kwt import format as fmt

TOKEN_COST = 0.0005  # per ~token (4 chars)
MAX_COST = 0.3
FALSE_CLAIM = 0.05
MAX_FALSE = 0.2
GENRE_BONUS = 0.05
BETA = float(os.environ.get("KWT_BETA", "0.5"))  # route_regret weight; 0 ablates it
# Tunix does not pass the step to reward functions (and calls them once per
# rollout micro-batch, not per step), so route_regret counts the completions it
# has scored and fades BETA linearly to 0 over this many: 0.6 * 150 steps * 4
# prompts * 8 samples = 2880. Real outcomes then decide the routing.
ANNEAL_COMPLETIONS = int(os.environ.get("KWT_ANNEAL_COMPLETIONS", "2880"))
_seen = 0
# Ablations (plan E2-E5): comma-separated reward terms to zero, e.g.
# KWT_REWARD_OFF=route_regret,faithful,genre for the outcome-only run E2.
OFF = {t for t in os.environ.get("KWT_REWARD_OFF", "").replace("+", ",").split(",") if t}  # "+" too: gcloud metadata cannot carry commas


def _column(kwargs: dict, name: str, n: int) -> list:
  """Tunix passes dataset columns as numpy arrays: `array or default` raises."""
  value = kwargs.get(name)
  return [None] * n if value is None else list(value)


def _off(name: str, n: int) -> list[float] | None:
  return [0.0] * n if name in OFF else None


def _parsed(completions, options, control):
  return [fmt.parse_v2(c, len(json.loads(str(o))), str(k))
          for c, o, k in zip(completions, options, control)]


def _correct(p: fmt.ParsedV2, answer: str, options: str) -> bool:
  return p.valid and fmt.is_correct_v2(p.trace, str(answer), json.loads(str(options)))


def correctness(prompts, completions, answer, options, control, **kwargs):
  del prompts, kwargs
  return [float(_correct(p, a, o))
          for p, a, o in zip(_parsed(completions, options, control), answer, options)]


def well_formed(prompts, completions, options, control, **kwargs):
  del prompts, kwargs
  out = []
  for p in _parsed(completions, options, control):
    if not p.obeyed:
      out.append(-1.0)
    else:
      out.append(0.1 if p.trace.valid else -0.2)
  return out


def compute_cost(prompts, completions, answer, options, control, **kwargs):
  del prompts, kwargs
  return [-min(MAX_COST, TOKEN_COST * len(c) / 4) if _correct(p, a, o) else 0.0
          for c, p, a, o in zip(completions, _parsed(completions, options, control),
                                answer, options)]


def faithful(prompts, completions, **kwargs):
  if (zeros := _off("faithful", len(completions))) is not None:
    return zeros
  del prompts, kwargs
  return [-min(MAX_FALSE, FALSE_CLAIM * len(checks.false_claims(c))) for c in completions]


def genre(prompts, completions, options, control, **kwargs):
  if (zeros := _off("genre", len(completions))) is not None:
    return zeros
  del prompts
  gold = _column(kwargs, "genre", len(completions))
  return [GENRE_BONUS if p.genre is not None and p.genre == g else 0.0
          for p, g in zip(_parsed(completions, options, control), gold)]


def route_regret(prompts, completions, options, control, **kwargs):
  if (zeros := _off("route_regret", len(completions))) is not None:
    return zeros
  """-BETA * (U* - U_actual) under auto control; 0 when the chosen mode is as
  good as the best one, so near-ties are never punished."""
  del prompts
  global _seen
  beta = BETA * max(0.0, 1.0 - _seen / ANNEAL_COMPLETIONS) if ANNEAL_COMPLETIONS else BETA
  _seen += len(completions)
  utils = _column(kwargs, "utilities", len(completions))
  out = []
  for p, k, u in zip(_parsed(completions, options, control), control, utils):
    if str(k) != "auto" or not u or not p.trace.first_mode:
      out.append(0.0)
      continue
    u = json.loads(str(u)) if isinstance(u, str) else u
    actual = u.get(p.trace.first_mode)
    if actual is None:  # a mode that is not available for this item
      out.append(-beta)
      continue
    out.append(-beta * (max(u.values()) - actual))
  return out
