"""Single-turn tasks rendered from the verified recovery-v2 episodes.

Lookup and clarification episodes are multi-turn and are left out of this
demo. Every remaining episode has an exact gold value. Whether lettered options
are shown decides whether JEV is available at all:

  fast            options shown   -> one JEV decision should suffice
  reason_decide   options shown   -> reason first, then decide (cot -> jev)
  direct          no options      -> short open answer
  reasoning       half with, half without options
"""

from __future__ import annotations

import hashlib
import json
import pathlib

from kwt import format as fmt

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "recovery-v2"
BEHAVIORS = ("fast", "direct", "reasoning", "reason_decide")
_LEAK = "Please solve this task: "  # wording that encoded the old forced route


def _stable_bit(text: str) -> int:
  return hashlib.sha256(text.encode()).digest()[0] & 1


def _render(episode: dict) -> dict:
  visible, behavior = episode["visible"], episode["behavior"]
  options = [c["text"] for c in visible["candidates"] if c["kind"] == "answer"]
  shown = behavior in ("fast", "reason_decide") or (
      behavior == "reasoning" and _stable_bit(episode["id"])
  )
  options = options if shown else []
  prompt = visible["prompt"].removeprefix(_LEAK)
  question = fmt.render_question(prompt, options)
  return {
      "id": episode["id"],
      "prompts": fmt.chat_prompt(question),
      "question": question,
      "answer": str(episode["oracle"]["answer"]),
      "options": json.dumps(options),
      "family": episode["family"],
      "behavior": behavior,
  }


def load(split: str, limit: int | None = None) -> list[dict]:
  rows = []
  with open(DATA / f"{split}.jsonl", encoding="utf-8") as handle:
    for line in handle:
      episode = json.loads(line)
      if episode["behavior"] in BEHAVIORS:
        rows.append(_render(episode))
  # Interleave behaviors so small prefixes stay balanced.
  rows.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
  return rows[:limit] if limit else rows


def create_dataset(split: str = "train", limit: int | None = None):
  """Tunix CLI `data_module` entry point."""
  import grain  # pylint: disable=g-import-not-at-top

  return grain.MapDataset.source(load(split, limit))
