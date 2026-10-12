"""Fresh-inference demo of native mode switching.

    python -m kwt.demo --lora runs/kwt/<run>/lora.npz            # val examples
    python -m kwt.demo --lora ... --ask "Is 17 prime?" --options yes no
    python -m kwt.demo --lora ... --record docs/demo-kwt.json     # save for the blog
"""

from __future__ import annotations

import argparse
import json

from kwt import grpo, tasks
from kwt import format as fmt

COLORS = {"jev": "\033[95m", "direct": "\033[92m", "cot": "\033[94m"}
RESET = "\033[0m"


def colorize(trace: str) -> str:
  for mode, color in COLORS.items():
    trace = trace.replace(f"<mode:{mode}>", f"{color}<mode:{mode}>{RESET}")
  return trace


def load_model(lora: str | None, overrides: list[str]):
  from tunix.cli.utils import model as model_lib  # pylint: disable=g-import-not-at-top
  from tunix.rl import rl_cluster  # pylint: disable=g-import-not-at-top

  pipe = grpo.build(overrides)
  tokenizer = pipe._get_tokenizer()  # pylint: disable=protected-access
  mesh = pipe.create_role_to_mesh()[rl_cluster.Role.ACTOR]
  model, _ = model_lib.create_model(
      dict(pipe.config["actor_model_config"]), pipe.config["tokenizer_config"], mesh)
  if lora:
    grpo.load_lora(model, lora)
  return model, tokenizer


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--lora")
  ap.add_argument("--ask")
  ap.add_argument("--options", nargs="*", default=[])
  ap.add_argument("--n", type=int, default=8)
  ap.add_argument("--record")
  ap.add_argument("overrides", nargs="*")
  args = ap.parse_args()

  from kwt import runtime  # pylint: disable=g-import-not-at-top

  model, tokenizer = load_model(args.lora, args.overrides)
  rt = runtime.SwitchingRuntime(model, tokenizer)
  if args.ask:
    rows = [{"question": fmt.render_question(args.ask, args.options),
             "options": json.dumps(args.options), "answer": "", "behavior": "user"}]
  else:
    rows = tasks.load("val", args.n)
  results = rt.run([r["question"] for r in rows], [runtime.load_options(r) for r in rows])
  records = []
  for row, res in zip(rows, results):
    print("=" * 72)
    print(row["question"])
    print("-" * 72)
    print(colorize(res.trace))
    if res.jev_probs:
      print("JEV probabilities:", {k: round(v, 3) for k, v in res.jev_probs.items()})
    print(f"[{' -> '.join(res.modes) or 'no mode'}] tokens={res.generated_tokens} "
          f"gold={row['answer']}")
    records.append({"question": row["question"], "gold": row["answer"],
                    "trace": res.trace, "modes": res.modes,
                    "jev_probs": res.jev_probs, "tokens": res.generated_tokens})
  if args.record:
    with open(args.record, "w", encoding="utf-8") as handle:
      json.dump({"fresh_inference": True, "lora": args.lora, "records": records},
                handle, indent=2)


if __name__ == "__main__":
  main()
