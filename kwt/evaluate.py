"""Compare learned switching against forced single-mode policies."""

from __future__ import annotations

import collections
import json

from kwt import format as fmt
from kwt import runtime as rt

POLICIES = ("learned", "direct", "cot", "jev")


def evaluate(runtime: rt.SwitchingRuntime, rows: list[dict], batch: int = 16,
             policies=POLICIES, keep_examples: int = 6) -> dict:
  report = {}
  for policy in policies:
    force = None if policy == "learned" else policy
    per = collections.defaultdict(lambda: collections.Counter())
    examples = []
    for start in range(0, len(rows), batch):
      chunk = rows[start:start + batch]
      results = runtime.run([r["question"] for r in chunk],
                            [rt.load_options(r) for r in chunk], force=force)
      # Progress line: also keeps the VM hang guard from firing mid-eval.
      print(f"[eval] {policy} {start + len(chunk)}/{len(rows)}", flush=True)
      for row, res in zip(chunk, results):
        opts = rt.load_options(row)
        parsed = fmt.parse(res.trace, len(opts)) if res.trace else None
        ok = bool(parsed) and fmt.is_correct(parsed, row["answer"], opts)
        for key in ("all", row["behavior"]):
          c = per[key]
          c["n"] += 1
          c["correct"] += ok
          c["valid"] += bool(parsed and parsed.valid)
          c["tokens"] += res.generated_tokens
          c["switched"] += bool(parsed and parsed.switched)
          c[f"mode:{res.modes[0] if res.modes else 'none'}"] += 1
        if len(examples) < keep_examples and (ok or policy == "learned"):
          examples.append({"behavior": row["behavior"], "question": row["question"],
                           "gold": row["answer"], "trace": res.trace, "correct": ok,
                           "jev_probs": res.jev_probs, "tokens": res.generated_tokens})
    report[policy] = {
        key: {
            "n": c["n"],
            "accuracy": c["correct"] / c["n"],
            "valid": c["valid"] / c["n"],
            "mean_tokens": c["tokens"] / c["n"],
            "switch_rate": c["switched"] / c["n"],
            "first_mode": {k[5:]: v / c["n"] for k, v in c.items() if k.startswith("mode:")},
        }
        for key, c in per.items()
    }
    report[policy]["examples"] = examples
  return report


def table(report: dict) -> str:
  lines = ["| policy | accuracy | valid | mean tokens | first mode (jev/direct/cot) |",
           "|---|---|---|---|---|"]
  for policy, by in report.items():
    a = by["all"]
    fm = a["first_mode"]
    lines.append(
        f"| {policy} | {a['accuracy']:.1%} | {a['valid']:.1%} | {a['mean_tokens']:.1f} | "
        f"{fm.get('jev', 0):.0%} / {fm.get('direct', 0):.0%} / {fm.get('cot', 0):.0%} |")
  return "\n".join(lines)


def dump(report: dict, path: str) -> None:
  with open(path, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
