"""Blog figures/tables from a fetched run (scripts/tpu.sh fetch <run>).

    python -m kwt.report runs/kwt/full3 [--before runs/kwt/full1/eval_before.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib

BEHAVIORS = ("fast", "direct", "reasoning", "reason_decide")


def steps(log: pathlib.Path) -> list[dict]:
  if not log.exists():
    return []
  text = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
  return [json.loads(line[7:]) for line in text.splitlines() if line.startswith("[step] ")]


def accuracy_vs_tokens(evals: dict[str, dict], path: pathlib.Path) -> None:
  import matplotlib  # pylint: disable=g-import-not-at-top
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

  fig, ax = plt.subplots(figsize=(6.4, 4.2))
  marks = {"before": "o", "after": "s"}
  for stage, rep in evals.items():
    for policy, by in rep.items():
      a = by["all"]
      ax.scatter(a["mean_tokens"], a["accuracy"], marker=marks.get(stage, "o"), s=60,
                 label=f"{policy} ({stage})")
      ax.annotate(f"{policy}", (a["mean_tokens"], a["accuracy"]), fontsize=8,
                  xytext=(4, 4), textcoords="offset points")
  ax.set_xlabel("mean generated tokens per question")
  ax.set_ylabel("accuracy (held-out val)")
  ax.set_title("Accuracy vs. compute: learned switching vs. fixed modes")
  ax.grid(alpha=0.3)
  ax.legend(fontsize=7, ncol=2)
  fig.tight_layout()
  fig.savefig(path, dpi=150)


def training_curves(rows: list[dict], path: pathlib.Path, window: int = 10) -> None:
  import matplotlib  # pylint: disable=g-import-not-at-top
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

  def smooth(key):
    vals = [r[key] for r in rows]
    return [sum(vals[max(0, i - window + 1):i + 1]) / len(vals[max(0, i - window + 1):i + 1])
            for i in range(len(vals))]

  fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
  for key in ("reward", "correct", "valid"):
    a1.plot(smooth(key), label=key)
  a1.set_title("GRPO training (rolling mean)")
  a1.set_xlabel("rollout batch")
  a1.legend()
  for mode in ("jev", "direct", "cot", "none"):
    a2.plot(smooth(f"mode_{mode}"), label=mode)
  a2.set_title("first mode chosen during rollouts")
  a2.set_xlabel("rollout batch")
  a2.legend()
  for ax in (a1, a2):
    ax.grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(path, dpi=150)


def table(rep: dict, title: str) -> str:
  lines = [f"**{title}**", "", "| policy | behavior | accuracy | valid | mean tokens | jev/direct/cot first |",
           "|---|---|---|---|---|---|"]
  for policy, by in rep.items():
    for key in ("all", *BEHAVIORS):
      if key not in by:
        continue
      a, fm = by[key], by[key]["first_mode"]
      lines.append(f"| {policy} | {key} | {a['accuracy']:.0%} | {a['valid']:.0%} | "
                   f"{a['mean_tokens']:.0f} | {fm.get('jev', 0):.0%} / {fm.get('direct', 0):.0%}"
                   f" / {fm.get('cot', 0):.0%} |")
  return "\n".join(lines)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("run")
  ap.add_argument("--before")
  ap.add_argument("--figures", default="docs/figures")
  args = ap.parse_args()
  run = pathlib.Path(args.run)
  figs = pathlib.Path(args.figures)
  figs.mkdir(parents=True, exist_ok=True)

  evals = {}
  for stage, path in (("before", args.before or run / "eval_before.json"),
                      ("after", run / "eval_after.json")):
    path = pathlib.Path(path)
    if path.exists():
      evals[stage] = {k: v for k, v in json.loads(path.read_text()).items()}
  clean = {s: {p: {k: v for k, v in by.items() if k != "examples"} for p, by in r.items()}
           for s, r in evals.items()}
  if clean:
    accuracy_vs_tokens(clean, figs / "kwt-accuracy-vs-tokens.png")
  rows = steps(run / "logs" / "job.log")
  if rows:
    training_curves(rows, figs / "kwt-training.png")
  out = [table(clean[s], f"{s} GRPO") for s in clean]
  out.append(f"steps logged: {len(rows)}")
  (run / "report.md").write_text("\n\n".join(out), encoding="utf-8")
  print("\n\n".join(out))


if __name__ == "__main__":
  main()
