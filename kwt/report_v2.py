"""Aggregate v2 eval outputs into paper tables, with paired bootstrap CIs.

    python -m kwt.report_v2 --runs runs/kwt/final-a runs/kwt/final-b --split test

Each run dir holds <name>/completions_<split>_<control>.jsonl from
kwt.evaluate_v2. Every completion is re-scored here (so scoring fixes never
need a re-run), and model differences use a paired bootstrap over items.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from kwt import format as fmt
from kwt.data import labels

LAM = labels.LAMBDA


def load(run_dirs, split, items):
  """name -> control -> id -> (correct, tokens, first_mode)."""
  out = {}
  for d in run_dirs:
    for sub in sorted(pathlib.Path(d).iterdir()):
      for f in sub.glob(f"completions_{split}_*.jsonl"):
        control = f.stem.split("_")[-1]
        rows = {}
        for line in f.open(encoding="utf-8"):
          r = json.loads(line)
          it = items[r["id"]]
          p = fmt.parse_v2(r["completion"], len(it["options"]), control)
          ok = p.valid and fmt.is_correct_v2(p.trace, str(it["gold"]), it["options"])
          rows[r["id"]] = (ok, r["tokens"], p.trace.first_mode)
        out.setdefault(sub.name, {})[control] = rows
  return out


def utility(rows, ids):
  c = np.array([rows[i][0] for i in ids], float)
  t = np.array([rows[i][1] for i in ids], float)
  return c, t, c - LAM * t


def boot_diff(a, b, n=4000, seed=0):
  """Paired bootstrap 95% CI of mean(a - b)."""
  rng = np.random.default_rng(seed)
  d = a - b
  idx = rng.integers(0, len(d), (n, len(d)))
  means = d[idx].mean(axis=1)
  return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--runs", nargs="+", required=True)
  ap.add_argument("--split", default="test")
  ap.add_argument("--items", default="data/kwt-v2/labelled.jsonl")
  ap.add_argument("--ref", default="e3", help="model compared against the others")
  args = ap.parse_args()
  items = {}
  with open(args.items, encoding="utf-8") as handle:
    for line in handle:
      r = json.loads(line)
      items[r["id"]] = r
  data = load(args.runs, args.split, items)
  print(f"## {args.split}\n")
  print("| model | control | n | acc | tokens | utility | JEV/direct/CoT |")
  print("|---|---|---|---|---|---|---|")
  for name in sorted(data):
    for control in ("auto", "jev", "direct", "cot"):
      rows = data[name].get(control)
      if not rows:
        continue
      ids = sorted(rows)
      c, t, u = utility(rows, ids)
      modes = [rows[i][2] for i in ids]
      share = "/".join(f"{modes.count(m) / len(modes):.0%}" for m in ("jev", "direct", "cot"))
      print(f"| {name} | {control} | {len(ids)} | {c.mean():.1%} | {t.mean():.1f} | {u.mean():.3f} | {share} |")
  ref = data.get(args.ref, {}).get("auto")
  if ref:
    print(f"\nPaired bootstrap, utility difference vs {args.ref}/auto (95% CI):\n")
    for name in sorted(data):
      for control in ("auto", "cot"):
        rows = data[name].get(control)
        if not rows or (name == args.ref and control == "auto"):
          continue
        ids = sorted(set(rows) & set(ref))
        _, _, ua = utility(ref, ids)
        _, _, ub = utility(rows, ids)
        m, lo, hi = boot_diff(ua, ub)
        ca, _, _ = utility(ref, ids)
        cb, _, _ = utility(rows, ids)
        ma, la, ha = boot_diff(ca, cb)
        print(f"- {args.ref}/auto − {name}/{control}: utility {m:+.3f} [{lo:+.3f}, {hi:+.3f}], "
              f"accuracy {ma:+.1%} [{la:+.1%}, {ha:+.1%}]  (n={len(ids)})")


if __name__ == "__main__":
  main()
