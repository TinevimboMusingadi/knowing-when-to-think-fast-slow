"""Figures for the v2 blog post, from saved eval outputs (runs/kwt/final-*).

    python docs/make_figures_v2.py

Writes docs/figures/v2-*.png. Re-scores every completion (kwt.report_v2.load),
so the figures always match the tables.
"""

from __future__ import annotations

import glob
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kwt import report_v2  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
# Reference palette (dataviz skill), fixed slot order; gray for baselines.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def style(ax, title, xlabel, ylabel):
  ax.set_facecolor(SURFACE)
  ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
  ax.set_xlabel(xlabel, color=INK2)
  ax.set_ylabel(ylabel, color=INK2)
  ax.grid(True, color=GRID, linewidth=0.8)
  ax.set_axisbelow(True)
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)
  for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
  ax.tick_params(colors=INK2)


def load_all(split):
  items = {}
  with open(ROOT / "data" / "kwt-v2" / "labelled.jsonl", encoding="utf-8") as handle:
    for line in handle:
      r = json.loads(line)
      items[r["id"]] = r
  runs = sorted(glob.glob(str(ROOT / "runs" / "kwt" / "final-*")))
  return report_v2.load(runs, split, items), items


def seed_of(name):
  """e2 -> ('e2', 0), e2s1 / e2-s1 -> ('e2', 1), e3bs3 -> ('e3b', 3)."""
  m = re.fullmatch(r"(e2|e3b)(?:-?s(\d))?", name)
  return (m.group(1), int(m.group(2) or 0)) if m else (name, None)


def stats(rows):
  ids = sorted(rows)
  c, t, _ = report_v2.utility(rows, ids)
  return c.mean() * 100, t.mean()


def fig_accuracy_tokens(data):
  fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
  fig.patch.set_facecolor(SURFACE)
  pts = []
  for name, ctl in data.items():
    fam, seed = seed_of(name)
    if fam not in ("e2", "e3b"):
      continue
    if "auto" in ctl:
      pts.append((fam, *stats(ctl["auto"])))
  for fam, color, marker, label in (("e2", ORANGE, "s", "RL, outcome-only reward"),
                                    ("e3b", BLUE, "o", "RL + suggested-vs-actual reward")):
    xs = [p[2] for p in pts if p[0] == fam]
    ys = [p[1] for p in pts if p[0] == fam]
    ax.scatter(xs, ys, s=70, color=color, marker=marker, edgecolor=SURFACE, linewidth=2,
               label=f"{label} (one dot per seed)", zorder=3)
  acc, tok = stats(data["sft"]["auto"])
  ax.scatter([tok], [acc], s=70, color=GRAY, marker="D", edgecolor=SURFACE, linewidth=2,
             label="SFT only (model picks)", zorder=3)
  acc_c, tok_c = stats(data["e3b"]["cot"])
  ax.scatter([tok_c], [acc_c], s=110, color=INK, marker="*", label="always reason (same weights)",
             zorder=3)
  ax.annotate("same accuracy,\n~30% fewer tokens", xy=(tok_c, acc_c), xytext=(tok_c - 14, acc_c - 2.2),
              color=INK2, fontsize=9, arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
  style(ax, "Learned routing vs always reasoning (sealed test, 799 items)",
        "mean generated tokens per question", "accuracy (%)")
  ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK2)
  fig.tight_layout()
  fig.savefig(OUT / "v2-accuracy-vs-tokens.png", facecolor=SURFACE)


def fig_seeds(data, ood):
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=160, sharey=False)
  fig.patch.set_facecolor(SURFACE)
  for ax, d, title in ((axes[0], data, "Test (799)"), (axes[1], ood, "Out-of-distribution (460)")):
    for x, fam, color, marker in ((0, "e2", ORANGE, "s"), (1, "e3b", BLUE, "o")):
      ys = [stats(ctl["auto"])[0] for name, ctl in d.items()
            if seed_of(name)[0] == fam and "auto" in ctl]
      jitter = np.linspace(-0.08, 0.08, len(ys)) if len(ys) > 1 else [0]
      ax.scatter(np.full(len(ys), x) + jitter, ys, s=60, color=color, marker=marker,
                 edgecolor=SURFACE, linewidth=2, zorder=3)
      if ys:
        ax.hlines(np.mean(ys), x - 0.2, x + 0.2, color=color, lw=2)
    ax.set_xticks([0, 1], ["outcome-only", "+ suggested-\nvs-actual"])
    ax.set_xlim(-0.5, 1.5)
    style(ax, title, "", "accuracy (%)" if ax is axes[0] else "")
  fig.suptitle("Every RL seed (dot) and the mean (bar): the regret reward makes routing stable",
               x=0.02, ha="left", fontsize=11, color=INK)
  fig.tight_layout()
  fig.savefig(OUT / "v2-seed-stability.png", facecolor=SURFACE)


def fig_price_sweep(data):
  ref, cot = data["e3bs1"]["auto"], data["e3bs1"]["cot"]
  ids = sorted(set(ref) & set(cot))
  lams = np.linspace(0, 0.004, 17)
  mean, lo, hi = [], [], []
  for lam in lams:
    m, a, b = report_v2.boot_diff(report_v2.utility(ref, ids, lam)[2],
                                  report_v2.utility(cot, ids, lam)[2], n=2000)
    mean.append(m), lo.append(a), hi.append(b)
  fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
  fig.patch.set_facecolor(SURFACE)
  x = lams * 100  # accuracy points charged per 100 tokens... expressed per token below
  ax.fill_between(lams * 1000, lo, hi, color=BLUE, alpha=0.15, lw=0)
  ax.plot(lams * 1000, mean, color=BLUE, lw=2)
  ax.axhline(0, color=INK2, lw=1)
  ax.axvline(0.5, color=GRAY, lw=1, ls="--")
  ax.text(0.55, max(hi) * 0.92, "trained at\nλ = 0.0005", color=INK2, fontsize=8.5)
  style(ax, "Learned routing minus always reasoning, as tokens get more expensive",
        "token price λ (×10⁻³ accuracy per token)", "utility difference (95% CI)")
  fig.tight_layout()
  fig.savefig(OUT / "v2-token-price-sweep.png", facecolor=SURFACE)
  del x


def fig_genres(data, items):
  genres = ["compare", "knowledge", "decision", "rubric", "arith", "wordproblem"]

  def cot_share(rows):
    out = []
    for g in genres:
      ids = [i for i in rows if items[i]["genre"] == g]
      out.append(100 * sum(rows[i][2] == "cot" for i in ids) / max(1, len(ids)))
    return out

  fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
  fig.patch.set_facecolor(SURFACE)
  x = np.arange(len(genres))
  w = 0.36
  for off, rows, color, label in ((-w / 2 - 0.01, data["sft"]["auto"], GRAY, "SFT only"),
                                  (w / 2 + 0.01, data["e3bs1"]["auto"], BLUE, "after RL")):
    ax.bar(x + off, cot_share(rows), width=w, color=color, label=label, zorder=3)
  ax.set_xticks(x, genres)
  style(ax, "How often the model chooses to reason, by question type (test)", "",
        "% of questions answered with CoT")
  ax.legend(frameon=False, labelcolor=INK2)
  fig.tight_layout()
  fig.savefig(OUT / "v2-reasoning-by-genre.png", facecolor=SURFACE)


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  test, items = load_all("test")
  ood, _ = load_all("ood")
  fig_accuracy_tokens(test)
  fig_seeds(test, ood)
  fig_price_sweep(test)
  fig_genres(test, items)
  print("wrote", sorted(p.name for p in OUT.glob("v2-*.png")))


if __name__ == "__main__":
  main()
