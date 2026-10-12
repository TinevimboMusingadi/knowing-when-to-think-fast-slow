# Knowing when to think, measured: capability-calibrated mode routing for a 4B model

*Draft, 2026-10-12. All numbers come from the sealed 799-item test split and the 460-item OOD split, each opened once.*

## Abstract

One model answers in one of three modes:
- **JEV:** a single forward step that picks an option letter, with no generated text;
- **direct:** a short answer;
- **CoT:** brief reasoning, optionally handing off to a JEV pick.

A mode tag at the start of the answer selects the mode, and the runtime executes it natively.

The main lesson from v1 was about labels. Routing labels assigned when the data is generated taught the model the wrong thing: the same task family appeared under every "behavior", and learned routing never beat always-CoT. Here, the target mode for every training item is **measured on the model itself**: we run Qwen3-4B in each mode and take the mode with the best accuracy minus token cost.

On the sealed test split, learned routing matches always-CoT accuracy (88.6% vs 88.1%) with **~30% fewer tokens**, and is +10.5 points over SFT routing. A reward term that penalises the *regret* of the chosen mode, against measured or probe-predicted per-mode utilities, makes RL **stable across seeds**: 88.6% ± 0.3 over 5 seeds, against 85.7% ± 3.2 for outcome-only RL. That's +2.9 points; one-sided exact permutation test p = 0.008.

On top of those labels:
- SFT teaches the tags as switches (forced-mode obedience 100%) and as a router.
- GRPO with a verifiable reward (plus a decaying *regret* term from a probe of the model's own hidden state) learns to route. On validation it matches always-CoT accuracy (89.3%) with **31% fewer tokens**.

## 1. Setup

- **Model:** Qwen3-4B with LoRA rank 16; Tunix (JAX); spot TPU v5e. Data work, SFT and eval run on one chip (v5litepod-1, ~$0.50/h); RL runs on v5litepod-4 (~$2/h). Total v2 compute is about **$25** (estimated from run times; ~16 single-chip hours and ~8.4 four-chip hours).
- **Modes and native runtime:** `<genre:g><mode:m>`, as in [kwt/format.py](../kwt/format.py) and [kwt/evaluate_v2.py](../kwt/evaluate_v2.py). A system-prompt line `Mode: auto|jev|direct|cot` either forces a mode or lets the model choose.

## 2. Data: measured labels ([datasheet](data-v2/DATASHEET.md))

- **Size and sources:** 12,998 single-turn items in six genres: bev-decision (the JEV-native decision dataset), GSM8K, AQuA, ARC, OpenBookQA, CommonsenseQA, plus procedural compare / arith / rubric tasks with difficulty knobs.
- **Cleaning:** schema checks, near-duplicate removal before splitting, and balancing.
- **Gold audit:** Qwen3-4B with thinking flagged 276 items. Hand review found ~0.4–1.7% gold errors in the maths and science sources and ~10% ambiguity in CommonsenseQA. 2 test golds were fixed and 4 items dropped.
- **Capability profile:** every item in every feasible mode (K=4). U_m = p_m − λ·t_m with λ = 5e-4, and suggested mode = argmax U_m.
- **Label gates:**
  - stability 89%;
  - cross-fitted headroom over the best genre-only router: 0.064 utility.
  - The "genre gives the label away" check fails for 3 of 6 genres (they are JEV-dominated), which is itself a finding.
- **Capability-relative labels:** 4B handles number comparisons by JEV at every digit count, where the v1 1.7B model needed reasoning. Rubric depth moves the label from JEV (depth 1) to CoT (depths 2–3).

## 3. Does the model already know when it needs to think?

A linear probe on the final hidden state at the end of the prompt, **before generating anything**, predicts per-mode success on held-out val:

| mode | AUROC |
|---|---|
| JEV | 0.90 |
| direct | 0.94 |
| CoT | 0.91 |

Calibration error is 0.04–0.08. As a router, though, the probe's argmax only ties the genre-lookup router (0.818 vs 0.819). Near-ties between modes flip on small errors. We use it as a *shaping* signal, not a decision rule.

## 4. Training

**SFT** (1 epoch, 14,168 examples, ~70 min). Half the examples are *obedience*: every item rendered in each mode under a forced control. Half are *routing*: the measured best mode under `Mode: auto`. CoT bodies are self-distilled: the shortest of 4 samples that is correct and whose arithmetic all checks out. 8.2% of the base model's samples with checkable arithmetic contain a false equation, and 79% of those still reach the right answer.

**GRPO** (150 steps, 4 prompts × 8 samples). The reward ([kwt/rewards_v2.py](../kwt/rewards_v2.py)):

| term | rule |
|---|---|
| correctness | exact match |
| format | +0.1 valid / −0.2 invalid; −1 if a forced mode is not obeyed |
| token cost | only on correct answers |
| faithful | −0.05 per false equation |
| genre | small bonus for the right genre tag |
| route_regret | −β·(U* − U_chosen) from the mode-value probe (trained on the SFT split only); β fades to 0 over the first 60% of training |

30% of RL prompts force a mode, so obedience cannot decay.

Exploit tests ([tests/test_rewards_v2.py](../tests/test_rewards_v2.py)):
- short and wrong never beats long and right;
- a false equation followed by a lucky pick scores below a clean trace;
- disobeying a forced mode is always the worst outcome.

## 5. Results

Sealed test split (799 items: 134 per genre, minus 5 dropped by the audit) and OOD split (460 items: the unseen bev domain "Game-state decisions" plus 6-digit arithmetic). 95% intervals are paired bootstraps over items ([kwt/report_v2.py](../kwt/report_v2.py)). Every model is scored under all four controls; the "auto" rows are learned routing.

| model (auto) | seed | test acc | tokens | utility | OOD acc | JEV / direct / CoT |
|---|---|---|---|---|---|---|
| SFT only | – | 78.1% | 40.1 | 0.761 | 69.3% | 71 / 4 / 24 |
| GRPO outcome-only (E2) | 0 | 87.0% | 49.7 | 0.845 | 78.0% | 57 / 4 / 39 |
| GRPO outcome-only (E2) | 1 | **80.4%** | 40.5 | 0.783 | 76.1% | 68 / 4 / 28 |
| GRPO + faithful + genre (E6\*) | 0 | 87.4% | 49.9 | 0.849 | 77.6% | 57 / 4 / 39 |
| **GRPO + route_regret, probe utilities (E3b)** | 0 | **88.6%** | 54.7 | **0.859** | **79.3%** | 54 / 0 / 46 |
| **GRPO + route_regret, probe utilities (E3b)** | 1 | **88.5%** | 53.4 | **0.858** | **78.9%** | 54 / 0 / 46 |
| GRPO + route_regret, measured profile (E3b-fixed) | 0 | 88.2% | 53.0 | 0.856 | 78.9% | 55 / 2 / 42 |
| GRPO + route_regret + faithful + genre (E3) | 0 | 87.9% | 54.5 | 0.851 | 79.1% | 54 / 0 / 46 |
| *always-CoT, E3b weights* | 0 | 89.0% | 79.1 | 0.850 | 79.1% | forced |
| *always-CoT, SFT weights* | – | 87.7% | 88.7 | 0.833 | 78.0% | forced |

\*E6 was meant to test the fixed profile. A bug turned its regret into a constant that GRPO normalises away, so it ran as "faithful + genre, no regret". E3b-fixed is the corrected fixed-profile run.

**Paired comparisons (test):**

| comparison | accuracy | utility |
|---|---|---|
| E3b s1 − E2 s0 | +1.5 [+0.1, +2.9] | +0.013 [−0.001, +0.027] |
| E3b s1 − E2 s1 | +8.1 [+5.9, +10.4] | +0.075 [+0.053, +0.096] |
| E3b s0 − E2 s0 | +1.6 [+0.6, +2.8] | +0.014 [+0.004, +0.025] |
| E3b s1 − E3b s0 (seed stability) | −0.1 [−1.3, +0.9] | −0.001 [−0.012, +0.010] |
| E3b-fixed − E3b s0 (profile vs probe) | −0.4 [−1.3, +0.4] | −0.003 [−0.012, +0.005] |
| E3b s0 − E3 (adding faithful + genre) | +0.8 [0.0, +1.5] | +0.007 [0.000, +0.015] |
| E3b s1 learned − E3b s1 always-CoT | −0.1 [−1.6, +1.4] | +0.011 [−0.004, +0.026] |

The last row is the same accuracy at ~31% fewer tokens.

**Findings:**
1. **Learned routing matches always-CoT accuracy at ~30% fewer tokens.** This holds on test (88.5% vs 88.6%, 53 vs 77 tokens) and OOD. v1's learned routing stayed 11 points below always-CoT. At λ = 5e-4 the utility gain is small (+0.011) and not significant; the token saving is the robust part.
2. **The route-regret reward makes routing RL stable** ([5-seed table](results-v2-seeds.md)).
   - **Test:** 88.6% (sd 0.27, range 88.2–89.0) with the regret term, against 85.7% (sd 3.25, range 80.4–88.4) without. That's +2.9 points, p = 0.008, with 12× less spread.
   - **Outcome-only seeds trade accuracy for tokens along a line.** Each run under-thinks to a different degree.
   - **OOD:** +1.0 point, p = 0.071. Same direction, not significant.
3. **A hidden-state probe can replace exhaustive profiling for the regret signal.** Probe utilities and measured-profile utilities give the same result (−0.4 [−1.3, +0.4]).
4. **The faithful and genre rewards do not help.** The true-equation rate is ~96–97% either way, and adding them costs ~0.8 points.
5. **What RL learned** (SFT → E3b): reason on multi-step rubric rules (55% → 100%) and arithmetic (83% → 94%), and stay on one-step JEV for comparisons, knowledge MCQs and most decision items. Forced-mode obedience is 100% for every model.

## 6. Honest limits (to complete)

- **Seeds:** 5 each for E2 and E3b; 1 seed for the other ablations.
- **Latency is runtime-bound.** At batch 1, auto takes 1.74 s mean against 2.19 s for always-CoT (−20%; median −30%). The JEV path still costs 1.67 s in this runtime, which re-prefills and decodes a 16-token header.
- **The labels are base-model labels.** They are partly stale for the SFT model (CoT accuracy 80% → 89%).
- **The labels are base-model labels.** SFT raised CoT accuracy from 80% to 89%, so they partly describe a different model.
- **Bev outcome and annotator tasks** (headline click-through, wine quality and the like) have an accuracy ceiling no mode can lift.
- **Token counts are not latency.** Wall-clock is not yet measured.
