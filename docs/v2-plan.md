# kwt v2 plan: capability-calibrated mode tags, obedience SFT, regret-shaped RL

Status: proposal, 2026-10-10. Nothing has been run yet. It builds on the v1 results in [article.md](article.md) and keeps the v1 stack: Qwen3-1.7B, LoRA, stock Tunix GRPO and one spot v5e-1 chip.

## 1. What v1 showed

| | worked | did not work |
|---|---|---|
| Format | Text mode tags, with JEV as one constrained letter token. Stock GRPO trains it. | `<mode:cot>` almost always ends in a JEV hand-off, so forced CoT broke on questions with no options (61% → 38%). |
| Warm-start | 256 SFT steps took accuracy from 9.8% to 50.5%, with 100% valid traces. | One short epoch on human GSM8K solutions whose style differs from the model's own reasoning. |
| RL | lr 3e-5 cut tokens by 20% at the same accuracy. Fast decisions rose from 58% to 82%. | RL from the base model collapsed to always-direct. lr 3e-6 did nothing. Reasoning accuracy fell (50% → 37%). |
| Reward | Exact-match correctness can't be gamed. The cost cap stopped cost from dominating. | The cost was charged on wrong answers too, so short-and-wrong always beat long-and-wrong. 0.002 per token was too steep for arithmetic. |
| Faithfulness | — | Correct JEV picks came after false arithmetic (`(7212*7213)+59=79981`). The reward can't see this. |
| Routing | The model picks modes by task type. | Learned routing (50%) is below always-CoT (60.8%). |

**The main dataset flaw** (found while writing this plan): v1 behavior labels were assigned **by construction, not by what the model can actually do**. The same family (`multiply_add`, `addition`, `half_sum`, `inventory`) appears under all four behaviors (see `data/recovery-v2/train.jsonl`). So "direct" contains items like *Split 10014 and 92 items equally*, which a 1.7B model cannot do without reasoning (direct accuracy was 24%). SFT therefore taught the model to route on wording cues, and RL had no clean signal to correct that. **v2 derives the target mode from measured per-mode success.**

## 2. What the field reports (2025–2026)

- **Mode collapse is the default failure.** The mode choice is one discrete token, and whichever mode gets a small edge takes over. This is what happened in our run 1.
- **[Thinkless](https://arxiv.org/abs/2505.13379), DeGRPO.** The control token's gradient is normalised by response length, which biases updates toward the short mode. The fix is to weight the control-token loss separately from the response loss.
- **[AdaptThink](https://arxiv.org/abs/2505.13417).** A constrained objective prefers no-thinking when accuracy doesn't drop. Importance sampling keeps both modes in every batch.
- **[ARM / Ada-GRPO](https://arxiv.org/abs/2505.20258).** A format-diversity scale on the reward, decayed over training, prevents format collapse.
- **[Stable Adaptive Thinking](https://arxiv.org/abs/2602.22556) (Feb 2026).** A hybrid SFT warm-up on balanced data, where each problem has *both* a thinking and a no-thinking response, plus advantage shaping and length-aware gradient control.
- **[Learning When to Think, Aug 2026](https://arxiv.org/abs/2608.20256).** A three-way first-token choice (NoThink, Short, Long) inside GRPO, with a shaped per-mode reward and hard per-mode token caps. This is closest to kwt, and it is a preprint.
- **Not covered by this work, and the gap kwt can fill:**
  1. a native decision mode (JEV) that skips generation;
  2. routing labels calibrated to *the model's own* measured ability;
  3. a verifiable faithfulness check on the scratch work.

## 3. v2 design

### 3.1 Tags: genre, suggested and actual

Each training row carries three tags:

| tag | who produces it | where it lives | use |
|---|---|---|---|
| **genre** `<genre:arith>` | the model emits it first; gold comes from the generator | in the completion | a cheap "look at the problem first" step; verifiable; small reward |
| **suggested** mode | the **capability profile** (§3.3), not a human | dataset only, never shown to the model | SFT target and RL regret baseline |
| **actual** (sampled) mode | whatever tag the policy samples in that rollout | in the completion | compared with suggested through *regret*, not exact match |

Completion format (auto mode):

```
<genre:wordproblem><mode:cot>
8+6+21=35
<answer>35</answer>
```

Forced mode: the system prompt contains `Mode: cot` (or `direct` / `jev`). The model must start with that tag. This is the switch that turns reasoning on and off. Auto is `Mode: auto`.

### 3.2 Dataset v2 (`data/kwt-v2/`)

Generators extend `kwt/tasks.py`. **Every genre gets a difficulty knob**, so the suggested mode changes *within* a genre. Without that, a genre lookup table would match the learned router.

| genre | source | difficulty knob | options shown |
|---|---|---|---|
| `compare` | procedural | digit count, gap size (5975 vs 5974 is hard; 1807 vs 11 is easy) | yes |
| `rubric` | procedural formula scoring | nesting depth | yes |
| `arith` | procedural add, mul-add, half-sum, inventory | digits (1–6) × steps (1–4) | 50/50 |
| `wordproblem` | GSM8K train (MIT) | number of solution steps | 50/50 |
| `fact` | short factual/commonsense MCQ, e.g. ARC-Easy (check licence) | none; tests "don't compute when there is nothing to compute" | yes |

Sizes:
- a ~6k pool, profiled;
- ~3k for SFT and ~2k for RL, disjoint;
- 300 for validation;
- **600 for a new sealed test**, plus the old 400-question test for comparison with v1;
- one held-out genre or difficulty band (for example 6-digit arithmetic, or GSM-hard) for out-of-distribution routing.

### 3.3 Capability profile, the core novelty

1. Train the **obedience SFT** checkpoint (§3.4a).
2. For every pool item, force each mode, take K=4 samples at T=0.7 (one JEV forward pass gives an exact probability), and record:
   - per-mode accuracy p_m;
   - mean tokens t_m.
3. Compute utility U_m = p_m − λ·t_m, with λ chosen so that 100 CoT tokens are worth about 5 accuracy points.
   - Suggested mode: argmax U_m.
   - The full `(p_m, t_m, U_m)` vector is stored, so labels are soft.
   - Near-ties are marked *ambiguous* and excluded from routing-accuracy metrics.
4. Sanity output: a table of suggested-mode share against genre × difficulty. Expected: direct for 2-digit sums, cot for 5-digit mul-add, JEV for clear-gap comparisons.

Cost: 6k × 3 modes × 4 samples, mostly short. Estimated at 1–2 v5e-1 hours (about $1), to be measured on a 200-item smoke first.

### 3.4 SFT v2 (two parts, one run)

**a. Obedience (about 50% of the mix).** Every item is rendered in **every feasible mode**, each with a `Mode: X` prompt (the "Stable Adaptive Thinking" balanced-pair idea). This teaches that the tag controls reasoning, and it fixes v1's broken forced CoT: CoT with no options must end in `<answer>`.

CoT traces are **self-distilled**: sample the model's own short reasoning, keep only correct traces whose equations check (§3.5), and prefer the shortest. Procedural genres can also use programmatically generated step traces. Human GSM8K solutions are only a fallback.

**b. Routing (about 50%).** With `Mode: auto`, the target is `<genre:g><mode:suggested>` plus that mode's trace.
- **Loss weighting:** upweight the genre and mode tag tokens by about 5×. One token among ~40 otherwise gets little gradient (the Thinkless argument, applied at SFT).
- **Check:** the v1 SFT loss in `kwt/sft.py` is a plain token mean, so per-token weights are a small change there.

Budget: about 1,000 steps at batch 4 and lr 1e-4, roughly 15 minutes on v5e-1.

### 3.5 RL v2 reward (`kwt/rewards_v2.py`)

Every term is a public function, so Tunix sums them as before.

| term | formula | v1 → v2 change |
|---|---|---|
| `correctness` | 1 if exact match | unchanged |
| `well_formed` | +0.1 valid / −0.2 invalid; **forced mode not obeyed → −1** | adds obedience |
| `compute_cost` | **only if correct:** −α·tokens, α=5e-4, cap 0.3 | 4× cheaper; no longer rewards short wrong answers |
| `route_regret` | −β·(U* − U_actual) from the profile; β=0.5, annealed to 0 by 60% of steps | the suggested-vs-actual term. No penalty when modes tie; large only for clearly wrong routing. Annealing hands control back to real outcomes. |
| `genre` | +0.05 if the genre tag is correct | new, ablated |
| `faithful` | −0.05 per false `expr=value` line in CoT (safe `ast` eval), cap −0.2 | new; targets the `7212*7213` problem |

Why regret rather than an exact match on the suggested tag: a hard match would make RL imitate the profile, including its noise. Regret is zero when the actual mode is as good as the suggested one, so RL can still discover cheaper routes the profile missed.

Training controls:
- The prompt mix is about 70% `Mode: auto` and 30% forced, so obedience can't decay.
- **Collapse guard:** log first-mode share per genre at each step. If any mode drops below 10% in auto prompts, apply ARM-style diversity scaling.
- **Dynamic sampling:** skip groups whose 8 rewards are all equal, because they carry no gradient. Check whether Tunix exposes this. If not, pre-filter prompts whose profile is solved by every mode or by none.
- lr 3e-5 (v1 run 3), 300 steps, 8 generations, β_KL 0.02.

### 3.6 Optional Track B: decoupled mode-token loss

This is DeGRPO-style: a separate weight on the genre and mode token positions in the GRPO loss. It needs a patched Tunix loss. Attempt it only if the collapse guard fires in Track A, or if budget remains.

## 4. Experiments and evaluation

Baselines are evaluated on the new sealed test:
- always-JEV, always-direct and always-CoT;
- **oracle router**, which uses each test item's profiled best mode (the upper bound);
- **genre-lookup router**, which uses the majority suggested mode per genre (the "did it learn difficulty, or just genre?" baseline).

| run | what | question |
|---|---|---|
| E1 | SFT v2 only | Do calibrated labels beat v1 SFT (50.5%)? |
| E2 | RL: correctness + format + cost (new α) | the v1 reward fixed but outcome-only |
| E3 | E2 + `route_regret` | does the suggested-vs-actual signal help? |
| E4 | E3 + `faithful` | do equations become true without losing accuracy? |
| E5 | E3 + `genre` | does an explicit genre step help routing? |

E2 and E3 use 2 seeds each. The rest use 1 seed.

Metrics:
- accuracy and mean tokens, plotted as a Pareto curve against the baselines;
- **utility** at the profile's λ;
- **routing accuracy** against the oracle, on non-ambiguous items and *within genre*;
- regret;
- forced-mode obedience;
- true-equation rate;
- first-mode entropy over training;
- the OOD split.

**Success means:**
- learned routing has higher utility than always-CoT, and beats genre-lookup on within-genre routing;
- obedience is at least 98%;
- the true-equation rate rises.

Report the result honestly if it doesn't happen.

## 5. Budget (spot v5e-1, about $0.50 per hour; $46 of $50 left)

| step | VM-hours | ≈ $ |
|---|---|---|
| obedience SFT and capability profiling | 2.5 | 1.25 |
| routing SFT and E1 evaluation | 1 | 0.50 |
| E2–E5, 7 RL runs of about 1 h each with evaluation | 8 | 4.00 |
| failures and preemption slack (×1.5) | 5 | 2.50 |
| **total** | **about 16.5** | **about $8** |

Every VM still self-deletes ([scripts/vm_startup.sh](../scripts/vm_startup.sh)). Run the CPU smoke test before any launch, and ask before creating a VM.

## 6. Build order (all free and local before any TPU time)

1. `kwt/tasks_v2.py`: genre generators with difficulty knobs and the new splits. Unit test: the suggested label is *not* constant within a genre after profiling (checked on CPU with Qwen3-0.6B).
2. `kwt/format.py`: the genre tag, `Mode:` forcing, and the "CoT with no options ends in `<answer>`" rule.
3. `kwt/profile.py`: forced-mode sampling and the utility labels.
4. `kwt/sft.py`: the obedience and routing mix, and tag-token loss weights.
5. `kwt/rewards_v2.py` and `tests/test_rewards_v2.py`: regret, the equation checker and the obedience term.
6. `kwt/evaluate.py`: oracle and genre-lookup routers, routing accuracy, obedience and the equation metric.
7. Smoke test, then the TPU steps in the order of §5.

## 7. Open questions

- Does Tunix GRPO allow per-token loss weights, and can it filter zero-variance groups? This decides Track B and dynamic sampling.
- λ sets what counts as the right mode. Sweep λ in post-hoc analysis (relabeling is free once the profile is stored), and report the Pareto curve rather than a single number.
- Should profiling use the post-obedience SFT model (planned) or re-profile after RL? Re-profiling is a natural v3 iteration, in the spirit of STaR.
