# kwt v2 dataset: datasheet (draft)

Status: built, profiled (Qwen3-4B, `profile-v2`; every item in every feasible mode) and audited on 2026-10-11. 12,998 items.

## Purpose

Teach one model **when** to use each answer mode:
- **JEV**: a one-step option pick, with no generation;
- **direct**: a short answer;
- **CoT**: brief reasoning.

The target mode for each item is **measured on the model** (§4), not assigned when the data is generated. Assigned labels were the main data flaw in v1. There, the same task family appeared under every behavior label, so "direct" included problems the 1.7B model can't solve without reasoning (24% accuracy).

## 1. Composition

13,076 single-turn items with exact gold answers. 10,244 show lettered options (2–6 of them), so JEV is available on those; 2,832 are open-ended.

| source | sft | rl | val | test | ood | total |
|---|---|---|---|---|---|---|
| bev-decision | 2372 | 1437 | 50 | 134 | 300 | 4293 |
| gsm8k | 1128 | 707 | 45 | 120 | 0 | 2000 |
| aqua_rat | 618 | 358 | 5 | 14 | 0 | 995 |
| arc | 466 | 242 | 25 | 67 | 0 | 800 |
| commonsenseqa | 464 | 290 | 13 | 33 | 0 | 800 |
| openbookqa | 489 | 262 | 12 | 34 | 0 | 797 |
| proc-arith | 873 | 496 | 50 | 134 | 0 | 1553 |
| proc-arith-ood (6-digit) | 0 | 0 | 0 | 0 | 160 | 160 |
| proc-compare | 616 | 398 | 50 | 134 | 0 | 1198 |
| proc-rubric | 169 | 127 | 50 | 134 | 0 | 480 |
| **total** | 7195 | 4317 | 300 | 804 | 460 | 13076 |

Genres: `decision` (bev), `wordproblem` (GSM8K, AQuA), `knowledge` (ARC, OBQA, CSQA), and the procedural `arith`, `compare` and `rubric`.

- **Val and test** hold 50 and 134 items **per genre**. Inside each genre, items are drawn round-robin over difficulty strata.
- **OOD** never appears in training. It has two parts:
  - the whole bev domain *Game-state decisions* (300 items);
  - 6-digit arithmetic (160 items); training uses 1–5 digits.
- **SFT and RL pools are disjoint**, split 5/8 and 3/8.
- **Known skew:** AQuA has no difficulty feature, so it is one stratum and contributes few test items (14 of 134 word problems).

Sources and licences are in [SOURCES.md](SOURCES.md).

## 2. Cleaning (automatic)

| check | effect |
|---|---|
| **Schema** ([kwt/data/schema.py](../../kwt/data/schema.py)): gold appears exactly once among the options, no duplicate or empty options, ≤ 6 options, numeric gold parses | AQuA: 2,283 of 97,467 rejected (2.3%). CommonsenseQA: 164 of 9,741 (1.7%). ARC: 4. OpenBookQA: 3. |
| **Near-duplicates** ([kwt/data/dedup.py](../../kwt/data/dedup.py)): MinHash on word 5-grams, Jaccard ≥ 0.8 with the same gold. Run **before** splitting, so no duplicate crosses a split. | Raw-source rates on a 20k AQuA sample plus all other sources: AQuA 7.2%, OBQA 1.2%, bev 0.4%, others ≈0. In the sampled pool, 64 were dropped. |
| **Balance** of the procedural generators | Found and fixed two bugs. Compare options were unshuffled, with yes/no gold skewed toward yes. Rubric gold piled up at the clipped scores 0 and 5. Both are now uniform by construction. Gold positions are shuffled for every source except ordinal score scales. |
| **Length** | bev states over 1,200 characters are dropped, which keeps prompts under ~400 tokens. |

## 3. Gold-label audit (model-assisted)

**Who was audited:** 2,161 items, answered by Qwen3-4B with native thinking (768-token budget, then a forced answer). That covers every val, test and OOD item, plus 100 training items from each human-labelled source.

Thinking accuracy by source:

| source | accuracy | source | accuracy |
|---|---|---|---|
| ARC | 97% | proc-compare | 100% |
| GSM8K | 91% | proc-rubric | 97% |
| OpenBookQA | 84% | proc-arith | 85% |
| AQuA | 77% | proc-arith-ood (6 digits) | 54% |
| CommonsenseQA | 75% | bev-decision | 73% |

67% of thinking runs finished within the budget.

**Review:**
- The queue was items answered against gold at confidence ≥ 0.9, plus every open-ended disagreement: **276 items**.
- **Procedural items** are correct by construction, so their flags are model errors.
- For **bev outcome and annotator labels**, gold is a recorded outcome or rating that the text can't settle. Those labels stand as the source record. This covers headline click-through, wine quality, purchases, toxicity, emotion and citation intent.
- **Bev items whose gold can be checked from the text** (logic, spatial, invoice policy) were re-checked by hand. All 28 eval-split items read in detail had a correct gold; the model was wrong.
- **All 87 flagged non-bev items** were re-solved by hand.

| source | flagged / audited | gold wrong | ambiguous | estimated error rate |
|---|---|---|---|---|
| AQuA | 10 / 119 | 2 | 0 | ~1.7% |
| GSM8K | 24 / 265 | 1 | 0 | ~0.4% |
| ARC | 6 / 192 | 1 | 1 | ~0.5% |
| OpenBookQA | 15 / 146 | 0 | 1 | ~0% |
| CommonsenseQA | 32 / 146 | 1 | 14 | ~0.7% wrong, **~10% ambiguous** |

These rates are lower bounds: unflagged items are assumed correct.

**Actions** ([audit_decisions.json](audit_decisions.json), applied by `kwt.data.build.apply_audit`):
- 2 test golds fixed: AQuA ore 263.66 → 266.67, GSM8K highlighters 115 → 95.
- 4 items dropped:
  - an AQuA item with no correct option;
  - an ARC item whose gold names the controlled variable;
  - 2 ambiguous CommonsenseQA test items.
- The profile answers of the fixed items were re-graded.
- **12,998 items remain.**
- A human spot-check of the review decisions is pending.

## 4. Capability profile and suggested-mode labels

Base Qwen3-4B, bf16, measured in each feasible mode by [kwt/data/profile.py](../../kwt/data/profile.py):
- **JEV:** one forward step, softmax over the shown letters.
- **direct:** "final answer only", 32-token cap.
- **CoT:** "a few short lines, then `<answer>`", 256-token cap.
- direct and CoT use K=4 samples at T=0.7. The example answer letter in the prompt rotates per item, so it can't anchor answers to one position.

Labels ([kwt/data/labels.py](../../kwt/data/labels.py)):
- U_m = p_m − λ·t_m, with λ = 5e-4 (100 CoT tokens cost 5 accuracy points). Suggested mode = argmax U_m.
- Items whose top two utilities are within 0.05 are *ambiguous* and excluded from routing-accuracy metrics.
- The full p, t and U vectors are stored, so λ can be re-swept without re-profiling.

### Results (all 13,002 items, λ = 5e-4)

| mode | mean accuracy | mean tokens |
|---|---|---|
| JEV (10,218 option items) | 65.6% | 1 |
| direct | 54.9% | 6.8 |
| CoT (K=4) | 80.5% | 85.7 |

Suggested-mode share by genre:

| genre | JEV | direct | CoT |
|---|---|---|---|
| decision | 83% | 4% | 13% |
| knowledge | 90% | 2% | 8% |
| compare | 94% | 1% | 5% |
| rubric | 51% | 4% | 45% |
| arith | 16% | 25% | 60% |
| wordproblem | 18% | 12% | 70% |

**The difficulty settings move the label within a genre:**
- **Rubric:** depth 1 goes to JEV 95% of the time; depths 2–3 go mostly to CoT.
- **Word problems:** the share of direct answers falls as steps rise, from 21% at 2 steps to 12% at 6.
- **Comparisons:** 4B handles them by JEV at every digit count. This is unlike the 1.7B model in v1, so the labels are relative to the model's capability, not the task.

Mean utility of each router (all items):

| router | utility |
|---|---|
| always-JEV / always-direct | 0.547 / 0.545 |
| always-CoT | 0.762 |
| genre-lookup (best mode per genre) | 0.769 |
| oracle, cross-fitted (pick on 2 samples, score on the other 2) | 0.833 |
| oracle, in-sample (inflated by the winner's curse) | 0.845 |

### Label-quality gates

| gate | target | result |
|---|---|---|
| stability: labels from even vs odd samples agree | ≥ 85% of non-ambiguous items | **89.1%**: pass |
| within-genre variance: at least 2 modes with ≥ 15% share | every genre | arith, wordproblem, rubric pass. Decision, knowledge and compare are JEV-dominated (83–94%): for those genres the genre alone predicts the label. |
| headroom: cross-fitted oracle minus genre-lookup utility | ≥ 0.05 | **0.064**: pass. Across a λ sweep: 0.049 (λ=2e-4), 0.058 (5e-4, val/test/rl subset), 0.066 (1e-3), 0.082 (2e-3). |

- **Ambiguity:** 63% of items are *ambiguous*, meaning their top two utilities are within 0.05. Most are JEV-vs-direct ties on easy option items, where the choice barely matters.
- **The baseline matters:** a genre-lookup router that takes each genre's *majority label* is weaker than one that takes the best *mean utility* (rubric: JEV wins the label vote 152–146, but always-CoT has twice the utility). We report the stronger baseline.

### Mode-value probe (`kwt/mode_value.py`)

A logistic probe on Qwen3-4B's final hidden state at the end of the prompt, **before any token is generated**. It is trained on the SFT and RL splits and evaluated on val:

| mode | AUROC | ECE |
|---|---|---|
| JEV | 0.90 | 0.05 |
| direct | 0.94 | 0.08 |
| CoT | 0.91 | 0.04 |

The model's own state largely encodes which modes will succeed. As a *router* (argmax of predicted utilities), though, it only ties genre-lookup on val (0.818 vs 0.819, against oracle 0.870). When modes are close, small errors flip the choice. RL-split utilities come from a probe trained on the SFT split only, so no RL item's own labels leak into its reward. That probe agrees with the measured best mode on 78% of non-ambiguous RL items.

### CoT traces for SFT (`kwt/data/prep.py`)

28,604 samples: 4 per SFT item, T = 0.7, 256-token cap.

| measure | result |
|---|---|
| correct | 80.6% |
| cut off at the cap | 3.4% |
| contain a false `expr = value` line (of samples with checkable arithmetic) | 8.2% |
| of those false-arithmetic samples, still reach the right answer | 79% |
| SFT items with a usable trace (correct, every equation true; shortest kept) | 85% |

## 5. Known limitations

- Labels are specific to **base Qwen3-4B** and to these prompts and caps. They will drift after SFT; a 500-item subset is re-profiled after SFT to measure that.
- bev-decision mixes upstream sources whose licences are unknown, and some upstream data is itself synthetic or model-generated.
- Public train splits may still overlap Qwen3's pretraining data. A contamination probe (original vs option-shuffled accuracy) is still to do.
