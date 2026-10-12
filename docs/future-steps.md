# Possible future steps (after v2)

Written 2026-10-12, after v2 finished.

**Where v2 stands:**
- **Accuracy and cost:** on the sealed test, learned routing matches always-CoT accuracy with ~30% fewer tokens and is ~20% faster at batch 1.
- **Stability:** the suggested-vs-actual regret reward gives 88.6% ± 0.3 over 5 seeds, against 85.7% ± 3.2 for outcome-only RL (p = 0.008).
- **Spend:** about $41 of the $50 budget (estimated from run times), so about $9 is left.
- **Write-ups:** [blog-v2.md](blog-v2.md) and [paper-v2-draft.md](paper-v2-draft.md).

The order is: publish, get free compute, then do cheap experiments that strengthen the story. v3 waits until compute isn't the bottleneck.

## 1. Ship what we have (this week, free)

- [ ] **Publish the blog post** ([blog-v2.md](blog-v2.md) and `docs/figures/v2-*.png`): a personal site, Medium or the Hugging Face blog, linking the repo.
- [ ] **Merge `kwt-text-modes` into `main`** (a PR), so the repo front page shows v2.
- [ ] **Release on Hugging Face:**
  - [ ] the measured labels: per-question best mode, per-mode accuracy and tokens, and the gold audit. For sources with unclear licences (bev-decision, OpenBookQA), ship IDs and labels, not raw text; see [data-v2/SOURCES.md](data-v2/SOURCES.md);
  - [ ] the mode-value probe (`runs/kwt/mode_value/mode_value.pkl`);
  - [ ] the best LoRA, E3b seed 3 (89.0% on test; `gs://.../kwt/rl-e3b-s3/out/lora.npz`), with a model card.
- [ ] **Post a short thread** with two charts: accuracy vs tokens, and seed stability.

## 2. Get free compute ($0)

- [ ] **Apply to Google's TPU Research Cloud (TRC)** for free TPU quota. Attach the published blog and repo. This could fund v3.

## 3. Cheap experiments that strengthen the paper (~$5 of the remaining ~$9)

- [ ] **A faster runtime (~$1).**
  - The problem: even a one-step JEV answer takes 1.67 s in the current runtime. It always decodes a 16-token header, re-prefills the prompt on every mode switch, and pays a fixed setup cost per sampler call.
  - The fix: stop decoding as soon as `<mode:x>` appears, and reuse the KV cache for the next segment.
  - Expected: JEV at roughly 0.1–0.2 s, turning "20% faster" into a much larger wall-clock win.
- [ ] **Re-measure the labels with the trained model (~$4).**
  - The problem: the labels came from the base model, but SFT raised CoT accuracy from 80% to 89%, so they are partly stale.
  - The fix: re-profile the RL split with the SFT model, then retrain E3b.
  - If it helps, that's a clean extra finding: iterate measure → train → measure (STaR-style).
- [ ] *(Optional)* **Train at a higher token price** (λ = 0.001, ~$2). The price sweep suggests the routing win becomes significant there, but that was by re-scoring, not by training at that price.

## 4. v3 (once TRC or more budget arrives)

Pick one; I'd suggest 8B first, then omni:
- [ ] **Scale to 8B (Qwen3-8B).** Does measured-label routing plus the regret reward hold on a bigger model? This is the safer research claim. It needs a multi-chip slice for every step.
- [ ] **Omni model.**
  - The modes become things like "answer from the image or audio directly", "describe, then reason in text" and "pick an option (JEV)".
  - The recipe stays the same: measure each mode on the model → teach the tags as switches (SFT) → GRPO with the regret signal.
  - This is the more novel direction, but more engineering.

## 5. Optional: a workshop paper

- [ ] Turn [paper-v2-draft.md](paper-v2-draft.md) into a 4–6 page workshop paper on efficient reasoning or test-time compute (NeurIPS or ICLR workshops). It needs:
  - a related-work section: Thinkless, AdaptThink, ARM, AutoThink, "To CoT or not to CoT", and Stable Adaptive Thinking;
  - a figures pass;
  - a final limits section.

## Housekeeping

- [ ] **Human spot-check** of ~15 of the gold-audit decisions in [data-v2/audit_decisions.json](data-v2/audit_decisions.json).
- [ ] **Check actual GCP billing** against the ~$41 estimate.
- [ ] **Known small issue:** `kwt/data/build.py` has an awkward line wrap in `drop_long` (works, cosmetic).
