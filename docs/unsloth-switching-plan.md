# Decision training and learned switching: next steps

Recorded 8 October 2026. This is a development plan, not a new paid launch or a change to the approved model, head, backend, or budget.

## What is running and what worked

The project's last owned TPU, `kws-recovery-20261007-220801` in `us-west4-a`, is absent according to today's fresh resource lookup. The local validation job was stopped last night. No training was restarted during this investigation. Checkpoints remain preserved; the other project's Kaggle/Gemma work is outside this experiment.

Historical SFT completed and the best checkpoint at step 150 remains the recovery source. A retained example performed lookup → JEV → a grounded correct answer. One historical RL update changed adapters and mode rows, but established no improvement. The invalid checkpoint is excluded.

The recovery TPU pilot passed the small numerical checks and three disposable full-model supervised updates. Larger-batch compilation stalled; its cause is unresolved. A subsequent local full-model check was killed by an established memory exhaustion event. Recovery corrective SFT and sampled full-model GRPO remain incomplete. These failures concern execution and validation; they do not establish that a decision head cannot be trained.

The recorded conservative cumulative spending bound before resumption is $66.24 of $120, not a reconciled invoice. On 8 October the user explicitly approved moving $5 from contingency to the pilot. The active allocations are pilot $11, SFT $8, RL $20, evaluation $16, and contingency $2.89. The pilot has approximately $6.87 remaining; the cumulative ceiling and evaluation reserve remain unchanged. Stages and measured cost projections still apply.

## What Unsloth actually provides

The official recipe uses `from unsloth import FastDecisionModel, DecisionTrainer`, then `FastDecisionModel.from_pretrained`, `get_peft_model`, and `build_dataset`. Use this interface rather than expecting `FastLanguageModel` alone to create a decision head. Its labeled data contain state, questions, and gold answers. `DecisionTrainer` performs supervised decision training. [Official recipe](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)

For an ordinary LM, the current loader adds a Clef schema head and retains the underlying LM. This head has a different architecture from our 128-wide, two-layer candidate head. Its checkpoint is not an interchangeable recovery checkpoint. [Loader source](https://github.com/unslothai/unsloth/blob/main/unsloth/models/decision_from_lm.py)

The decision wrapper's forward path returns decision scores using backbone hidden states. A hybrid implementation would also need an explicit generation path through the underlying language model and shared trainable state. The existing supervised trainer does not supply our mixed-action GRPO update. This conclusion follows from its forward and supervised-loss implementation. [Decision implementation](https://github.com/unslothai/unsloth/blob/main/unsloth/models/decision.py)

The provided 4-bit recipe is a GPU route. Unsloth's documented training platforms do not establish TPU support for this recipe. It is not a drop-in replacement for Tunix. [Hardware requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)

Unsloth documents GRPO and points to ART for multi-turn agent training. Those provide useful generation/trajectory infrastructure; categorical custom-head training remains an integration requirement for our model. [Agent training guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/training-ai-agents-with-rl)

## Do we need a learned reward model?

For our verifiable tasks, use a deterministic answer/evidence verifier and a reward function. Training another neural reward model is not required. Unsloth's RL guide describes this approach. [RL and verifiable rewards](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)

Our existing environment already implements this reward:

- +1 for a correct answer with all required evidence acquired.
- +0.05 once for acquiring a required evidence field.
- −0.5 for answering without required evidence.
- −0.25 for invalid execution or exhausted limits.
- A compute penalty capped at 0.1, including generated tokens, candidate encoding, tool calls, and transitions.

There is no positive reward for using CoT, selecting JEV, or changing modes. Correctness dominates efficiency. On an easy task, an immediate correct answer can win. On a task requiring missing information, a supported answer after acquiring that information can win. Ordinary reasoning is judged through its outcome, not its length or resemblance to a private trace.

Today's [reward audit](reward-audit-20261008.json) passed 14 scripted counterexample checks. Eight existing environment/trajectory tests also passed. The audit checks correctness, unsupported guesses, irrelevant and repeated acquisition, available-context controls, invalid execution, mode neutrality, transition cost, candidate computation, bounded penalties, and oracle separation. It contains **no sampled model results**.

One shaping detail is explicit: acquiring just one of several required fields still earns +0.05, even if the eventual answer is wrong or unfinished. Full correctness still requires all fields. We preserve this historical behavior and record it; any future tightening must receive a new reward revision and be evaluated on the same counterexamples. The audit does not prove sufficient reward variation or training stability.

## What the earlier mode-switch repository already implements

A read-only inspection of the existing repository found token-based mode switching, a causal-LM LoRA SFT path, typed choice/noul/score schemas, a trajectory parser, deterministic rewards, transition statistics, and group-relative advantages. Eight existing RL-environment tests passed on 8 October. These are reusable foundations; a new learned reward model is not the missing component.

The inspected `src/rl/grpo_trainer.py` exposes `evaluate_group` and `train_step_mock`. The latter scores constructed completions and returns metrics; it does not generate policy samples, calculate differentiable policy likelihoods, or update weights. Likewise, `src/training/pipelined_trainer.py::_simulate_or_run_grpo` scores fixed examples and prints a completion message without an optimizer update. That message must not be treated as completed policy training. The local code does not establish what may have run elsewhere.

Its `JevEvaluator` scores candidate token IDs using the language model's final-position logits. It does not contain this project's isolated-candidate attention head. The current Tunix engine already implements the additional mixed-event probability path: generated tokens and real categorical head choices both enter its GRPO loss and gradient update. The remaining gap is stable full-model sampled execution and acceptance on the selected TPU, not inventing mode switching or a reward scorer. Private formats and teacher traces remain excluded from this public experiment.

## Recommended training sequence

Keep Qwen3-1.7B, the custom JEV head, the three learned mode tokens, and the complete language head for the main experiment. Finish the existing Tunix acceptance gates. Treat Unsloth decision SFT as a separate potential baseline, rather than silently replacing the model or head to make a tutorial run.

1. **Finish free compatibility checks.** Complete the pending full-model reference comparison with bounded memory, then regenerate strict revision-matched numerical/protocol reports. Preserve fixed tolerances and report failures. Do not relaunch the TPU while these gates are incomplete.
2. **Validate a complete mixed rollout/update before long training.** A supervised head update is insufficient. Sample modes and typed actions, collect observations, verify rewards, and replay exactly the sampled probabilities. Run save/reload continuation and confirm changes in LoRA, mode rows, and the head on diagnostic batches.
3. **Corrective SFT from step 150.** Train assistant language targets and typed decisions with equal task weighting. Include valid examples of JEV deferral, ordinary reasoning, returning to JEV, direct answers, and necessary information acquisition. Preserve family-separated paired context variants. Require the approved 95% valid protocol threshold before RL.
4. **Inspect real reward groups.** Sample four independent trajectories per prompt on a balanced validation collection. Log correctness, grounding, component rewards, actions, and variance. Require at least 20% of groups with varied rewards across the first two collections. Distinguish task-success variation from differences caused only by token cost. Inspect difficulty/verifier behavior if the signal is weak; do not manufacture reward variation.
5. **Real mixed-action GRPO.** Start fresh optimizer state from the selected corrected SFT policy; freeze that exact reference. Use the approved clipping, KL, learning rate, update limits, finite checks, and checkpoint schedule. Every sampled mode token is a language policy event. Every JEV choice is a categorical head event with its actual log probability. Prompt/tool/history tokens contribute no policy loss. An off-the-shelf text-only rollout trainer cannot substitute an invented word token for that choice.
6. **Held-out evaluation.** Compare the same frozen examples under direct, JEV, CoT, confidence routing, learned SFT routing, and valid RL routing if available. Count candidate computation as well as generated tokens. Record natural transitions, failures, grounding, accuracy, and synchronized warmed latency. Improvements remain an experimental question.

The next paid pilot requires a projection that includes restoration, compilation, sampled GRPO, validation, uploads, and deletion, within its stage allowance. The total cap remains $120 and the evaluation reserve remains protected. No on-demand fallback, larger hardware, 4B model change, or automatic numerical-failure retry is authorized by this document.

## If we choose an Unsloth experiment later

First establish a reproducible standalone decision baseline using the official loader/trainer, public labeled fixtures, a pinned release/commit, a separate environment, and verified GPU requirements. Measure head updates, candidate behavior, calibration, and save/reload. The guide's reported runtime and accuracy do not predict our custom experiment.

Before calling it a replacement switcher, implement and validate both generation and decision paths, mode-token input/output rows, shared LoRA state, an identical frozen reference, and a mixed likelihood objective. ART/GRPO can help manage multi-turn text trajectories; integration must expose the real categorical head actions to the learner. This is a separate engineering experiment until those checks pass. Do not reuse or interrupt the other agent's Kaggle session.

## Reproduce today's offline audit

From the repository root:

```powershell
python -m switching.audit_reward_v2 --output docs/reward-audit-20261008.json
python -m unittest tests.test_episode_v2 tests.test_trajectory_v2 -v
```

Neither command provisions cloud resources, loads model weights, or sends private traces to an external service.
