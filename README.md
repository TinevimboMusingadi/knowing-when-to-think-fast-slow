# Knowing When to Switch

A research prototype for **Qwen3-1.7B**, one shared backbone, a language head,
and a permutation-equivariant candidate-scoring head. The model learns three
modes: JEV-style typed decisions, direct answers, and ordinary CoT.

Emitting the JEV token immediately stops language decoding. The head scores
candidates against observed conversation context, including earlier reasoning or
retrieved evidence. No decision JSON body is generated on this fast path.

## Status

Implementation and local verification are in progress. No trained-model
performance or TPU throughput is claimed until recorded by a real experiment.
The initial tasks are verified arithmetic and controlled context fixtures.
They do not establish general-purpose reasoning or autonomous tool competence.

## Data

Existing math prompts are matched against the original GSM8K **training** answers
before reuse. Existing teacher completions are never imported. The public GSM8K
source is [grade-school-math](https://github.com/openai/grade-school-math), licensed
under MIT. Retain its license when redistributing derived examples. Held-out
evaluation uses disjoint procedural expression families rather than GSM8K test.

```powershell
python -m switching.data --source 'D:/mode-switch-llms/data/train_phase3_60k.jsonl' --gold-file data/gsm8k_train.jsonl
python -m unittest discover -s tests -v
```

## TPU execution

`scripts/cloud.py --launch` uploads an explicit allowlist to a unique GCS prefix,
creates only a four-chip preemptible v5e, runs a pilot, then SFT and GRPO. It
installs pinned PyTorch/XLA 2.8 and Qwen-compatible Transformers 4.57.3 on the VM.
Do not launch until local checks pass. A pilot must retain 15% memory headroom
and projected SFT cost must fit the stage allowance.

Interrupted SFT can resume from completion-marked GCS checkpoints, with one
optimizer and random-state snapshot per replica. Recovery checks the archived
dataset hashes before continuing. After downloading that run's budget record
to `runs/<run-id>/budget-remote.json`, use
`python scripts/cloud.py --launch --zone us-west4-a --resume-sft-run <run-id>`.
Recovery retains the successful microbatch of one and skips a repeated pilot.
The cost projection uses full optimizer updates after a twelve-update warmup;
elapsed spending remains checked on every microbatch.

New spend is limited to $50: pilot $5, SFT $20, RL $15, evaluation $5, storage $5.
The guard uses a conservative whole-slice $4.80/hour rate and a 15% margin;
this is an estimate, not a claim about the live Spot price or actual billing.
The independent local watchdog and startup trap delete only the uniquely named
TPU carrying the run's ownership label. Other projects and Kaggle sessions are
outside this launcher's resource scope.

GCS: `gs://keeper-file-storage/knowing-when-to-switch/<run-id>/`.
Checkpoints contain trainable parameters, optimizer/scheduler state, random
state and progress. Uploads use checksums and completion markers written last.
No GCS object is overwritten, and no credential is included in the payload.

The workflow performs a 24-example SFT evaluation smoke check, then attempts
paired comparisons on the full 600-example holdout within the evaluation budget.
Incomplete comparisons are labeled as partial; never describe a smoke check or
budget-truncated run as a complete benchmark.

## Research comparisons

Evaluate learned SFT and RL switching against static direct, static CoT, static
JEV, and a validation-selected confidence cascade. Untouched Qwen requires its
native chat protocol and is a separate baseline, not a randomly initialized
decision head. Report accuracy, calibration, generated tokens, forward passes,
cold/warmed latency, and actual cost with failures and confidence intervals.

Related work: [AdaptThink](https://arxiv.org/abs/2505.13417),
[PATS](https://arxiv.org/abs/2505.19250),
[Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B), and
[BEV's decision architecture](https://github.com/avbiswas/bev-train).
This implementation uses independent candidate branches rather than copying
BEV source. Candidate permutation invariance must be verified numerically.

No broad novelty claim is made: test whether adding typed decisions and context
acquisition improves the accuracy/compute tradeoff in this controlled setup.
