# Knowing When to Switch

A research prototype for **Qwen3-1.7B**, one shared backbone, a language head,
and a permutation-equivariant candidate-scoring head. The model learns three
modes: JEV-style typed decisions, direct answers, and ordinary CoT.

Emitting the JEV token immediately stops language decoding. The head scores
candidates against observed conversation context, including earlier reasoning or
retrieved evidence. No decision JSON body is generated on this fast path.

## Status

SFT completed 438 optimizer updates on a four-chip preemptible TPU. Real GRPO
reached step 3 and stopped after its gradient guard rejected a non-finite update.
Only four context-acquisition episodes were evaluated before the comparison
timed out; that sample does not establish a performance gain. The TPU was deleted.
A bounded Qwen3-1.7B TPU restart from SFT with a fresh optimizer completed one
finite RL update. Its second update timed out; multi-step stability and any
performance gain remain unproven. The restart TPU was also deleted.
See [the measured recovery report](docs/rl-restart-report.json).
The initial tasks are verified arithmetic and controlled context fixtures.
They do not establish general-purpose reasoning or autonomous tool competence.

### Bounded RL recovery

`python scripts/restart_rl.py` prepares an RL-only recovery payload; add
`--launch` to create a preemptible `v5litepod-4`. It verifies the saved best SFT
checkpoint, starts a fresh optimizer, retains the repaired numerical checks,
samples four episodes per prompt, and stops after at most three updates.
The recovery uses six balanced validation fixtures as a pilot check rather than
a capability benchmark. Setup counts against its $1.80 allowance, and an
independent teardown deadline preserves the $50 ceiling and $5 storage reserve.
Prior compute is conservatively bounded at $43.19 using the audited deletion
completion timestamp; this is not an actual billing total. The 60-example
comparison remains pending.
The recorded recovery cost bound was $1.76, bringing the cumulative compute
bound to $44.95 while preserving the $5 storage reserve. No additional paid
attempt fits that conservative allocation until billing is reconciled or the
experiment budget is revised.
RL progress logs now separate policy scoring, reference scoring, each backward
pass, and optimizer execution. Completion timings include the existing device
synchronization where present; these diagnostics have CPU integration coverage
and have not yet been exercised in another TPU run.

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
The cost projection uses compilation-free optimizer updates after warmup, a
25% throughput margin, and an explicit reserve for additional cold graphs.
Elapsed spending remains checked on every microbatch. Replicas agree on budget
decisions, and independent process timeouts bound both SFT and RL stages.

Dataset changes require fresh SFT. `--fresh-sft-from-run <run-id>` carries the
earlier cost ledger and reuses the verified batch size without restoring old
weights or pretending to resume a different dataset.

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

### Offline recovery and smaller evaluation

The loss now computes in FP32, uses `expm1` for the KL estimator, and bounds
exponential log-ratio tails at 20. This changes the objective outside that range;
tail counts are recorded. Non-finite inputs/losses are rejected before backward
on every replica. Gradient norms are inspected before clipping can alter the
evidence, and RL saves each completed update. Full public sampled traces are
retained in run artifacts for diagnosing a future failure.

```powershell
python scripts/offline_probe.py
python -m unittest discover -s tests -v
python -m switching.data --output data/procedural-reproduction
python -m switching.subsets --data data/procedural-reproduction/test.jsonl --output data/test60.jsonl --per-behavior 10
python -m switching.subsets --data data/procedural-reproduction/val.jsonl --output data/validation30.jsonl --per-behavior 5
```

These holdouts reproduce the original procedural test/validation episodes
without any private training source. Test selection contains 10 episodes per
behavior; validation contains five. Selection uses seeded ID hashes, never
model results. This reproduction's synthetic training split does **not** replace
the original 4,000 verified reused training prompts.

With the base weights already cached and a verified local SFT checkpoint:

```powershell
python -m switching.compare --device cpu --offline --sft checkpoints/best-sft --data data/test60.jsonl --validation data/validation30.jsonl --batch-size 2 --output runs/sft-validation60
```

Omit `--rl` for an SFT-only comparison. Add a verified local RL checkpoint to
include RL switching. `--offline` prohibits Hugging Face downloads. The full
1.7B model is not cached in the current local workspace, so this 60-episode
benchmark has **not** been run. See [offline recovery evidence](docs/offline-numerics.json).
CPU checkpoint loading ignores TPU-only random-state metadata. On TPU, the
native Qwen baseline now uses fixed-cache decoding and suppresses the three
added mode tokens. Reports are saved after each completed policy batch and on
interruption; per-policy applicability and completeness are explicit. Use a fresh
output directory to preserve existing records. No new paid run is authorized
by these recovery commands.

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
