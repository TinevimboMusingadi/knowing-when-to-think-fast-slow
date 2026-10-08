# Knowing When to Switch

An exploratory **Qwen3-1.7B** experiment with a complete shared backbone, its original language head, and a typed decision head. Three learned tokens select **JEV**, **direct**, or **ordinary CoT**. Asking and controlled lookup are actions inside those modes.

## Current evidence

Historical SFT completed 438 updates; its best retained checkpoint is step 150. One saved example performs lookup → JEV → a grounded correct answer. One verified RL update changed adapters and mode embeddings. Stable RL and performance gains remain unestablished. The final PyTorch/XLA pilot failed after reduction/optimizer execution and was verified deleted.

The recovery implementation uses pinned Tunix/JAX with a custom mixed-action learner, FP32 trainable/optimizer state, versioned records and cost gates. Upstream's text-only GRPO learner does not support our custom head unchanged. **103 reference/control tests pass**. The first locked Linux acceptance passed 12 numerical and ten protocol checks, plus full 1.7B probability parity at the unchanged 0.001 tolerance. These are engineering checks, not held-out performance results.

The first recovery TPU attempt passed 12 numerical checks and three full-model supervised diagnostic updates on all four devices. Warmed updates took 81.6 and 83.2 seconds, after a 13.6-minute cold step. The larger-batch compilation stopped responding to remote health checks. The exact cause remains unresolved; another numerical failure was not established. The owned TPU was **verified deleted**, preserving the numerical report. No corrective SFT or full-model mixed-action GRPO update completed. The [pilot record](docs/recovery-pilot-report.json) separates observed progress, incomplete logs and cost estimates. On October 8, compiler and host-dispatch changes passed fresh locked Linux acceptance: 13 numerical checks, ten protocol checks and full-model probability parity. Their TPU behavior remains unmeasured. See the [recovery validation status](docs/recovery-validation.json).

See [the postmortem](POSTMORTEM.md), [article draft](docs/article.md), [retained weights](docs/saved-weights.json), and [v2 data manifest](docs/recovery-data-manifest.json). Historical reports retain their original schemas and budgets.

## Model and protocol

Train rank-16/alpha-32 LoRA on Q/K/V/O, the preserved 128-wide two-layer candidate-attention head, and three tied input/output mode rows. Other weights remain frozen. JEV stops language decoding, encodes independent state/candidate branches with reset positions, and compares them only inside the permutation-equivariant head. Typed actions include answer, defer, ask and lookup. Defer leaves the next mode to the model.

Private teacher completions are excluded. Buckets of 512/1,024/2,048 tokens are allocation shapes. **There is no fixed output token allowance.** Generation stops at EOS, task completion or actual context capacity. Episodes permit four transitions, two lookups and eight actions.

## Preparation and local acceptance

Use isolated Python 3.12 and hash-locked dependencies. The TPU lock requires Linux glibc 2.31+. Tunix revision and Qwen revision are fixed in [the single experiment configuration](configs/recovery.json).

```bash
python scripts/bootstrap_tunix.py
uv venv --python 3.12.14 .venv-tunix-linux
uv pip sync --python .venv-tunix-linux/bin/python --require-hashes requirements-tunix-cpu.lock
export PYTHONPATH="$PWD/.vendor/tunix:$PWD"
export XLA_FLAGS=--xla_force_host_platform_device_count=4
```

The related source repository remains read-only. Data rebuilding needs the existing audited legacy training JSONL and cached original public GSM8K training JSONL. The new splits contain 1,536 corrective SFT, 120 validation and 600 sealed test episodes. Paired source/context variants stay together; template families differ across splits. The manifest records 252 public-gold examples reused and zero private completions imported. Retain the [GSM8K MIT license](https://github.com/openai/grade-school-math/blob/master/LICENSE).

```bash
python -m switching.data_v2
python -m unittest discover -s tests -q
python -m switching.verified_download --uri gs://keeper-file-storage/knowing-when-to-switch/20261006-122830/checkpoints/step-000150-best-sft-rank0-565877d3 --output runs/recovery-v2/source-sft
python -m switching.conversion --checkpoint runs/recovery-v2/source-sft --output runs/recovery-v2/converted
python scripts/export_reference.py --output runs/recovery-v2/tiny-reference
.venv-tunix-linux/bin/python -m switching.validate_tunix --fixture runs/recovery-v2/tiny-reference --output runs/recovery-v2/local-numerical.json
.venv-tunix-linux/bin/python -m switching.validate_protocol_v2 --fixture runs/recovery-v2/tiny-reference --output runs/recovery-v2/local-protocol.json
```

Reference exports/conversion use the separate PyTorch environment. Download the pinned Qwen revision, then run sequentially to bound host memory:

```bash
python scripts/export_full_reference.py --base /absolute/path/to/pinned-qwen
.venv-tunix-linux/bin/python -m switching.validate_full --base /absolute/path/to/pinned-qwen
python -m switching.experiment_v2
```

The audit cannot provision a TPU. Reports must match current code, configuration, dataset and installed locked dependencies, and import the complete pinned package. Missing, failed, development-only or stale reports block launch. Full probability tolerance is fixed at maximum absolute difference 0.001.

## Owned TPU execution

```bash
python -m switching.experiment_v2 --execute
```

This is the only active paid entry point. Old PyTorch TPU launch/restart paths are disabled. Credentials, bucket access, quota, runtime, current interruptible pricing and ownership are checked first. It requests only interruptible `v5litepod-4` in `us-west4-a`, with no larger hardware, on-demand fallback or model substitution.

The **$120 cumulative cap** includes conservative prior **$62.11**. On October 8, the user approved transferring $5 from contingency to the pilot: active allocations are $11 pilot, $8 SFT, $20 RL, $16 evaluation and $2.89 storage/cleanup/contingency. The first recovery pilot used a conservative **$4.13**, bringing the cumulative bound to **$66.24** and leaving **$6.87** for the next pilot. Accounting uses the whole-slice $4.80/hour bound plus 15%, including setup, compilation and verified deletion. These are estimates, not billing totals. Evaluation money is protected. Current interruptible rates must be checked at launch against [Google pricing](https://cloud.google.com/tpu/pricing).

The worker performs pilot → corrective SFT → protocol validation → mixed-action GRPO → sealed comparison → export. VM and independent local watchdogs enforce deadlines. Only resources carrying both this run's ownership label and `kws_schema=v2` may be deleted. Absence must be verified. The separate Kaggle/Gemma project is outside this workflow.

Updates check local gradients, FP32 reduced gradients, clipping, parameters and moments separately. Invalid state terminates the attempt without retries or publication. Orbax bundles include actor/reference, optimizer/scheduler, RNG, cursor and accepted-update count. GCS completion markers come last after checksums succeed. Restore uses the caller's device topology.

## Evaluation and demo

Compare native Qwen thinking/non-thinking, SFT/RL switching, fixed modes and validation-selected confidence switching. Native Qwen uses its supported templates and original vocabulary/head without custom mode annotations. Freeze test IDs, decoding, checkpoint hashes and threshold before outcomes. All applicable policies use the same 120 or 60 balanced paired test episodes; insufficient funds produce an incomplete report.

Retain failures, token/candidate work, synchronized warmed latency, cold first-call overhead, calibration, Wilson intervals and paired source-group bootstrap intervals. JEV calibration is conditional over answer candidates. One training seed is exploratory. Short JEV output is not an assumed speed advantage.

```bash
python -m switching.demo_v2 --replay docs/demo-record.json
.venv-tunix-linux/bin/python -m switching.demo_v2 --base /absolute/path/to/pinned-qwen --checkpoint-gcs gs://bucket/owned-run/checkpoints/completed-bundle --episode /absolute/path/to/episode.json
```

Saved replays and live execution are labeled separately. Demonstrations retain all attempts and show only naturally selected transitions. Missing transitions remain missing results.

Related work: [Think Only When You Need](https://arxiv.org/abs/2505.14631), [AdaptThink](https://arxiv.org/abs/2505.13417), and [the JEV-style Qwen architecture](https://github.com/avbiswas/bev-train). This project tests a shared generative/typed model with within-task transitions; it promises neither novelty nor gains.
