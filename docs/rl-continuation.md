# RL continuation and budget revision

Historical PyTorch/XLA record through the failed $70 pilot. Its old launch commands are disabled. The current $120 Tunix recovery uses [the recovery configuration](../configs/recovery.json), [current validation status](recovery-validation.json) and [the recovery instructions](../README.md). The earlier numbers and findings below retain their original scope.

The user approved a $60 total cap on October 7, 2026. Historical $50 records remain historical. The launcher defaults to $50; this continuation explicitly supplies $60 and reserves $5 for storage.

Prior compute upper bound: $44.949919. A 90-minute whole-attempt window on preemptible v5litepod-4 is budgeted at the conservative on-demand bound of $4.80/hour plus 15%, or $8.28. Combined with the storage reserve this is $58.229919. These are estimates, not billed charges. Setup, compilation, restore, uploads and teardown consume this window. Useful training time and throughput are unverified.

```powershell
python scripts/restart_rl.py --continue-rl --source-run 20261006-223323 --total-cap 60 --max-dollars 8.28 --launch
```

This resumes the fresh-SFT run's completed RL update, restoring four synchronized rank checkpoints, optimizer, scheduler, RNG and frozen SFT reference. The generated-token limit remains 512 and the GRPO group size remains four. Every completed update is saved; validation runs every five updates on eight validation episodes. The prior best checkpoint remains available in its original run. An independent watchdog deletes only the owned TPU at its deadline. No automatic hardware upgrade is permitted.

Tunix exploration follows this attempt. Its Qwen3/GRPO support does not establish compatibility with the custom JEV head or PyTorch checkpoint. Adapter/mode embedding conversion and joint token/decision policy-loss parity must be verified before paid migration.

## Diagnosed memory failure

The continuation finished rollout and old/reference scoring, then device allocation failed at the synchronized finite-input check. A worker requested another 96 MB with approximately 545 KB free. Other replicas waited in the collective; ordered process futures hid the worker exception from the parent log. This was device memory exhaustion, not evidence of a new non-finite gradient. No additional optimizer update was completed before the repair.

Worker exceptions now persist separately. The memory repair enables XLA-aware reentrant activation checkpointing for the decoder and uses training mode for gradient scoring. LoRA, attention and decision-head dropout remain zero. CPU parity checks compare both token and categorical decision probabilities and all trainable gradients with the uncheckpointed path. The repair also records per-replica memory after each backward pass. TPU validation is in progress; no speedup or successful update is asserted here.

Both retries reused the same VM, source RL optimizer state, frozen SFT reference and original deadline. Their historical generated-token limit was 512.

## Final outcome and recovery work

The memory repair completed optimizer step 2 with at least 16.37% measured peak memory headroom. Step 3 was refused by the non-finite input/loss guard. Integrity-checked ZIP storage from the saved rank-0 checkpoint contains 18 NaNs among its 6,144 mode-embedding elements; the corresponding source step-1 storage has none. The step-2 checkpoint must not be resumed. This is not evidence of a performance gain or stable RL training.

The TPU deletion completed at 2026-10-07 10:16:59 UTC. The watchdog failed without preserving its underlying error; manual cleanup was verified. Diagnostics now retain those errors and use the installed Python SDK entry point on Windows. The conservative cumulative compute estimate is $53.686962, or $58.686962 with the full $5 storage reserve. Actual billing remains unverified. No TPU is running.

New guards check updated parameters and optimizer moments before increasing the completed-step counter or publishing a checkpoint. A failed update cannot overwrite recoverable checkpoints. Optimizer restore now preserves original FP32 moments rather than rounding through BF16. The rounding defect is verified independently; its role in the TPU NaNs is not established. Numerical logs distinguish policy, old policy, reference, advantages, and loss. CPU regression checks pass; these repairs still require TPU validation.

At the user's request, `max_generated_tokens: null` now removes the fixed episode output allowance. Generation stops at completion or available context; current context capacity remains 2,048 tokens, with XLA prefill buckets consuming part of that capacity. Longer generation requires a new memory and cost pilot. Spending, mode-transition and lookup controls remain active.

Tunix is researched but not ported. Its [official repository](https://github.com/google/tunix) supports GRPO and Qwen models. A valid migration needs the candidate head, reset positions and isolated candidate attention, mode embeddings, tokenizer/vocabulary alignment, adapter conversion, and joint token/categorical policy-loss parity. Standard text-only GRPO does not preserve this experiment.

## Approved $70 pilot

The user approved a $70 cumulative cap, and run `20261007-132053` requested preemptible v5litepod-4 with an $8.28 allowance and a 90-minute deadline. It restored the synchronized finite step-1 checkpoints from `20261006-223323`, using the repaired FP32 optimizer restore and post-update guard. The pilot was limited to two new updates and validation after each.

All four devices were detected, 66 regression tests passed on the VM, and weights, optimizer moments and reference state passed the restore audit on every replica. Four sampled episodes completed backward passes per replica with finite policy, old-policy, reference, advantage and loss checks. The lowest recorded peak-memory headroom was 16.36%.

The first optimizer update failed the new validity check. That check rejects non-finite parameters or moments and negative second moments; its log did not identify which tensor or component failed. No new completed step or recoverable checkpoint was published. This rules out calling the pilot successful and does not establish the precise optimizer mechanism. The last finite source checkpoint remains available.

The initial detached Windows watchdog reported empty cloud errors. A hidden background probe with closed stdin succeeded; a replacement monitor confirmed the original deadline. Cloud child processes now use explicit hidden-window execution, errors retain an exit code when output is empty, and successful monitor state changes are logged. The original failure's precise Windows cause remains unverified.

Failure records were recovered from GCS. Deletion completed at 2026-10-07 13:58:05.595242366 UTC and resource absence was verified. Conservative attempt compute was $3.422147, cumulative compute $57.109109, or $62.109109 including the full storage reserve. These are bounds rather than verified bills. No TPU is running. See [the machine-readable pilot report](rl-pilot-report.json). The next paid test should isolate and compare the device optimizer against a CPU reference before another full-model attempt.
