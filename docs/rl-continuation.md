# RL continuation and budget revision

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

Both retries reuse the same VM, source RL optimizer state, frozen SFT reference and original deadline. The generated-token limit remains 512.
