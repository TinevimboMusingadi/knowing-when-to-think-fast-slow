# RL continuation and budget revision

The user approved a $60 total cap on October 7, 2026. Historical $50 records remain historical. The launcher defaults to $50; this continuation explicitly supplies $60 and reserves $5 for storage.

Prior compute upper bound: $44.949919. A 90-minute whole-attempt window on preemptible v5litepod-4 is budgeted at the conservative on-demand bound of $4.80/hour plus 15%, or $8.28. Combined with the storage reserve this is $58.229919. These are estimates, not billed charges. Setup, compilation, restore, uploads and teardown consume this window. Useful training time and throughput are unverified.

```powershell
python scripts/restart_rl.py --continue-rl --source-run 20261006-223323 --total-cap 60 --max-dollars 8.28 --launch
```

This resumes the fresh-SFT run's completed RL update, restoring four synchronized rank checkpoints, optimizer, scheduler, RNG and frozen SFT reference. The generated-token limit remains 512 and the GRPO group size remains four. Every completed update is saved; validation runs every five updates on eight validation episodes. The prior best checkpoint remains available in its original run. An independent watchdog deletes only the owned TPU at its deadline. No automatic hardware upgrade is permitted.

Tunix exploration follows this attempt. Its Qwen3/GRPO support does not establish compatibility with the custom JEV head or PyTorch checkpoint. Adapter/mode embedding conversion and joint token/decision policy-loss parity must be verified before paid migration.
