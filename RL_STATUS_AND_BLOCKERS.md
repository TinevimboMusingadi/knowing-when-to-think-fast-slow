# Knowing When to Switch: what we did, what failed, and why full TPU RL is unfinished

Updated October 8, 2026. This report explains the project state and the causes of delay. It complements [the chronological postmortem](POSTMORTEM.md), which preserves the original records.

## 1. The direct answer

**We completed supervised training and retained a usable SFT checkpoint. We did not complete a stable full reinforcement-learning run on the TPU.**

There are three different reasons:

1. **The original RL engine failed numerical checks after backward/update execution.** This was a real training failure, not a missing dataset or a lack of money alone.
2. **The replacement Tunix pipeline encountered compilation and operational failures before completing its full-model mixed-action RL pilot.** These included a larger-batch compilation stall, an incompatible Windows SSH client, and payload upload timeouts.
3. **Those attempts consumed most of the pilot allowance.** The current remaining pilot allocation cannot cover another complete compatibility/stability run plus setup and cleanup under the conservative planning rate.

The tight pilot budget is the immediate reason paid recovery is paused now. It does not explain all the earlier failures, and more money by itself would not establish that the implementation is correct.

The total approved ceiling has not been exhausted: approximately **$49.62 remains under $120** according to the conservative ledger. That money is allocated across pilot, SFT, RL, evaluation, and contingency. Only **$2.73 remains in the pilot**. The protected evaluation reserve is $16.

**Our TPU is currently off.** No training was launched while preparing this report. The other agent's Kaggle/Gemma project was not touched.

## 2. What this experiment is trying to train

The intended model is Qwen3-1.7B with one shared backbone and exactly three execution modes:

- **JEV:** score typed candidates with the preserved custom decision head.
- **Direct:** generate an answer without a reasoning trace.
- **Ordinary CoT:** generate ordinary reasoning before an answer or another action.

The learned mode tokens are `<mode:jev>`, `<mode:direct>`, and `<mode:cot>`. Clarification, lookup, defer, and answering are actions within the modes, not extra modes.

We keep the complete Qwen backbone and language head. The trainable components are rank-16/alpha-32 shared attention LoRA adapters, the original 128-wide two-layer candidate-attention head, and the three tied mode-token rows. Candidate backbone branches have isolated attention and reset positions; candidate comparison happens inside the permutation-equivariant head.

This is more involved than ordinary language-model LoRA training. The learner must combine two kinds of sampled action:

- Generated tokens, using language-model probabilities.
- Candidate selections, using categorical probabilities from the JEV head.

Both contribute to the policy objective. Prompt text, tool responses, padding, and replayed history must receive no policy loss. The frozen reference must include the corresponding adapters, decision head, and mode rows.

These requirements are why a stock language-only GRPO example is insufficient, and why replacing the JEV head with final-token scoring would change the experiment.

Private structured-CoT material remains excluded. The output limit is **max_generated_tokens:null**. The 512/1,024/2,048 buckets are context allocation shapes, not an arbitrary 512-token generation cutoff. Actual current context capacity is 2,048.

## 3. What we completed and can prove

### Supervised training

The original supervised run completed **438 optimizer updates on four TPU devices**. Its best retained checkpoint was **step 150**, selected by validation loss.

This is a real trained artifact. It is not proof of strong held-out performance, learned optimal routing, or an RL improvement.

The original prepared dataset contained 8,000 training episodes, 600 validation episodes, and 600 test episodes.

### A real saved demonstration

We retained a genuine SFT rollout where the model:

1. Requests a controlled lookup in direct mode.
2. Receives the missing record value, **1912**.
3. Selects JEV mode.
4. Scores four candidates and chooses **1912**.

The answer was correct and grounded in the observation. It used **19 generated tokens**. The always-direct comparison also answered correctly, using 35 generated tokens.

This is a useful example for the article, with explicit limitations:

- It is a saved replay, not fresh inference.
- The lookup is a controlled fixture, not live web browsing.
- One example does not establish an accuracy, latency, or general efficiency gain.
- Candidate encoding and head computation still cost work.
- It does not demonstrate a naturally learned JEV -> CoT or CoT -> JEV transition.

Evidence: [saved demo](docs/demo-record.json).

### One verified real RL update

The original fresh-optimizer restart completed **one verified finite RL update**.

Checkpoint comparison proved that 224 adapter tensors and the mode rows changed, while the frozen reference matched SFT. The decision head did not change during that first warmup update.

The run then hit its independent time limit before completing the second update.

This proves that a real optimizer update happened. It does not establish multi-step stability, improved decisions, improved routing, or a completed RL training run.

Evidence: [finite-update report](docs/rl-restart-report.json).

### Tunix conversion and local validation

We subsequently ported the recovery to Tunix/JAX while preserving the custom model. We downloaded and verified the best-SFT source, converted the adapters/head/mode rows, pinned the upstream model/backend revisions, and locked the dependencies.

Current local evidence includes:

- **116 Windows tests passed**, one backend-dependent test skipped.
- That skipped test passed separately in locked Linux.
- Full 1.7B FP32 probability parity passed against PyTorch.
- **13 locked numerical checks passed**.
- **10 locked protocol checks passed**.
- **Six locked transfer/orchestration tests passed**.
- **21 targeted transfer/connection controls passed**.

Some test groups overlap; these counts should not be summed as independent tests.

Maximum absolute full-model probability differences were approximately 0.00003952 for language probabilities and 0.000000477 for candidate probabilities, below the predeclared 0.001 tolerance.

Checks cover candidate isolation/order, masking, update state, reference preservation, mixed-action loss, zero-variance groups, and checkpoint continuation.

These are engineering acceptance results. Small-model or scripted diagnostic success does not replace sampled full-model GRPO on the real TPU.

Evidence: [overall validation](docs/recovery-validation.json), [full-model parity](docs/recovery-full-model-parity.json), [numerical checks](docs/recovery-numerical.json), [protocol checks](docs/recovery-protocol.json).

### Data and environment repair

The recovery dataset contains:

- **1,536 corrective SFT episodes**, 256 per behavior.
- **120 validation episodes**, 20 per behavior.
- **600 sealed test episodes**, 100 per behavior.

The six behaviors are fast decisions, direct answers, reasoning, reasoning-to-decision, clarification, and lookup.

We removed forced routing instructions, grouped related context variants, separated source/template families, randomized candidate order, checked clarification fields, and added typed defer/ask/lookup candidates.

252 verified public-gold examples were reused; zero private teacher completions were imported.

We preserved the original data and historical results rather than reinterpreting them as new evaluation.

Evidence: [versioned provenance manifest](docs/recovery-data-manifest.json).

## 4. What failed in the original TPU RL engine

The first engine used PyTorch/XLA. It did generate rollouts, calculate rewards, and perform backward passes. It did not remain numerically stable across the required updates.

An October 7 continuation saved a step-2 checkpoint whose rank-0 mode rows contained **18 NaNs**. Storage integrity checks passed, so this was not explained by a corrupt download. That run is permanently excluded from resume.

The last old-engine diagnostic pilot restored finite parameters, optimizer moments, and reference state on all four replicas. Every replica completed four backward passes. Policy/reference probabilities, losses, and advantages passed finite checks.

Then every replica failed the post-update parameter/optimizer-state validity check. No new step was accepted and no new checkpoint was published.

The evidence narrows the failure to the boundary between local gradients and resulting updated state. The old combined XLA optimizer call performed distributed gradient reduction and the optimizer together. We did not capture enough intermediate information to distinguish:

- A problem in gradient reduction.
- A problem in the optimizer calculation or state.
- A problem in the compiled execution of those operations.

The logs do not prove a specific AdamW bug, TPU hardware fault, or precise tensor-level mechanism.

Activation checkpointing helped memory: all required backward passes fit, with minimum measured memory headroom **16.36%**. FP32 moment restoration fixed a tested precision-loss path. Neither change resolved the complete distributed update failure.

There was an RL feedback signal. In the final pilot, two groups had varied rewards and two had identical rewards. Therefore, absent evaluation data or universally identical rewards did not explain the rejection.

None of those 16 episodes reached the removed output cutoff. The longest recorded generated sequence was 125 tokens. The failure was not caused by retaining a 512-token output allowance.

The numerical guard did its job by refusing invalid progress. The instrumentation was too coarse to establish the exact cause after the rejection.

Evidence: [old final pilot report](docs/rl-pilot-report.json) and [chronological investigation](POSTMORTEM.md).

## 5. Why switching to Tunix has not yet produced full RL

Tunix supports the relevant Qwen model, LoRA, and RL building blocks. Our custom head, tokenizer rows, checkpoint conversion, and mixed token/categorical objective still required integration and validation.

### First recovery pilot: some TPU updates worked

On October 7, the Tunix pilot ran on four actual TPU devices.

- Twelve small numerical checks passed, including 100 optimizer-reference updates.
- Three full-model supervised diagnostic updates completed.
- Adapters, head, and mode rows all changed.
- The first cold update took **814.5 seconds**, approximately 13.6 minutes.
- Warmed diagnostic updates took **81.6 and 83.2 seconds**.
- Minimum measured device memory headroom was **39.8%**.

Then the microbatch-2 compilation became unresponsive to remote health checks. The observed host process RSS was about 145 GiB.

Large compiler workload and host memory pressure are plausible explanations. We did not recover enough logs to prove OOM, hardware failure, or a non-finite update. The owned TPU was deleted and absence verified.

**Those three updates were disposable supervised diagnostics, not three sampled GRPO updates.** Corrective SFT had not started, and no selected new policy checkpoint was produced.

We repaired candidate encoding to use a bounded rematerialized loop instead of expanding a complete backbone computation per candidate. We also changed pilot ordering so the microbatch-1 sampled mixed-action checks come before optional larger-batch tuning.

The changed code passed current local acceptance. Its repaired full-model TPU behavior remains to be measured.

Evidence: [Tunix partial pilot report](docs/recovery-pilot-report.json).

### Subsequent connection failures prevented training

The October 8 attempts failed before training release at the SSH connection gate.

The first record did not retain enough scanner stderr to identify its cause.

The second did identify it: native Windows OpenSSH rejected the server's key-exchange method, with the message:

~~~text
choose_kex: unsupported KEX method sntrup761x25519-sha512@openssh.com
~~~

The Git-bundled scanner successfully obtained the owned endpoint's Ed25519 key. The runner now selects that compatible installed client, pins the key, checks ownership, and requires an exact fresh remote nonce.

This is a verified narrow repair. A complete nonce connection on the next live TPU still needs to pass.

No RL update ran in either failed connection attempt. Both nodes were deleted and absence verified.

Evidence: [first connection record](docs/recovery-connection-attempt-20261008.json), [identified compatibility failure](docs/recovery-connection-attempt-20261008-120500.json).

### Latest upload failure occurred before renting

The next attempt failed uploading a 35 MB payload before any TPU provisioning command.

The Windows CLI timed out at 600 seconds, with orphan workers keeping pipes open. Sequential CLI and larger SDK requests also timed out. Windows and Linux both reported socket-write timeouts. We have not established the underlying network cause.

A **256 KiB resumable transfer** then completed in **619.6 seconds**, with checksum verification. The production uploader independently verified those large bytes and created/verified a small new file.

The runner now completes verified upload before provisioning or starting accelerator accounting. Existing objects are not overwritten.

No model training happened in this attempt. The transfer repair is verified; the complete repaired cloud sequence has not yet run.

Evidence: [upload failure and repair](docs/recovery-upload-attempt-20261008.json).

## 6. How much of this is the budget?

### The total budget still has room

Approved total: **$120 including prior work**.
Current cumulative conservative estimate: **$70.38**.
Remaining under the total ceiling: **about $49.62**.

Current remaining allocations:

- Pilot: **$2.73**.
- Corrective SFT: **$8**.
- RL: **$20**.
- Evaluation: **$16**.
- Contingency: **$2.89**.

These are conservative planning bounds, not reconciled bills. Actual billing is still unknown.

The compute planning rate is **$5.52 per whole-slice hour**, including a 15% margin. The last launch checked a lower spot quote, but that historical quote is not guaranteed and has not replaced the approved planning method.

At the planning rate, $2.73 allows approximately **30 minutes**. That must cover provisioning, setup/restoration, compilation, required checks, checkpoint/export, and cleanup. It is not 30 minutes of warmed training.

### The pilot allowance became the bottleneck

The original recovery pilot had $6. Its first incomplete attempt used an estimated $4.13, leaving $1.87.

The user approved moving **$5 from contingency to pilot**, making pilot $11 and contingency $2.89.

Subsequent conservative attempt charges were approximately:

- First connection failure: **$1.50**.
- Identified Windows SSH failure: **$1.38**.
- Latest pre-provisioning upload failure: **$1.26**.

That left **$2.73** in pilot.

The upload failure's ledger charge conservatively includes setup time even though no accelerator existed. This is an over-inclusive planning bound, not proof that Google billed a TPU for that attempt. The repaired runner avoids starting the accelerator clock during future payload upload.

We have not silently refunded that historical charge or replaced the ledger with an unverified bill.

### What the pending transfer would change

A later proposal asks to move **$4 from RL to pilot**:

- Pilot allocation would become $15.
- RL allocation would become $16.
- Evaluation would remain $16.
- Total ceiling would remain $120.
- Remaining pilot would become approximately **$6.73**, about **73 minutes** at the planning rate.

**That proposal is unanswered. This report does not authorize it.**

It would provide more room to complete the pilot. It would not prove that the repaired connection/compiler/learner will work, and it would reduce the later RL allocation.

### Why we cannot give a reliable full-RL duration yet

The approximately 83-second warmed timing was a supervised diagnostic update, not a GRPO update.

GRPO requires sampled multi-action trajectories, language and candidate scoring, old/current/reference likelihoods, distributed gradient execution, and periodic validation/checkpointing. Different context lengths can trigger additional compilation.

We have not measured a complete stable full-model Tunix GRPO update. Multiplying 83 seconds by 64 would therefore give a misleading completion estimate.

We need measured cold and warmed mixed-action update timings before projecting whether the allocated RL budget can finish the target.

## 7. What is already fixed, and what remains unresolved

Repairs with local or narrow direct evidence:

- Best-SFT source integrity and conversion.
- Full-model PyTorch/JAX probability agreement.
- Isolated candidates, reset positions, loss masking.
- Explicit FP32 local/reduced-gradient and optimizer-state checks.
- Correct mixed categorical/token loss and frozen-reference preservation.
- Checkpoint next-update continuation in acceptance fixtures.
- Windows SSH client selection and exact nonce verification logic.
- Small-chunk checksummed transfer.
- Upload-before-rental sequencing.
- Core mixed-action pilot before optional larger-batch tuning.
- Owned cleanup verification and preserved failure reports.

Still unresolved or unproved:

- Exact old-engine numerical failure mechanism.
- Exact cause of the earlier Tunix larger-batch compiler stall.
- Complete repaired live TPU connection sequence.
- Three accepted sampled full-model GRPO pilot updates with save/reload.
- Corrective SFT reaching 95% valid protocol execution.
- At least 20 consecutive finite full-model RL updates; target 64.
- Reliable mixed-action throughput/cost projection.
- Balanced held-out accuracy/compute comparison.
- Naturally learned JEV -> CoT and CoT -> JEV demos.
- Any accuracy, calibration, or efficiency improvement from RL.

The old mode-switch repository does not fill this gap: its mock GRPO paths do not perform real optimizer updates, and its evaluator differs from this custom head.

The investigated Unsloth DecisionTrainer is a different decision-SFT architecture, not a verified replacement for this TPU experiment. A separately learned reward model is not currently necessary for our verifiable tasks.

Evidence: [reward counterexample audit](docs/reward-audit-20261008.json), [Unsloth/reference-repo investigation](docs/unsloth-switching-plan.md).

## 8. What we should improve about the process

Some delays were avoidable engineering choices, and should not be attributed only to a small budget.

**Connection compatibility should have been exercised before repeated paid attempts.** Retaining scanner stderr earlier would have exposed the actual Windows key-exchange failure sooner. The Git scanner proof now narrows this issue.

**Optional throughput tuning ran too early.** The first Tunix pilot reached the larger-batch compiler stall before demonstrating sampled mixed-action GRPO. The revised order prioritizes the smallest complete learning path.

**Large payload transfer should have finished before accelerator rental accounting.** The repaired runner verifies transfer first. The old conservative ledger had also counted pre-provisioning time, consuming allowance without proving billed accelerator use.

**Status messages needed clearer evidence boundaries.** Passing local tests, observing a cloud VM READY state, performing diagnostic updates, and completing stable RL are different milestones. Reporting should state the exact accepted update type/count and whether a recoverable checkpoint exists.

**Numerical diagnostics needed a separate reduction boundary.** The new update sequence records local gradients, reduced gradients, clipping, new parameters, and optimizer moments separately. This should make a future rejection more useful, but needs real TPU validation.

**Repeated validation is necessary after backend changes, but is not training progress.** Most current acceptance runs are local and do not incur accelerator rental. They have consumed time, and the report should say so plainly.

Keeping the evaluation reserve was appropriate for the approved research plan. It means we cannot spend every remaining dollar on debugging and still claim to have completed the planned comparison.

## 9. What is needed before calling the experiment complete

1. Resolve the current pilot allocation decision while keeping the total cap and evaluation reserve intact.
2. Revalidate any configuration/source change against the locked current-revision checks.
3. Upload/verify the payload before renting.
4. Confirm the current owned endpoint by pinned nonce, then detect all four TPU devices.
5. Complete microbatch-1 full-model mixed-action pilot updates, including save/reload and finite/frozen-state checks.
6. Measure complete mixed-action timing and project every remaining stage against its allowance.
7. Run corrective SFT from best step150 with a fresh optimizer; require 95% valid protocol execution.
8. Run real GRPO from corrected SFT with a frozen exact reference, checking every update and preserving accepted state.
9. Complete affordable balanced evaluation with frozen test choices and honest external baselines.
10. Export verified artifacts and demo attempts, then verify owned TPU deletion.

If a gate fails or the allocation cannot cover the measured work, retain the best valid weights and report the incomplete result. Do not substitute a mock update, discard the decision head, silently shorten the required run, or claim gains from a selected example.

This report itself did not launch another attempt or change the budget.

## 10. Where everything is

Workspace: **D:\knowing-when-to -think**

Repository: https://github.com/TinevimboMusingadi/knowing-when-to-think-fast-slow

Current pushed code: main, b21bd87. This newly written report is local until separately committed/pushed.

Documents:

- [Chronological postmortem](POSTMORTEM.md)
- [Medium article draft](docs/article.md)
- [Saved demonstration](docs/demo-record.json)
- [Weight inventory and exclusions](docs/saved-weights.json)
- [Current validation record](docs/recovery-validation.json)
- [Data provenance](docs/recovery-data-manifest.json)

Active configuration and local evidence:

- configs/recovery.json — active $120 configuration and backend/model settings.
- runs/recovery-v2/budget.json — conservative cumulative ledger.
- runs/recovery-v2/session.json — last owned session; absent/deleted.
- runs/recovery-v2/source-sft/ — verified original best-step150 source/tokenizer.
- runs/recovery-v2/converted/ — converted trainable state/manifest.
- runs/recovery-v2/local-numerical.json, local-protocol.json, full-model-parity.json — strict local reports.
- data/recovery-v2/ — train/validation/sealed test and manifest.
- .cache/public-base/ — verified frozen Qwen shards.
- runs/recovery-v2/history/ — archived sessions and operational evidence.

Implementation:

- switching/experiment_v2.py — canonical audit/gate/execution entry.
- scripts/recovery_cloud.py — owned cloud runner and verified transfer.
- switching/tunix_experiment.py — staged pilot/SFT/RL/evaluation worker.
- switching/tunix_train.py — actual mixed-action learner.
- switching/tunix_model.py — preserved custom model/head.
- switching/tunix_optim.py and tunix_distributed.py — numerical updates/reduction.
- switching/episode_v2.py, trajectory_v2.py, tunix_environment.py — environment and rollouts.
- switching/checkpoint_v2.py — checkpoint/recovery state.
- switching/tunix_evaluate.py and evaluation_v2.py — comparisons.

Exact best-SFT cloud restore location is recorded in docs/saved-weights.json and configs/recovery.json. These are adapter/head/mode-row recovery bundles; original Qwen weights must be loaded separately.

For a complete new-session operational handoff, including commands, exact restore URI, dependency pins, helper caveats, and the unanswered approval, use:

**D:\knowing-when-to -think\runs\recovery-v2\HANDOFF.md**

That handoff is private and ignored by Git. Do not publish its operational details or private source material.
