# Knowing When to Switch: end-of-run postmortem

October 7, 2026. This postmortem preserves the stopped PyTorch/XLA attempt. Its TPU was verified deleted. The later authorized Tunix recovery is recorded separately below; no recovery TPU has launched yet.

We built a working supervised prototype and saved a real context-acquisition demonstration. We did not finish stable reinforcement learning or establish an accuracy or efficiency improvement. That distinction is the main result of this attempt.

The project asks whether one Qwen3-1.7B model can learn when to use three execution paths: JEV-style candidate scoring, direct answering, and ordinary chain-of-thought. Three learned mode tokens select those paths. Clarification and lookup are actions within them. The model shares its backbone and language head, with LoRA adapters, a decision-scoring head and new token embeddings as the trainable components. Private teacher material is excluded from this write-up and its demo export.

## What worked

Supervised fine-tuning completed 438 optimizer updates across four devices on a preemptible v5e TPU. The best retained SFT checkpoint was selected at step 150 by validation loss. Completing training and selecting a checkpoint do not establish held-out accuracy.

The data pipeline prepared 8,000 training episodes, 600 validation episodes and 600 test episodes. It reused 4,000 verified public math prompts without importing private teacher completions, and checked duplicate problems and source-family overlap. These are controlled arithmetic and context tasks, rather than a general reasoning benchmark. Split separation does not establish that the base model never encountered related material during pretraining.

The local checks exercised candidate-order behavior, isolated candidate backbone branches, reset positions, padding masks, trainable head updates, joint token/decision policy loss, frozen reference preservation and checkpoint recovery. The last confirmed full suite passed 66 tests locally and on the TPU VM. These checks use a small randomly initialized Qwen architecture. A separate 12-test cloud suite passed after watchdog changes. Later diagnostic edits remain untested because work was stopped.

A restart from the SFT checkpoint completed one finite real GRPO update. A checksum-verified comparison found changes in 224 LoRA tensors and the mode-token embedding, with the frozen reference matching SFT. The decision head did not change in that first warmup update. We therefore have evidence of a real adapter update, rather than evidence that RL improved routing or decision accuracy.

Activation checkpointing repaired an earlier memory failure sufficiently to complete all four backward passes per replica in the final pilot. Its lowest recorded peak-memory headroom was 16.36%, above the 15% target. This does not mean the complete training path was stable.

## The demo we actually have

The saved SFT rollout for `test-lookup-36` asks the model to return an undisclosed record value. The model requests the controlled lookup in direct mode, receives the value **1,912**, emits the JEV token, and selects **1,912** from four candidates.

The recorded route is:

```text
Direct: request lookup of record-test-36
Environment: return the record value, 1912
JEV: score [1893, 1912, 1906, 1902]
Decision: 1912
```

The answer was correct and grounded in the returned evidence. The rollout used 19 generated tokens. An always-direct baseline also answered this same fixture correctly, using 35 tokens. This is one recorded example, not evidence of a general token, latency or accuracy advantage. The JEV path still incurs routing, candidate encoding and head computation.

[The sanitized demo record](docs/demo-record.json) preserves the actions, candidate probabilities, answer and token counts. It is a replay of an existing saved rollout; no new inference was run for this postmortem. The lookup is a controlled fixture, not live web research. It contains no reasoning trace.

Learned JEV → CoT and CoT → JEV demonstrations are still missing. Scripted controller tests exercise those transitions, but supplied actions are not evidence that the trained model learned to select them.

## What did not work

RL did not become reliably stable. An earlier run reported progress through step 3 before rejecting non-finite gradients; that counter does not establish a verified finite step-3 checkpoint. A fresh-optimizer restart completed one verified finite update and then timed out. The October 7 continuation completed step 2, but its saved rank-0 mode embeddings contained 18 NaNs. The affected embedding storage passed its integrity check, so those NaNs were not a download artefact. That checkpoint must not be resumed.

The final repaired pilot, `20261007-132053`, restored finite weights, optimizer moments and reference state on all four replicas. Each replica completed four backward passes, with finite policy probabilities, reference probabilities, advantages and losses. The optimizer then failed the post-update state check. The check rejects non-finite tensors and negative second moments; its log did not identify the precise failing component. No new step was accepted and no new recoverable checkpoint was published.

We know the failure occurs in the update stage after the sampled likelihood and backward checks. We have not proved its exact numerical mechanism. The FP32-moment restore repair is covered by local checks, but this pilot shows that it did not resolve the complete TPU problem. More detailed failure diagnostics were added locally before the stop request and remain unvalidated.

## Investigation: what the records actually establish

The final pilot separates several questions that were previously being mixed together:

1. **Was the starting checkpoint already broken?** The restore audit checked all four replicas and found finite parameters, optimizer moments and reference weights. The final pilot started from the previously verified step-1 checkpoint, not the invalid step-2 checkpoint.
2. **Did the model generate usable actions?** All 16 recorded rollout entries had no environment error. Generated lengths ranged from 1 to 125 tokens. Lookup rollouts used 21 tokens each; one reasoning group used 2, 88, 125 and 2. These records support successful execution of those sampled actions, not universal answer correctness.
3. **Was there an RL feedback signal?** Two groups had varied rewards and two had identical rewards. Group-relative advantages therefore had a learning signal in part of the batch. Missing evaluation data or universally equal rewards do not explain this failure.
4. **Did the likelihood calculation fail first?** Every recorded policy, old-policy, reference, advantage and loss check was finite. Each replica finished four backward passes. The local gradient-norm guard did not reject the update.
5. **What happened during the distributed update?** The next recorded failure was the post-update parameter/optimizer-state guard. Every replica rejected the update. It did not advance the completed-step counter or publish a new checkpoint.

The important boundary is between local gradients and the resulting updated state. In the pinned PyTorch/XLA 2.8 implementation, `xm.optimizer_step` calls gradient reduction, then the optimizer, then synchronization when the barrier is requested. Our engine checks the local gradient norm before that combined call, but does not separately inspect the reduced gradients before AdamW runs. The logs therefore cannot distinguish a gradient-reduction problem, an optimizer calculation problem, or an issue in their compiled execution. Calling this an established AdamW bug or an established TPU hardware fault would go beyond the evidence. [PyTorch/XLA 2.8 source](https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/core/xla_model.py#L1114-L1117).

The final failure also cannot be attributed to hitting the old 512-token output cutoff: none of these 16 episodes reached it. The cutoff had been removed for this pilot. The engine still has a 2,048-token context capacity; removing an output allowance does not remove that capacity.

Several fixes helped narrower parts of the pipeline without solving the whole problem. Activation checkpointing let backward passes fit in memory. The revised loss passed finite checks on this batch. FP32 moment restoration addressed a tested precision-loss path. The post-update guard prevented another corrupted checkpoint from being accepted. Each of those is useful, but none is proof of stable end-to-end RL.

There were also operational weaknesses. Worker failures could previously be hidden while another replica waited at a collective; persistent worker error files and the supervisor made the final failure visible. The deadline watchdog initially failed its cloud inspections, then a replacement confirmed the original deadline and later observed deletion. Cleanup required direct verification after the CLI's deletion wait timed out. The resource was ultimately verified absent, rather than treating a timeout as evidence of deletion.

The investigation remains incomplete because the failing post-update log records only the aggregate validity result. It does not contain the exact affected tensor names, reduced gradients or moments before and after the update. The next diagnostic would need to capture those values, compare the update with a CPU reference, and check the reduction separately. That work was not run after the stop request.

The balanced 60-episode comparison was prepared but not completed. The saved comparison covers only two lookup and two clarification fixtures. SFT switching, the earlier evaluated RL policy and always-direct answered those four correctly. That small result does not show an RL improvement. Some baseline failures were protocol incompatibilities, which must not be presented as broad reasoning failures. Calibration, warmed request latency and a complete accuracy/compute comparison remain unfinished.

At the end of that stopped attempt, Tunix had been researched without a port. Its model and GRPO support alone did not validate our custom head, checkpoint conversion or joint probabilities. There was no Tunix training result.

## Weights and evidence retained

We have the best SFT checkpoint and an earlier verified finite RL step-1 checkpoint, with synchronized recovery states for all four replicas. These checkpoints contain trainable adapters, the decision head, mode-token embeddings and recovery state. They require the original Qwen3-1.7B base weights separately; they are not a standalone merged model or a public weight release.

[The saved-weight inventory](docs/saved-weights.json) records the exact GCS locations and identifies the invalid step-2 checkpoint. [The earlier RL report](docs/rl-restart-report.json) records the verified parameter differences. [The final pilot report](docs/rl-pilot-report.json) records the rejection, per-replica checks, memory headroom and verified deletion. Bucket access is required to retrieve the weights. Existing checkpoints were preserved.

## Spending and cleanup

The approved cap started at $50, increased to $60, and then to $70. The final pilot's conservative compute estimate was **$3.42**. Cumulative compute is estimated at **$57.11**, or **$62.11** with the full $5 storage reserve. These figures use creation through audited deletion completion, the four-chip on-demand bound and a 15% margin. They are conservative estimates, not verified bills.

The final TPU deletion completed at **3:58 p.m. Harare time on October 7**, and resource absence was verified. An independent watchdog initially had cloud-inspection failures; a replacement successfully confirmed the original deadline. The failed pilot was deleted early instead of waiting for that deadline. The unrelated Kaggle/Gemma project was not touched.

## What this leaves us with

The project is a supervised mode-switching prototype with saved weights, a narrow learned lookup-to-decision demo, real but incomplete RL experiments, and recorded engineering failures. It is suitable to describe honestly as an unfinished research project. It is not yet a validated adaptive-reasoning result, a completed Tunix integration, or evidence of a new state-of-the-art method.

The subsequently authorized recovery preserves those acceptance requirements: numerical correctness and checkpoint recovery must precede paid training, followed by a balanced held-out comparison. Its implementation progress is recorded separately, rather than rewriting the historical failures as successes.

## Recovery implementation addendum

The user authorized a $120 cumulative ceiling, including a conservative $62.11 prior estimate. The new allocations are $6 pilot, $8 corrective SFT, $20 RL, $16 evaluation and $7.89 storage/cleanup/contingency. Billing remains unreconciled. No new TPU compute has been incurred by the recovery work so far.

The best-SFT step-150 source has been downloaded and all eight files verified. Its portable conversion retains 224 adapter tensors, the complete existing head, and tied mode rows with IDs 151669/151670/151671. Conversion does not by itself establish model parity or authorize a launch.

The data repair produced 1,536 corrective training episodes, 120 validation episodes and 600 sealed test episodes. It grouped paired context variants, separated template families, removed forced routing instructions, randomized candidates, validated advertised clarification fields, and added typed defer/ask/lookup actions. The provenance inventory found 60,025 records across 124 raw files, overlapping phase files, and 33,411 distinct normalized problems. It reused 252 verified public-gold examples; private completions imported: zero. [The versioned manifest](docs/recovery-data-manifest.json) records checksums and source licenses.

The code now includes a pinned Tunix Qwen wrapper, the preserved JEV head, FP32 mixed-action GRPO, explicit reduction/update checks, Orbax/GCS recovery, a gated experiment worker, native baselines, sealed comparisons and labeled replay/live demos. The old paid PyTorch paths are disabled. These implementation statements do not mean every production path has passed execution tests.

The current reference/control suite passes 100 tests. Separate development checks passed 12 small-model numerical checks, including a 100-update PyTorch/Optax comparison and deterministic checkpoint continuation. Eight protocol checks cover cache likelihood equality/reuse, packed isolation/masking, four-CPU gradient averaging, three mixed-action diagnostic updates, BF16 with rematerialization, an actual distributed supervised update and per-episode context exhaustion. The diagnostic actions are not learned rollouts or an RL capability result. Complete hash-locked Linux acceptance now passes 12 numerical and ten protocol checks, including the sampling and on-device state-reuse repairs. Full 1.7B probability parity passed the unchanged 0.001 tolerance (maximum language difference 0.00003952; candidate difference 0.000000477). The paid TPU pilot and all new training/evaluation outcomes remain pending. The [recovery validation status](docs/recovery-validation.json) records these boundaries.

Local preparation encountered slow public dependency/base-weight downloads; verified local caches resolved that bottleneck. Base weights are accepted only against the pinned Hugging Face revision's SHA-256 values, even when using a public alternate transport. Missing, failed, development-only or stale acceptance reports refuse paid launch. No model-size, tolerance or budget change is used to bypass those gates.

## Evidence used for this postmortem

- [SFT lookup demonstration](docs/demo-record.json): sanitized actions and the actual saved answer.
- [Saved-weight inventory](docs/saved-weights.json): retained checkpoint locations and the invalid checkpoint warning.
- [Finite step-1 RL report](docs/rl-restart-report.json): checksum-verified parameter-change evidence.
- [Final pilot report](docs/rl-pilot-report.json): per-replica outcomes, cost bounds and deletion time.
- [Validation record](docs/jev-validation.json): confirmed local coverage and remaining limits.
- Local files under `runs/20261007-132053/`: restore audit, per-rank phase logs, numerical checks, memory records, worker tracebacks, cloud-session accounting and deletion-operation metadata. These operational records were inspected; no new training or tests were executed for this investigation.
