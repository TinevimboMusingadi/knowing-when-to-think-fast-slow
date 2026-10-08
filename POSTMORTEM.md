# Knowing When to Switch: end-of-run postmortem

Updated October 8, 2026. This postmortem preserves the stopped PyTorch/XLA attempt. Its TPU was verified deleted. The later authorized Tunix recovery, including its first paid pilot, is recorded separately below.

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

The user authorized a $120 cumulative ceiling, including a conservative $62.11 prior estimate. The new allocations are $6 pilot, $8 corrective SFT, $20 RL, $16 evaluation and $7.89 storage/cleanup/contingency. Billing remains unreconciled. The first recovery attempt used a conservative $4.13, taking the cumulative bound to $66.24 and leaving $1.87 of the pilot allocation. Evaluation funds remain reserved.

The best-SFT step-150 source has been downloaded and all eight files verified. Its portable conversion retains 224 adapter tensors, the complete existing head, and tied mode rows with IDs 151669/151670/151671. Conversion does not by itself establish model parity or authorize a launch.

The data repair produced 1,536 corrective training episodes, 120 validation episodes and 600 sealed test episodes. It grouped paired context variants, separated template families, removed forced routing instructions, randomized candidates, validated advertised clarification fields, and added typed defer/ask/lookup actions. The provenance inventory found 60,025 records across 124 raw files, overlapping phase files, and 33,411 distinct normalized problems. It reused 252 verified public-gold examples; private completions imported: zero. [The versioned manifest](docs/recovery-data-manifest.json) records checksums and source licenses.

The code now includes a pinned Tunix Qwen wrapper, the preserved JEV head, FP32 mixed-action GRPO, explicit reduction/update checks, Orbax/GCS recovery, a gated experiment worker, native baselines, sealed comparisons and labeled replay/live demos. The old paid PyTorch paths are disabled. These implementation statements do not mean every production path has passed execution tests.

The reference/control suite now passes 103 tests. Separate development checks passed 12 small-model numerical checks, including a 100-update PyTorch/Optax comparison and deterministic checkpoint continuation. Eight protocol checks cover cache likelihood equality/reuse, packed isolation/masking, four-CPU gradient averaging, three mixed-action diagnostic updates, BF16 with rematerialization, an actual distributed supervised update and per-episode context exhaustion. The diagnostic actions are not learned rollouts or an RL capability result. The first complete hash-locked Linux acceptance passed 12 numerical and ten protocol checks, including sampling and on-device state reuse. Full 1.7B probability parity passed the unchanged 0.001 tolerance (maximum language difference 0.00003952; candidate difference 0.000000477). On October 8, changed compiler/dispatch code passed fresh locked acceptance: 13 numerical checks, ten protocol checks and full-model probability parity. The [recovery validation status](docs/recovery-validation.json) records these boundaries.

Local preparation encountered slow public dependency/base-weight downloads; verified local caches resolved that bottleneck. Base weights are accepted only against the pinned Hugging Face revision's SHA-256 values, even when using a public alternate transport. Missing, failed, development-only or stale acceptance reports refuse paid launch. No model-size, tolerance or budget change is used to bypass those gates.

### First Tunix TPU pilot: finite updates, incomplete compilation

Run `20261007-220801` restored SFT step 150 on the approved interruptible four-chip slice. Its actual TPU numerical suite passed all 12 checks in 36.0 seconds, including 100 small-model optimizer updates against PyTorch and checkpoint continuation. [The preserved numerical report](docs/recovery-tpu-numerical.json) was downloaded and uploaded with a verified checksum before the later stall.

At supervised microbatch 1, three disposable full-model updates completed. Adapters, the JEV head and mode rows all changed. The first step took 814.5 seconds; warmed steps took 81.6 and 83.2 seconds. Minimum measured device-memory headroom was 39.8%. The repeated diagnostic training batch's final loss was 0.6521. This is an execution and update check, not corrective SFT, a selected checkpoint or held-out improvement.

[The diagnostic figure](docs/figures/recovery-pilot.png) can be regenerated from the preserved partial record with `python docs/render-pilot-figure.py` in an environment containing Matplotlib. It distinguishes the cold step from warmed execution and makes no held-out performance claim.

Compilation at microbatch 2 became unresponsive to repeated remote health checks. Before access failed, the observed process RSS had grown to roughly 145 GiB on a host with roughly 189 GiB of RAM. That supports investigating compiler workload and host-memory pressure. It does not prove an OOM, a hardware fault or a non-finite update. The operator deleted only this owned TPU; absence was verified. The complete guest logs and final artifact export could not be recovered. The [partial pilot record](docs/recovery-pilot-report.json) explicitly identifies the full-model numbers as live operator observations rather than a completed guest report.

Code inspection found that candidate encoding expanded a complete backbone computation for each candidate. The repair uses a bounded, rematerialized loop and compiles gradient accumulation and optimizer-state checks together, preserving separate numerical acceptance boundaries. Its new small-model candidate-gradient comparison passes the original fixed tolerance. Distributed protocol and full-model probability checks must pass again before the changed code can launch. No real full-model mixed-action GRPO update completed in this recovery attempt.

The cloud connection gate also needed repair: a Windows SSH wrapper could return success after abandoning a host-key prompt. Launch now requires an exact response to a fresh remote nonce, pins the public host key for the owned endpoint, and rechecks ownership after connecting. It does not modify global SSH settings. Three control tests cover the false-success path and verification helpers. This fixes a misleading connection status; it does not explain the later compiler stall.

The previous measured cold step, two warmed steps and five minutes of cleanup alone reserve $1.96 at the conservative hourly bound, excluding setup and the required mixed-action pilot. That exceeded the $1.87 then left in the pilot allocation. On October 8, the user approved moving $5 from contingency to the pilot, leaving $6.87 for another attempt. The total cap remains $120, with the $16 evaluation reserve intact. A retry still requires fresh acceptance and measured cost projections; faster TPU compilation is not assumed.

### Local reference recovery on October 8

Two owned local containers exhausted their six-GiB memory allowance. Instrumentation located the second failure during full-backbone loading, before any probability comparison or RL update. A subsequent run copied the same verified base shards to Linux-local storage and permitted swap within the owned container. Full FP32 language and JEV probability parity then passed in 306.5 seconds, with maximum absolute differences of 0.00003952 and 0.000000477 respectively. The tolerance remains 0.001. Storage placement and swap changed together, so this result does not isolate the source of the earlier loading peak.

The reference harness records execution stages, disables asynchronous CPU dispatch and releases language-forward intermediates before candidate comparison. It interprets the candidate scan body for this memory-limited full-model reference. Compiled candidate forward/gradient behavior is checked separately on the small model and remains subject to the TPU pilot. This local pass establishes conversion parity; it is not training or a throughput result.

### Connection-gate attempt on October 8

Run `20261008-110948` passed local launch gates and created the approved interruptible slice. Its Ed25519-only public host-key scan returned no accepted key, so remote verification refused to release training. The scan's stderr was not retained; boot readiness, supported key types and network reachability cannot be distinguished from that record. No training began and no update was accepted.

Deletion commands timed out while Google reported `DELETING`. The owned deletion operation subsequently completed at 11:27:16 UTC, and fresh API and CLI checks confirmed absence. The attempt adds a conservative $1.50 from the pre-upload start to the API deletion completion, using the whole-slice bound plus 15%. The cumulative bound is $67.75, with $5.36 left for the pilot. This remains an estimate rather than reconciled billing. [The connection-attempt record](docs/recovery-connection-attempt-20261008.json) preserves the operation evidence and uncertainty.

The connection repair probes supported Ed25519, ECDSA and RSA host keys in deterministic preference order, permits up to 90 seconds for SSH readiness, and rejects ambiguous or changed keys. The exact remote nonce and resource-ownership checks remain required. Failure reports now replace any old success record. Cleanup observes a pending deletion rather than repeatedly reissuing it. Fourteen targeted control tests pass; fresh locked acceptance and an actual successful connection remain required before further training.

The next attempt, `20261008-120500`, established the connection failure's cause. The native Windows scanner reached the server but repeatedly rejected `sntrup761x25519-sha512@openssh.com` key exchange. Git's installed scanner immediately obtained the same owned endpoint's Ed25519 key; ownership and endpoint identity were verified before and after that probe. The client now selects that installed scanner without modifying global SSH configuration. Sixteen connection-control tests pass. The [second connection record](docs/recovery-connection-attempt-20261008-120500.json) preserves both clients' evidence. No training began, and deletion was subsequently verified. The attempt adds $1.38 to the conservative bound, bringing it to $69.12 and leaving $3.99 for the pilot.

The pilot now checks microbatch 1 and three complete sampled mixed-action updates, including save/reload continuation, before optional larger-batch tuning. Larger probes need a measured cost projection that includes ten minutes of cleanup reserve. Skipped settings are reported as untested rather than stable. This preserves the required learning checks while making throughput optimization conditional on the remaining allowance. A locked-backend orchestration test verifies that an unaffordable larger probe cannot displace the core checks. Full-model parity, 13 numerical checks and ten protocol checks passed again with this code. The Windows suite passed 111 tests and skipped this one backend-dependent test; the skipped test passed in the locked Linux environment. Actual TPU mixed-action acceptance remains pending.

## Evidence used for this postmortem

The next launch failed while uploading its 35 MB payload, before any TPU provisioning command. API and CLI checks confirmed the resource absent. The CLI's orphan workers held subprocess pipes after the 600-second timeout; terminating only the verified upload descendants allowed cleanup to finish. Sequential CLI execution and API transfers with larger request bodies also timed out. Windows and Linux both reported socket write timeouts, so the underlying network slowdown remains unexplained.

A 256 KiB resumable transfer completed in 619.6 seconds with transport checksum verification. The production uploader independently verified the completed file's CRC32C and successfully created a new small file. It now rejects existing objects with mismatched checksums and never overwrites them. Payload transfer finishes before the accelerator cost clock starts; the worker reloads the finalized session and ledger after the remote-access gate. Twenty-one targeted controls pass, and the full Windows suite passes 116 tests with one backend-dependent skip. Fresh locked acceptance passed six transfer/orchestration tests, full-model parity, 13 numerical checks and ten protocol checks. The [sanitized upload record](docs/recovery-upload-attempt-20261008.json) distinguishes this transport failure from model training. The conservative ledger retains the failed pre-provisioning setup charge, bringing its bound to $70.38; this is not an actual bill. No new model training began. The remaining pilot allowance is insufficient for the complete required checks, and paid execution awaits the allocation decision.

- [SFT lookup demonstration](docs/demo-record.json): sanitized actions and the actual saved answer.
- [Saved-weight inventory](docs/saved-weights.json): retained checkpoint locations and the invalid checkpoint warning.
- [Finite step-1 RL report](docs/rl-restart-report.json): checksum-verified parameter-change evidence.
- [Final pilot report](docs/rl-pilot-report.json): per-replica outcomes, cost bounds and deletion time.
- [Validation record](docs/jev-validation.json): confirmed local coverage and remaining limits.
- Local files under `runs/20261007-132053/`: restore audit, per-rank phase logs, numerical checks, memory records, worker tracebacks, cloud-session accounting and deletion-operation metadata. These operational records were inspected; no new training or tests were executed for this investigation.
