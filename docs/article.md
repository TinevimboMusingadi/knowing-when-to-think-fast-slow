# Knowing When to Switch

Choosing a known value from a few options should require less work than solving
a multi-step problem. When a necessary fact is missing, neither a quick guess
nor a long explanation solves the problem. The useful next step is to ask for
that fact.

Knowing When to Switch asks whether one small model can learn those choices:
when to make a typed decision, when to answer directly, when to reason, and when
to acquire more context before deciding. The last item is an action within a
mode, rather than a fourth mode.

The prototype extends [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) with a typed decision head. Three learned tokens
select direct answering, ordinary chain-of-thought, or candidate scoring. The
runtime executes the selected path. A mode token is a control signal, not proof
that the model has learned good judgment.

Each candidate is encoded with the state in an isolated branch. A shared
attention head compares the branch representations without candidate-position
embeddings. The design targets permutation-equivariant probabilities: moving a
candidate should move its probability with it rather than change its meaning.
Routing and prefill costs still count toward inference latency. Scoring four
candidates means encoding four branches. Calling that path "fast" is a hypothesis
to test with synchronized timing, rather than a conclusion from its short output.

For a typed decision, emitting `<mode:jev>` stops language decoding immediately
and invokes the scoring head. `<mode:direct>` selects ordinary answering or a
context request. `<mode:cot>` selects ordinary reasoning. The same backbone and
LoRA adapters serve all three paths; the model keeps its original language head.
Only adapters, the small decision head, and the new token rows are trainable.

A typed assessment can continue into reasoning, and reasoning can finish in a
typed decision. Another intended trajectory asks for an unknown record value,
receives it, and selects the matching candidate. Controller tests exercise these
trajectories with supplied actions. Learned demonstrations must come from saved
model rollouts; passing the controller tests does not establish learned routing.

Training starts with supervised examples, then group-relative reinforcement
learning. Rewards depend on verified answers, necessary information acquisition,
and compute costs. A plausible reasoning trace is not itself evidence of a
correct answer. Missing information requires asking or using a controlled lookup.

The dataset contains 8,000 training episodes and two holdouts of 600 episodes
each. Four thousand existing math prompts were reused after checking their answers
against the original GSM8K training source. Existing teacher completions were
not imported. Problem duplicates and overlapping source families are checked
before upload. The typed tasks include choices, yes/no/unknown, and bounded
rubric scores. Separate examples cover reasoning, transitions, clarification,
and controlled lookup.

The public tasks are arithmetic and context fixtures, with source-separated
holdouts. They cannot establish broad-domain reasoning or reliable open-web
research. A benchmark may also resemble the base model's pretraining material:
split isolation in this experiment does not prove absence of pretraining exposure.

The TPU run uses a four-chip preemptible v5e slice and BF16. Its spending ceiling
is $50, including storage and evaluation. The guard uses the four-chip on-demand
rate as a conservative bound, adds a margin, and carries failed attempts into
the next run. [Google publishes TPU prices per chip-hour](https://cloud.google.com/tpu/pricing),
so using a one-chip price for the whole slice would undercount the experiment.

Early runs found deployment and recovery bugs before they could become claims
about learning. A larger pilot batch exceeded TPU memory. A memory-counter
compatibility bug incorrectly rejected the smaller batch. An early SFT cost
projection spread startup compilation across every future step. XLA checkpoint
serialization also converted a random-state tuple to a list. Each failure left
logs or recoverable checkpoints, and the corresponding fixes received tests.

An early saved Qwen SFT checkpoint, from before the final dataset audit, contains two completed optimizer updates across
four replicas. Its checksums were verified after downloading it from GCS. All
112 LoRA B matrices, initialized at zero, contain nonzero values. The three mode
rows have diverged from their identical initialization, the head has nonzero
optimizer moments, and the saved trainable parameters are finite. The
[public parameter evidence](parameter-update-evidence.json) records these checks.
This establishes an operational training path, not improved task accuracy.

## What the run established

This article remains a draft because the full experiment is incomplete. Supervised
training completed 438 optimizer updates across four TPU replicas on the corrected
dataset. The recorded training losses and gradient norms stayed finite. This
does not establish held-out accuracy: those losses concern supervised training,
and the best checkpoint was selected by validation loss. An initial autoregressive
evaluation produced no first rollout result after several minutes. The pipeline
was changed to use fixed-size KV caches for TPU rollouts, with CPU equivalence
tests, and real GRPO was launched from the saved best SFT checkpoint. A mixed-head
distributed gradient synchronization fault interrupted progress; giving every
replica the same gradient list allowed the resumed run to reach step 3. The next
update was rejected for non-finite gradients. A recoverable step-3 checkpoint
was saved, and the experiment's TPU was deleted after the comparison stage.

The saved comparison records contain two lookup and two clarification episodes.
SFT switching, the selected step-1 RL checkpoint, and always-direct each answered
those four fixtures correctly with grounded context. These records neither
establish an RL improvement nor compare all six task behaviors. Native Qwen has
no applicable tasks in that four-episode sample, and produced no evaluated
records before its next batch timed out.

One actual saved lookup example asks for an undisclosed record value. The model
selects direct mode to request `record-test-36`, receives its value, and emits
JEV to score the candidates. It returns 1,912 using 19 generated tokens. This is
a narrow learned context-acquisition example; it does not demonstrate a general
ability to choose between fast and slow reasoning. JEV-to-CoT and CoT-to-JEV
transitions still have controller-test coverage rather than a completed learned
benchmark.

Offline repairs reproduced overflow in the old exponential KL calculation. The
new loss computes in FP32, uses `expm1`, and bounds exponential tails at a
log-ratio magnitude of 20. That tail bound changes the extreme-tail objective,
so its activation counts are logged. A random tiny Qwen architecture completed
four finite fixed-action optimizer updates, changing adapters, decision-head
weights, and mode rows while preserving base weights. This regression probe
does not prove that the original 1.7B TPU failure is resolved: the exact failing
rollout was not saved. New runs now retain sampled traces and numerical diagnostics.

A reproducible 60-episode holdout contains ten episodes from each behavior,
with a separate 30-episode validation subset for choosing the confidence
threshold. It is prepared, not evaluated. The full 1.7B weights are not cached
locally, and no new paid resource was created for the repair. A complete
comparison and any accuracy/compute improvement remain unestablished.
The final report must distinguish warmed
inference from compilation, include failures, and report billing separately from
conservative spending estimates. There is no accuracy or efficiency gain to claim
from training updates alone.

## Prior work

Qwen3 already supports thinking and non-thinking operation.
[AdaptThink](https://arxiv.org/abs/2505.13417) learns adaptive thinking choices
with reinforcement learning. [PATS](https://arxiv.org/abs/2505.19250) studies
switching during the reasoning process using process rewards and search.
[BEV's Qwen decision architecture](https://github.com/avbiswas/bev-train) provides
another relevant starting point for typed, choice-order-invariant decisions.
The narrower question here concerns learned transitions
between typed candidate scoring, ordinary generation, and gathering missing
context. The experiment must earn its claims through comparisons.

The [code and reproducible workflow](https://github.com/TinevimboMusingadi/knowing-when-to-think-fast-slow)
are public. The useful research outcome will be a measured answer to whether
these execution paths share a better accuracy/compute tradeoff on the controlled
tasks, including the cases where switching makes the result worse.
