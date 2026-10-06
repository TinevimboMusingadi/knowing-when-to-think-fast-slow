# Knowing When to Switch

Fast decisions and careful reasoning are useful at different moments. This
project asks whether a small language model can learn when to move between them
while sharing one backbone, rather than requiring separate deployed models.

The prototype extends Qwen3-1.7B with a typed decision head. Three learned tokens
select direct answering, ordinary chain-of-thought, or candidate scoring. The
runtime executes the selected path. A mode token is a control signal, not proof
that the model has learned good judgment.

Each candidate is encoded with the state in an isolated branch. A shared
attention head compares the branch representations without candidate-position
embeddings. The design targets permutation-equivariant probabilities: moving a
candidate should move its probability with it rather than change its meaning.
Routing and prefill costs still count toward inference latency.

Training starts with supervised examples, then group-relative reinforcement
learning. Rewards depend on verified answers, necessary information acquisition,
and compute costs. A plausible reasoning trace is not itself evidence of a
correct answer. Missing information requires asking or using a controlled lookup.

The public tasks are arithmetic and context fixtures, with source-separated
holdouts. This limits the claims: results here will not establish broad-domain
reasoning, robust web research, or production deployment readiness.

## Results pending

This is an article draft, not a completed experimental report. Replace this
section only with measured SFT/RL comparisons, resource usage, errors, and
confidence intervals from recorded runs. Include actual cloud costs and clearly
distinguish initialization, pilot tests, and completed training.

## Prior work

Qwen3 already supports thinking and non-thinking operation. AdaptThink and PATS
study adaptive reasoning. The narrower question here concerns learned transitions
between typed candidate scoring, ordinary generation, and gathering missing
context. The experiment must earn its claims through comparisons.
