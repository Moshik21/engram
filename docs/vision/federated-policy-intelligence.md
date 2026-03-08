# Federated Policy Intelligence

Most people hear "federated learning" and think:

Train one big global model without directly centralizing raw data.

That is already better than shipping every private interaction into a central
database, but it still points in the wrong direction for memory systems.

For Engram, the real opportunity is not centralized training on private
memories. It is coordinated learning about memory policy.

That distinction matters.

## The Wrong Model

The wrong long-term design for AI memory looks like this:

1. many private memory systems collect user data
2. those systems send gradients, embeddings, or opaque updates upward
3. a central trainer produces a stronger shared model
4. the model is pushed back down to everyone

That approach has three problems.

First, it is hard to reason about privacy. Even if raw text is not exported,
the training signal can still encode more than people expect.

Second, it is hard to reason about control. If the memory behavior gets worse,
it is not obvious why.

Third, it is a poor fit for Engram specifically. Engram is not just a black-box
model. It is an instrumented memory system with explicit policies:

- triage
- cue promotion
- recall arbitration
- consolidation thresholds
- forgetting and pruning

That means Engram should learn at the policy layer first.

## The Better Model

Each Engram instance should remain a sovereign private brain.

That brain should be able to measure how its memory policies perform:

- what kinds of cues get used
- what kinds of recall are noisy
- which triage thresholds waste work
- which consolidation settings create regret
- how often continuity actually improves

Only privacy-bounded aggregate telemetry should leave the brain.

A coordinator should use that telemetry to produce signed policy packs:

- better priors
- better thresholds
- better calibration defaults
- better evaluation batteries

Then each local brain should test those candidate packs in shadow mode and
adopt them only if they improve local performance.

So the global system is not learning "your memories."
It is learning "which memory strategies work better."

## Telemetry Capsules, Not Memory Export

The atomic export unit should be a telemetry capsule.

A capsule contains sufficient statistics and evaluation summaries, not memory
content.

Examples:

- surfaced -> used conversion curves
- triage calibration histograms
- consolidation merge/prune ratios
- schema support sketches
- graph physiology summaries
- recall benchmark outcomes

What should not leave the brain:

- raw text
- raw entities
- readable schema names
- raw embeddings
- relationship triples

This keeps the exported layer about memory behavior, not personal content.

## Constitutional Memory Packs

The most interesting part of this idea is that the returned policy pack should
be explicit and interpretable.

Think of it as a memory constitution.

A constitutional memory pack declares how a brain should behave:

- what deserves immediate projection
- what can remain latent until proven useful
- how aggressively recall should fire
- how cautious merge/infer should be
- how conservative pruning should remain

That is much better than a hidden global update because it is:

- inspectable
- rollbackable
- testable
- overridable

Operators can pin it.
Brains can reject it.
Different brain types can blend different packs.

## Archetypes Instead Of One Global Average Brain

Another reason this approach is promising is that there is probably no single
best memory policy.

A code-heavy engineering brain may need different defaults from a research brain
or a life-management assistant brain.

So federation should not aim for one universal memory model.

It should learn archetype priors from telemetry:

- engineering-heavy
- research-heavy
- mixed work/personal
- assistant-heavy

Those priors can improve cold-start behavior while still allowing local
adaptation.

## Challenge Packs And Counterfactual Evaluation

The final piece is evaluation.

A federated memory system should not ask local brains to trust new policies on
faith. It should send challenge packs: local evaluation batteries that test
whether a candidate memory constitution actually improves things like:

- false recall
- continuity
- cue-to-projection yield
- stale-memory correction
- consolidation regret

This turns Engram into something more interesting than a storage layer or a
retrieval engine.

It becomes a policy lab for memory systems.

Many private brains can help discover better memory behavior, while each brain
remains private and sovereign.

That is the direction worth pursuing.
