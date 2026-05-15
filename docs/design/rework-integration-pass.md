# Rework Integration Pass — Extraction, Recall, and Consolidation

## Status: Integration Checklist

This document is the integration pass for the three reworks:

- [extraction-rework.md](./extraction-rework.md)
- [recall-rework.md](./recall-rework.md)
- [consolidation-rework.md](./consolidation-rework.md)

The architecture is directionally correct. The remaining work is mostly in the
handoffs between systems:

- cue creation and cue lifecycle
- recall interaction semantics
- promotion from latent memory to projection
- lite/full store parity
- end-to-end verification across all three flows

This is not a redesign document. It is a delivery checklist for making the
three reworks behave as one coherent memory loop.

## Goal

Ship a version of Engram where:

1. `observe()` produces usable latent memory immediately.
2. recall can surface that latent memory without reinforcing it as if it were
   actually used.
3. cue hits and usage feedback can promote the right episodes into projection.
4. projected graph facts flow into consolidation with the same semantics as
   ingestion and replay.
5. SQLite and FalkorDB/Redis expose the same user-visible behavior for the
   reworked paths.

## Out Of Scope

- redesigning the extraction model
- redesigning ACT-R retrieval scoring
- changing the consolidation phase order
- replacing the current recall planner with a new architecture

## Integration Definition Of Done

The integration pass is complete when all of the following are true:

- auto-surfaced recall does not materially reinforce ranking or activation
  state
- explicit recall and confirmed use still reinforce memory normally
- cue-backed results receive the same post-response feedback loop as entity
  results
- worker batching does not leave stale or duplicate cue state behind
- cue promotion can reliably schedule projection and the worker processes it
- raw episode recall works in both lite and full mode
- replay, ingestion, and cue-prompted projection all apply relationship
  semantics through the same path
- the end-to-end integration suite passes in lite mode and parity checks pass
  against full mode abstractions

## System Loop To Validate

```text
store_episode
  -> build_episode_cue
  -> cue search / cue packets / cue feedback
  -> projection scheduling
  -> project_episode
  -> shared apply path
  -> graph + episode indexing
  -> recall packets + usage feedback
  -> consolidation phases
```

## Deliverables

### 1. Recall Interaction Semantics Hardening

Problem:

- auto-surfaced recall still mutates retrieval-learning state
- the recall rework's surfaced-vs-used contract is not fully enforced

Deliverables:

- [x] gate Thompson Sampling feedback by interaction type or retrieval mode
- [x] ensure `surfaced` and `selected` do not count as true usage
- [x] ensure `used` and `confirmed` still reinforce normally
- [x] ensure `dismissed` and `corrected` apply neutral/negative feedback only
- [x] document the final interaction matrix in README and recall docs

Acceptance criteria:

- `surfaced` recall leaves `access_count` unchanged
- `surfaced` recall leaves TS state unchanged
- `selected` recall leaves true access unchanged
- `used` increments access as before
- `corrected` increases negative feedback without recording access

Suggested test coverage:

- `manager.recall(... interaction_type="surfaced")`
- chat tool path: `selected` then response-derived `used`
- explicit recall path: direct `used`

### 2. Cue Lifecycle Correctness

Problem:

- worker batching can merge episode content without regenerating the surviving
  cue
- merged-away episodes can retain live cue records and duplicate latent memory

Deliverables:

- [x] regenerate the primary cue after adjacent-turn batch merge
- [x] retire or suppress cues for merged-away episodes
- [x] make merged episode state explicit in episode metadata or cue metadata
- [x] ensure cue search cannot return latent duplicates for merged-away turns

Acceptance criteria:

- after batching, the surviving episode's cue matches merged content
- merged-away episodes do not show up as active cue-backed recall results
- projection state and cue state remain aligned after merge

Suggested test coverage:

- batch merge of `auto:prompt` + `auto:response`
- cue search after merge returns one surviving latent memory
- scheduled projection after merge uses the regenerated cue

### 3. Cue Feedback Closure

Problem:

- cue-backed results participate in retrieval, but post-response feedback
  mostly upgrades entity results only
- the cue policy loop is therefore incomplete

Deliverables:

- [x] extend usage detection to include `cue_episode` results
- [x] propagate `selected`, `used`, `dismissed`, and `near_miss` back into cue
  policy updates
- [x] ensure chat/tool recall can promote cue-backed episodes from real usage
- [x] ensure cue packet assembly and cue policy learning agree on IDs and
  provenance

Acceptance criteria:

- a cue result selected by chat recall can later be marked `used`
- dismissed cue results do not promote projection
- repeated cue usage can schedule projection without manual intervention

Suggested test coverage:

- cue result returned by chat recall
- assistant response mentions cue content
- cue policy score increases and projection gets scheduled

### 4. Retrieval Metadata And Follow-On Behavior

Problem:

- entity results are missing explicit `result_type="entity"`
- downstream logic that depends on entity-only filtering can silently miss
  them

Deliverables:

- [x] mark entity recall results explicitly as `result_type="entity"`
- [x] verify retrieval priming sees normal entity results
- [x] audit packet assembly and UI/tool formatting for assumptions around
  result typing

Acceptance criteria:

- entity results are consistently typed on all recall surfaces
- retrieval priming buffer populates from top entity hits when enabled
- packet assembly behavior is unchanged or improved

Suggested test coverage:

- explicit recall with `retrieval_priming_enabled=True`
- MCP recall formatting
- API recall formatting

### 5. Full-Mode Recall Parity

Problem:

- lite mode supports raw episode recall
- full mode indexes episodes but does not expose the same raw episode recall
  surface

Deliverables:

- [x] implement `search_episodes()` for Redis search
- [x] verify cue-backed search and raw episode search coexist correctly in full
  mode
- [x] add parity coverage for lite vs full recall result shapes
- [x] fail loudly when a required recall capability is missing from a backend

Acceptance criteria:

- raw episode packets can be produced in full mode
- cue packets and episode packets both work in full mode
- recall pipeline feature coverage matches lite mode for the reworked paths

Suggested test coverage:

- protocol-level parity tests
- Redis search unit tests for episode lookup
- recall pipeline tests with a full-mode search double

### 6. Profile And Rollout Wiring

Problem:

- the three reworks are individually implemented, but their feature flags are
  not automatically enabled together
- users can enable recall waves without enabling the cue layer those waves are
  meant to exploit

Deliverables:

- [x] define the intended integrated rollout profile
- [x] either add a new preset or document the exact flag bundle required
- [x] ensure README examples use a coherent integration configuration
- [x] document which combinations are partial rollouts vs full integration

Minimum integrated flag set:

- `cue_layer_enabled`
- `cue_recall_enabled`
- `cue_policy_learning_enabled`
- `projector_v2_enabled`
- `projection_planner_enabled`
- `targeted_projection_enabled`
- `recall_need_analyzer_enabled`
- `recall_planner_enabled`
- `recall_usage_feedback_enabled`
- `memory_maturation_enabled`
- `episode_transition_enabled`

Acceptance criteria:

- a documented config exists for "full rework integration"
- README and design docs no longer imply the loop is active when only a subset
  of flags is enabled

### 7. End-To-End Integration Tests

Problem:

- the subsystem tests are strong, but most cross-flow contracts are not
  asserted directly

Deliverables:

- [x] add a lite-mode end-to-end integration suite for the full loop
- [x] add full-mode parity tests at the protocol/search abstraction layer
- [x] add negative tests for surfaced-vs-used semantics
- [x] add tests for batching, cue promotion, and worker scheduling

Required scenarios:

- [x] `observe` -> cue created -> cue recall -> no reinforcement on `surfaced`
- [x] cue result marked `used` -> projection scheduled -> worker projects
- [x] projected episode becomes recallable as graph/entity memory
- [x] consolidation sees projected episode state and can semanticize it
- [x] replay uses the same relationship semantics as ingestion
- [x] batching two auto-turns does not produce duplicate cue recall results
- [x] lite and full mode both return episode packets for the same scenario

### 8. Docs And Operator Visibility

Problem:

- the codebase now has three sophisticated flows, but operators still need a
  simple way to understand what state an episode is in and why

Deliverables:

- [x] add an episode/cue state table to README
- [x] document how cue promotion works
- [x] document the surfaced/selected/used lifecycle
- [x] expose enough status in APIs/UI to debug integration issues

Recommended operator-visible fields:

- episode `status`
- episode `projection_state`
- episode `last_projection_reason`
- cue `hit_count`
- cue `policy_score`
- cue `projection_state`
- cue `last_feedback_at`

## Execution Order

Recommended order:

1. Recall semantics hardening
2. Cue lifecycle correctness
3. Cue feedback closure
4. Retrieval metadata correctness
5. Full-mode recall parity
6. Profile wiring
7. End-to-end tests
8. Docs cleanup

This order fixes the correctness risks first, then closes the feedback loop,
then locks in parity and rollout.

## Exit Review Checklist

- [x] all P1 integration bugs from review are closed
- [x] all new integration tests pass in lite mode
- [x] full-mode recall parity checks pass
- [x] README reflects actual rollout state
- [x] the three rework docs can be read as one coherent system without hidden
  caveats

## Notes

The current state should be interpreted as:

- extraction rework: integrated into the cue/projection loop
- recall rework: integrated with surfaced-vs-used semantics and cue feedback closure
- consolidation rework: integrated with projected-episode state and replay parity

The integration pass closed the cross-flow handoffs that previously made the
three reworks feel separate.
