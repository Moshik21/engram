# Identifier-Aware Entity Dedup — SKU-Safe Resolution and Consolidation

> Status: implemented — entity_dedup_policy.py IDENTIFIER type + exact-identifier match


## Status: Proposed Build Path

This document proposes a targeted rework of Engram's entity dedup behavior for
identifier-like names such as SKUs, part numbers, serials, and model codes.

The current dedup stack treats these names as ordinary natural-language names.
That is the root cause of merges like:

- `1712061` <- `1712018`
- `1712515` <- `1712516`
- `1855012` <- `1855015`

Those are lexically similar strings, but they are not semantically similar in
the same way as `React` vs `React.js` or `AWS` vs `Amazon Web Services`.

For identifier-like entities, a one-character difference often means
"different thing," not "same thing with noisy formatting."

## Summary

The recommended fix is not a threshold tweak. It is a policy change.

Engram should classify entity names into lexical regimes and apply different
matching rules to each regime:

1. `natural_language`
   Use the current fuzzy/name-heuristic behavior.
2. `identifier`
   Require exact canonical identifier equivalence before resolving or merging.
3. `hybrid`
   Compare identifier chunks exactly and language chunks fuzzily.

This policy must be shared across both:

- ingestion-time entity resolution
- consolidation-time merge

If it only exists in consolidation, false merges can still happen during
`resolve_entity_fast()`. If it only exists in resolution, consolidation can
still collapse distinct identifiers later.

## Why This Needs a Shared Policy

The false-positive behavior currently exists in multiple places.

### Ingestion-time resolution

`resolve_entity_fast()` fuzzy-matches new entity candidates against session and
DB candidates using `compute_similarity()` plus a same-type boost.

Relevant code:

- [server/engram/extraction/resolver.py](/Users/konnermoshier/Engram/server/engram/extraction/resolver.py)
- [server/engram/storage/sqlite/graph.py](/Users/konnermoshier/Engram/server/engram/storage/sqlite/graph.py)
- [server/engram/storage/falkordb/graph.py](/Users/konnermoshier/Engram/server/engram/storage/falkordb/graph.py)

In SQLite mode, candidate lookup can also surface nearby code-like entities via
FTS and prefix fallback, which increases the chance of accidental code/code
comparison.

### Consolidation-time merge

The merge phase uses the same generic name similarity and same-type boost for:

- pairwise fuzzy merge
- ANN candidate acceptance
- soft-zone candidate selection
- merge record display in the dashboard

Relevant code:

- [server/engram/consolidation/phases/merge.py](/Users/konnermoshier/Engram/server/engram/consolidation/phases/merge.py)
- [server/engram/consolidation/scorers/merge_scorer.py](/Users/konnermoshier/Engram/server/engram/consolidation/scorers/merge_scorer.py)
- [dashboard/src/components/ConsolidationPanel.tsx](/Users/konnermoshier/Engram/dashboard/src/components/ConsolidationPanel.tsx)

## Current-State Findings

The current system has four important properties:

1. `compute_similarity()` is generic fuzzy matching.
   It does not distinguish names from identifiers.

2. Same-type pairs get a small positive boost.
   That is helpful for ordinary aliases, but harmful for code-like names where
   even a tiny boost can push a near-match across the merge threshold.

3. The multi-signal scorer is still vulnerable to identifier false positives.
   Distinct SKUs can have:
   - similar embeddings
   - shared neighbors
   - overlapping summaries
   - positive exclusivity signals

4. The dashboard currently displays fuzzy name similarity, not true decision
   confidence.
   That can make an identifier false positive look more principled than it is.

## Goals

1. Prevent accidental merges of distinct identifier-like entities.
2. Preserve good fuzzy dedup for natural-language aliases.
3. Apply the same protection during ingestion and consolidation.
4. Keep the MVP narrow enough to ship without a schema rewrite.
5. Produce clear audit reasons when identifier policy blocks a merge.

## Non-Goals

1. Removing numeric entities from extraction.
2. Requiring a new `entity_type` before shipping a fix.
3. Eliminating fuzzy matching for normal names.
4. Solving all product-catalog normalization in the first iteration.
5. Automatically repairing every historical merge in the first iteration.

## Core Design

### 1. Introduce lexical regime classification

Add a shared helper that classifies a name as one of:

- `natural_language`
- `identifier`
- `hybrid`

Suggested intent:

- `identifier`
  A mostly code-shaped string, or a label plus one code token.
- `hybrid`
  A natural-language phrase anchored by a specific code.
- `natural_language`
  Everything else.

Examples:

| Name | Regime | Notes |
| --- | --- | --- |
| `1712061` | `identifier` | pure numeric code |
| `SKU 1712061` | `identifier` | labeled code |
| `P/N AB-1712061-C` | `identifier` | labeled structured code |
| `RTX 4090` | `hybrid` | product family + code |
| `iPhone 15 Pro` | `natural_language` | product name with number, not just a code |
| `k8s` | `natural_language` | preserve numeronym semantics |
| `ACT-R` | `natural_language` | technical term, not part code |
| `Model AB-1712061-C bracket` | `hybrid` | language plus code anchor |

### 2. Add canonical identifier parsing

For `identifier` and `hybrid` names, build a canonical parsed form.

Suggested outputs:

- normalized text
- detected labels removed for comparison
- alpha chunks
- digit chunks
- separator-stripped code form
- regime

Important default:

- preserve leading zeros

This should treat:

- `AB-1712061-C`
- `AB 1712061 C`
- `ab1712061c`

as the same identifier structure, while still keeping:

- `AB-1712061-C`
- `AB-1712062-C`

strictly separate.

### 3. Enforce hard policy before fuzzy scoring

Before fuzzy or embedding-based acceptance:

1. If both names are `identifier` and canonical identifiers differ:
   reject immediately.
2. If both names are `identifier` and canonical identifiers match:
   allow normal alias/dedup flow.
3. If either name is `hybrid`:
   require exact agreement on detected identifier chunks before using fuzzy
   comparison on the remaining language.
4. If both names are `natural_language`:
   use the current logic.

This rule should be applied on every path:

- `resolve_entity()`
- `resolve_entity_fast()`
- consolidation pairwise fuzzy path
- consolidation ANN candidate path
- consolidation soft-zone candidate path
- multi-signal scorer

### 4. Add an identifier-aware confidence ceiling

Even after hard constraints, the scorer should avoid treating code-like names
as ordinary semantic duplicates.

Recommended behavior:

- if code chunks disagree, confidence is forced to `0.0`
- if regime is `identifier`, disable fuzzy booster rules that were designed for
  human-readable aliases
- emit a specific reason such as `identifier_mismatch`

This makes future debugging easier and prevents embeddings or shared neighbors
from "talking the system into" merging distinct part numbers.

### 5. Expose policy reasons in audit traces

Consolidation should log when a candidate was blocked because of identifier
policy, not just because of generic low confidence.

Examples:

- `identifier_mismatch`
- `hybrid_code_mismatch`
- `identifier_exact_match`

This makes the behavior explainable in dashboard detail APIs and future review
tools.

## Why Threshold Tuning Is Not Enough

Raising `consolidation_merge_threshold` is the wrong primary fix.

It would:

- reduce useful natural-language dedup
- still leave ingestion-time fuzzy resolution vulnerable
- still allow false positives when embeddings and shared context are strong
- fail unpredictably across stores and graph states

This is a semantics problem, not just a numeric threshold problem.

## Edge Cases and Handling

### Leading zeros

Default policy:

- `001234` and `1234` are distinct unless a future explicit alias rule says
  otherwise.

Reason:

- many real identifier systems treat leading zeros as significant

### Separator-only differences

These should match:

- `AB-1234-C`
- `AB 1234 C`
- `AB1234C`

Reason:

- formatting noise should not create duplicate entities

### Labeled variants

These should match:

- `1712061`
- `SKU 1712061`
- `Part #1712061`
- `P/N 1712061`

Reason:

- labels are metadata around the same underlying code

### Revision suffixes

These should remain distinct by default:

- `1712061-A`
- `1712061-B`

Reason:

- revision suffixes often designate different parts or revisions

This can be revisited later with domain-specific alias rules.

### Family names versus exact codes

These should not auto-merge:

- `RTX 4090`
- `RTX 4080`
- `iPhone 15`
- `iPhone 15 Pro`

Reason:

- these are related products, not duplicate entities

### Numeronyms and technical abbreviations

These should continue to work as today:

- `k8s` <-> `kubernetes`
- `i18n` <-> `internationalization`
- `ACT-R` <-> `ACTR`

Reason:

- these are language aliases, not product identifiers

The identifier classifier should explicitly avoid swallowing these cases.

### Years, counts, and ordinary numeric mentions

These should not be treated as identifiers just because they contain digits:

- `2024 roadmap`
- `Phase 2`
- `Top 10 ideas`

Reason:

- digits alone are not enough; the overall lexical shape matters

### OCR and transcription mistakes

Examples:

- `1712061` vs `17120G1`
- `O0X123` vs `00X123`

Default policy:

- keep separate unless exact canonical agreement is found

Reason:

- for identifiers, false merge cost is usually higher than false split cost

## Proposed Implementation Shape

### Shared helper module

Add a central helper module for name-shape analysis and policy.

Suggested responsibilities:

- classify lexical regime
- detect identifier-like labels
- canonicalize identifier-like names
- compare code chunks
- return structured policy decisions

Possible API shape:

```python
class NameRegime(Enum):
    NATURAL_LANGUAGE = "natural_language"
    IDENTIFIER = "identifier"
    HYBRID = "hybrid"


@dataclass
class IdentifierForm:
    regime: NameRegime
    normalized: str
    canonical_code: str | None
    alpha_chunks: tuple[str, ...]
    digit_chunks: tuple[str, ...]


@dataclass
class DedupPolicyDecision:
    allowed: bool
    reason: str
    exact_identifier_match: bool = False


def analyze_name(name: str) -> IdentifierForm: ...
def dedup_policy(name_a: str, name_b: str) -> DedupPolicyDecision: ...
```

### Resolver integration

Use the policy before scoring candidates in:

- `resolve_entity()`
- `resolve_entity_fast()`

If policy says `allowed=False`, skip the candidate before fuzzy scoring.

### Merge integration

Use the same policy in:

- `_compare_block()`
- ANN candidate acceptance
- `_collect_soft_zone_pairs()`
- `score_merge_pair()`
- final merge record reasoning

### Candidate retrieval hardening

For identifier-like queries, candidate retrieval can be made more precise.

SQLite:

- prefer exact normalized match first
- avoid broad prefix fallback for strict identifier regimes

FalkorDB:

- avoid broad `CONTAINS` retrieval for strict identifiers when a more exact
  lookup path is possible

This is an enhancement, not the core safety mechanism. The core safety
mechanism remains policy rejection before dedup.

## Alternative Solutions Considered

### Option A: Raise merge thresholds

Rejected as primary solution.

This would suppress legitimate natural-language dedup and still miss
ingestion-time failures.

### Option B: Add a new entity type first

Useful long-term, not required for MVP.

The extractor does not currently emit a dedicated identifier/product-part type,
so blocking on schema expansion would slow down the fix unnecessarily.

### Option C: Disable merging for numeric names entirely

Too blunt.

It would fail to merge equivalent labeled aliases such as:

- `1712061`
- `SKU 1712061`

### Option D: Store a lexical regime attribute on entities

Promising follow-on improvement.

This could make later comparisons cheaper and more auditable, but the MVP can
compute regime on the fly.

### Option E: Route suspicious identifier merges to a review queue

Interesting optional enhancement.

This is useful for future human review workflows, but it should not replace
hard constraints for obvious identifier mismatches.

## Recommended Rollout

### Phase 1: Safety fix

Ship the shared identifier-aware policy and wire it into both resolver and
merge.

This is the minimum safe correction.

### Phase 2: Audit and UI clarity

Expose explicit identifier-policy reasons in decision traces and stop showing
raw fuzzy similarity as if it were decision confidence.

### Phase 3: Optional schema and cleanup work

Consider:

- adding a stored lexical regime or identifier facet
- adding extraction guidance for product parts or identifiers
- building a report for historical suspect merges

## Deliverables

### MVP deliverables

- [ ] Add a shared identifier-aware name analysis helper.
- [ ] Add lexical regime classification: `natural_language`, `identifier`,
      `hybrid`.
- [ ] Add canonical identifier parsing that preserves leading zeros.
- [ ] Add a shared dedup policy function that can hard-reject identifier
      mismatches.
- [ ] Apply identifier policy in `resolve_entity()`.
- [ ] Apply identifier policy in `resolve_entity_fast()`.
- [ ] Apply identifier policy in consolidation pairwise fuzzy comparison.
- [ ] Apply identifier policy in consolidation ANN candidate acceptance.
- [ ] Apply identifier policy in consolidation soft-zone candidate selection.
- [ ] Apply identifier policy in the multi-signal merge scorer.
- [ ] Add decision-trace reasons for identifier-policy accepts/rejects.
- [ ] Add tests covering ingestion-time and consolidation-time behavior.

### Visibility deliverables

- [ ] Update cycle detail APIs or traces so identifier-policy reasons are easy
      to inspect.
- [ ] Update dashboard merge display to distinguish fuzzy similarity from true
      decision confidence.

### Optional follow-on deliverables

- [ ] Add a stored lexical regime or identifier facet on entities.
- [ ] Add extraction prompt guidance or attributes for explicit identifiers.
- [ ] Add a one-off analysis tool to find likely historical identifier merges.
- [ ] Add a review or quarantine path for suspicious code/code merges.

## Acceptance Criteria

The change is correct when the following are true:

1. Distinct pure numeric identifiers do not merge during resolution or
   consolidation.
2. Equivalent labeled forms of the same identifier do merge or resolve.
3. Existing natural-language alias behavior still passes current tests.
4. Numeronyms such as `k8s` still work.
5. Revisioned identifiers such as `1712061-A` and `1712061-B` remain separate.
6. Audit traces explain why identifier-like candidates were blocked or allowed.

## Suggested Test Matrix

### Should match

- `1712061` <-> `SKU 1712061`
- `P/N AB-1712061-C` <-> `AB1712061C`
- `Part #001234` <-> `001234`
- `k8s` <-> `kubernetes`
- `ACT-R` <-> `ACTR`

### Should not match

- `1712061` <-> `1712018`
- `1712515` <-> `1712516`
- `1855012` <-> `1855015`
- `1712061-A` <-> `1712061-B`
- `RTX 4090` <-> `RTX 4080`
- `iPhone 15` <-> `iPhone 15 Pro`
- `001234` <-> `1234`

### Should be handled carefully

- `Model AB-1712061-C bracket` <-> `AB-1712061-C`
- `SKU 1712061 bracket` <-> `SKU 1712062 bracket`
- `2024 roadmap` <-> `2025 roadmap`
- `O0X123` <-> `00X123`

## Open Questions

These do not block the MVP, but they should be made explicit.

1. Should leading-zero aliases ever be merged automatically, or only by
   explicit user action?
2. Should hybrid names like `RTX 4090 GPU` be treated as `hybrid` or
   `natural_language` in the first release?
3. Should suspicious identifier mismatches be logged only, or also surfaced as
   a first-class dashboard review item?
4. Should legacy merge records be post-analyzed for likely identifier
   false positives after the fix ships?

## Recommendation

Implement the shared identifier-aware policy first and keep it narrow.

The first version should optimize for:

- false-positive prevention
- cross-path consistency
- transparent audit reasons

It should not try to become a full product-catalog normalization system in one
step.

That is the smallest change that directly addresses the observed SKU problem
without regressing the parts of dedup that are already working.
