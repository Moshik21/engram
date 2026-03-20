# Extraction Quality — Issue Tracker

Tracking extraction/entity resolution failures discovered 2026-03-20.
Observed: 90% of "Person" entities are misclassified (18/20 sampled).

**Architecture context**: Engram uses the **narrow deterministic pipeline** for extraction
(zero LLM). Only the embedding model (Gemini) is used as an external API. Extraction
is regex-based via staged extractors: `IdentityEntityExtractor`, `RelationshipPatternExtractor`,
`AttributeEvidenceExtractor`, `TemporalEvidenceExtractor`. No Haiku, no Ollama.

---

## Issue 1: Proper name regex too aggressive

**Status:** ✅ Fixed (2026-03-20)
**Severity:** Critical — root cause of most garbage entities
**Location:** `server/engram/extraction/narrow/entity_extractor.py:12,264-286`

The `_PROPER_NAMES` regex `\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b` matches **any
capitalized word 3+ characters**. At sentence starts (which are always capitalized),
this captures every opening word as an entity candidate. The `_STOPWORDS` blocklist
(~30 words) is far too small to filter the noise.

**Examples of sentence-start captures:**
- "Lets run with..." → "Lets" (capitalized sentence start)
- "Either use one adaptive..." → "Either"
- "Subtle but transformative" → "Subtle"
- "Refresh and try" → "Refresh"

**Fix applied (Layer 1):**
- Expanded `_STOPWORDS` from ~30 to ~85 words (discourse markers, protocol fragments, AI sentence openers like "Lets", "Subtle", "Either")
- Added `_is_sentence_initial()` helper — single-word proper names at sentence-initial position are skipped unless they also appear mid-sentence elsewhere in the text
- Lowered proper_name confidence from 0.65 → 0.55 (Layer 2), putting them in the defer band at all entity counts
- Cross-episode corroboration gate in evidence adjudication (Layer 3): bare proper_name entities need count ≥ 2 before promotion

---

## Issue 2: `_infer_entity_type` defaults to Person

**Status:** ✅ Fixed (2026-03-20)
**Severity:** Critical — direct cause of 90% type misclassification
**Location:** `server/engram/extraction/narrow/entity_extractor.py:206-208`

Previously:
```python
# Default to Person for proper names, Concept for others
if name[0].isupper() and name.isalpha():
    return "Person"
```

**Fix applied (Layer 1a):**
Changed default from `"Person"` to `"Concept"`. Identity captures (`_IDENTITY_CAPTURES`) still hardcode "Person" via their signal — so real people declared via "my name is X", "I am X", "my wife X" etc. still get typed correctly. Bare capitalized words now safely default to Concept.

**Additional fix (2026-03-20):** Expanded `_TECH_KEYWORDS` from ~20 to ~100+ entries (languages, frameworks, cloud, AI/ML, vector DBs, ORMs, DevOps). Added `_COMPANY_SUFFIXES` (inc, llc, corp, labs, ai, io) → returns "Organization". Added `_PRODUCT_SUFFIXES` (app, pro, studio, cloud, hub, kit, os) → returns "Product". Entities like Webpack, Vercel, Terraform now correctly typed as Technology instead of Concept.

---

## Issue 3: Input contamination from hooks

**Status:** ✅ Fixed (2026-03-20)
**Severity:** High — affects all hook-captured content
**Location:** `capture-prompt.sh`, `capture-response.sh` hooks → `observe()`

Raw conversation text includes protocol markers (`[user|web]`, `[assistant|server]`),
tool use XML, markdown tables, code blocks, and task notification fragments. The
proper name regex fires on capitalized words inside these protocol artifacts.

**Examples:**
- `[user|web] Lets run with n` → "Lets" extracted as entity
- `[assistant|web] Try this — without --turbopack` → words after `. ` captured
- `<task-notification>` XML content → capitalized tag values extracted

**Fix applied (Layer 1d):**
Added `_NOISE_STRIP` regex that sanitizes input before extraction. Strips:
- Protocol markers: `[user|...]`, `[assistant|...]`, `[system|...]`
- XML/HTML tags: `<anything up to 80 chars>`
- Fenced code blocks: `` ```...``` ``
- Inline code: `` `...` ``
- URLs: `https://...`

Applied in both `entity_extractor.py` (top of `extract()`) and `cues.py` (`_entity_mentions()`).

---

## Issue 4: Bootstrap artifact pollution

**Status:** ✅ Fixed (2026-03-20)
**Severity:** High — creates fake entities from documentation examples
**Location:** `bootstrap_project` tool → `graph_manager.bootstrap_project()`

Project file bootstrapping ingests README, design docs, and config files. The narrow
pipeline's proper name regex fires on every capitalized word in these docs. Additionally,
example sentences in docs create identity-pattern entities.

**Examples:**
- README example: "My son Liam plays soccer every Tuesday" → `_IDENTITY_CAPTURES` fires,
  creates "Liam" as a Person entity (correct type, but it's a documentation example, not real)
- Design doc: "Emma scored two goals!" → "Emma" as Person (doc example)
- Design doc: "my dad retired" → `_IDENTITY_PATTERNS` fires on "my dad"

**Fix applied (Layer 3c):**
Bootstrap episodes are now immediately updated to `projection_state=CUE_ONLY` + `status=COMPLETED` after `store_episode()` in `_observe_project_files()`. This preserves FTS/embedding searchability via the cue layer while preventing triage/replay from running entity extraction on documentation content.

---

## Issue 5: Summary contamination

**Status:** ✅ Fixed (2026-03-20)
**Severity:** Medium — degrades recall quality
**Location:** Entity summary generation in extraction + reconsolidation updates

Entity summaries contain raw conversation protocol markers, task notification XML,
tool output fragments, and unprocessed markdown. Since the narrow pipeline uses
`source_span` (the sentence containing the entity) as the basis for summaries,
protocol noise in the input becomes protocol noise in the summary.

**Examples:**
- Summary: `[user|web] Try this — without --turbopack`
- Summary: `<task-notification><task-id>ada9e0bfa064b5a62</task-id>`
- Summary: `konnermoshier@Mac server % uv run engram serve`

**Fix applied (2026-03-20):**
- `_NOISE_STRIP` sanitization (Layer 1d) cleans input before extraction → new `source_span` values are clean
- Added `_is_noisy_span()` in `entity_extractor.py` — rejects source spans where >50% of characters are non-alphanumeric, returns `None` instead
- Added `is_noisy_text()` to `utils/text_guards.py` — shared utility for noisy text detection
- Microglia `_score_c3_summary` now also flags noisy segments (via `is_noisy_text`) alongside existing `is_meta_summary` checks — cleans up existing contaminated summaries during consolidation cycles

---

## Issue 6: Entity resolution doesn't catch garbage

**Status:** ✅ Mitigated (2026-03-20) — upstream fixes prevent garbage creation
**Severity:** Medium — garbage entities persist permanently
**Location:** `resolve_entity_fast()`, `find_entity_candidates()`

Entity resolution focuses on merging duplicates but has no validation gate. Garbage
entities like "had", "Lets", "Either" don't match anything in the graph, so they
get created as new entities and persist.

**Mitigated by upstream fixes:**
- Issue 1 fix: expanded stopwords + sentence-position demotion prevent most garbage from reaching extraction
- Issue 2 fix: default type is now Concept, not Person
- Layer 2: proper_name confidence lowered to 0.55, no cold-start relaxation for bare proper_name signals
- Layer 3a: evidence adjudication requires cross-episode corroboration (count ≥ 2) for bare proper_name entities
- Layer 3b: replay graph-vocabulary linking skips entity names < 3 chars or matching stopwords

**Additional fix (2026-03-20):** Added `validate_entity_name()` in `resolver.py` as pre-creation gate, called in `apply.py` before entity resolution/creation. Rejects:
- Names shorter than 2 characters
- Names longer than 5 words (sentence fragments)
- All-lowercase names (unless containing dots/slashes indicating tech tokens like `next.js`, `src/utils`)

10 new tests in `test_resolver.py`.

---

## Issue 7: Graph traversal amplifies noise

**Status:** ✅ Mitigated (2026-03-20) — downstream of Issues 1-6 fixes
**Severity:** Medium — degrades recall precision
**Location:** Recall pipeline (BFS/PPR traversal)

With ~90% garbage entities in the graph, traversal walks through junk nodes and
pulls in noise. This is downstream of Issues 1-6 — cleaning extraction cleans
traversal automatically.

**Mitigated:** Issues 1-6 now prevent garbage entity creation. New entities entering the graph are properly filtered. Existing garbage entities will be cleaned by the prune phase over time (low access count, no relationships).

**Additional fix (2026-03-20):** Added `traversal_min_edge_weight` config field (default 0.05) to `ActivationConfig`. Both BFS (`bfs.py`) and PPR (`ppr.py`) now skip neighbor edges below this threshold during traversal. Garbage entities with only weak inferred edges (weight < 0.05) are no longer reached during spreading activation. Real entities with episodic evidence have weight ≥ 0.5 and are unaffected.

---

## Root cause summary

The failures trace to two lines in `entity_extractor.py`:

1. **Line 12**: `_PROPER_NAMES = re.compile(r"\b[A-Z][a-z]{2,}...")` — matches any
   capitalized word, including sentence starts, brand names, tool names, etc.

2. **Line 207-208**: `if name[0].isupper() and name.isalpha(): return "Person"` —
   defaults anything capitalized to Person type.

Combined with unsanitized input (hooks dump raw conversation with protocol markers),
the narrow pipeline creates a Person entity for every capitalized word it encounters.

## Priority order

1. ~~**Issue 2** (`_infer_entity_type` default)~~ — ✅ Fixed. Default Person → Concept + expanded tech keywords + company/product suffix detection.
2. ~~**Issue 1** (proper name regex)~~ — ✅ Fixed. Stopwords expanded, sentence-position demotion, confidence lowered, corroboration gate.
3. ~~**Issue 3** (input sanitization)~~ — ✅ Fixed. `_NOISE_STRIP` strips protocol markers, XML, code blocks, URLs.
4. ~~**Issue 4** (bootstrap extraction)~~ — ✅ Fixed. Bootstrap episodes marked CUE_ONLY.
5. ~~**Issue 6** (validation gate)~~ — ✅ Fixed. `validate_entity_name()` rejects short/long/lowercase names before creation.
6. ~~**Issue 5** (summary contamination)~~ — ✅ Fixed. Noisy span rejection + microglia cleanup for existing summaries.
7. ~~**Issue 7** (traversal noise)~~ — ✅ Fixed. `traversal_min_edge_weight=0.05` filters garbage edges in BFS/PPR.
