# G7 — Citation-Scan Yield/Precision + Echo-Immunity (the M1.4 gate)

**Date:** 2026-07-21 (HEAD d6cfeb2)
**Question:** Does the echo-guarded citation scan (M1.4) fire `used`-tier
events when the agent genuinely relies on a surfaced entity, and stay silent
on echoes and incidental mentions? G7 is a **mandatory precondition to G6 and
the `usage_ranking_enabled` flip** (RECENCY_FREQUENCY_GOAL, resolved
decision 10 / F1).
**Rig:** `<scratchpad>/experiments/rf_echo/run_echo.py` — runs the
PRODUCTION scan code (`engram.retrieval.feedback`) against planted
transcripts. Fully local, zero LLM calls, judge-free (planted labels).
**Results JSON:** persisted next to the rig at
`<scratchpad>/experiments/rf_echo/rf_echo_results.json`.

## Verdict

**PASS — all four G7 gates clear.**

| G7 gate | Threshold | Measured |
|---|---|---|
| (a) used-event rate on planted-reliance transcripts | ≥ 0.70 | **1.00** (10/10) |
| (b) precision vs planted labels | ≥ 0.80 | **1.00** (tp=10, fp=0, fn=0) |
| (c) adversarial common-word fires | ~0 | **0** (3 cases) |
| (d) verbatim top-k fed back as next turn | ~0 fires | **0** |
| (e) surfaced-episode-text echoed verbatim | ~0 fires | **0** |

Consequence: per resolved decision 10, the echo guard + G7 have now landed —
the citation scan (`recall_usage_feedback_enabled`) is cleared to accumulate
`used`-tier events. **Ranking impact still waits**: `w_used` reaches ranking
only through `u`, and `usage_ranking_enabled` stays default-off until the
remaining flip gates (M2.6: G2/G6 real-corpus + live organic continuity) pass.

## Mechanism (what shipped in M1.4, `retrieval/feedback.py`)

- **Surfaced ring buffer.** Recall surfacing appends `(entity_id, name,
  snippet)` per group to a bounded ring (`_USAGE_RING_CAP=32`, snippet capped
  at 400 chars, tokens normalized). The Capture fast path (`store_episode`)
  scans the next observed turn against it; when the ring is empty the scan
  short-circuits on one dict lookup, so the CQRS no-LLM path pays ~0.
- **Echo mask (token shingles).** Every buffered payload is decomposed into
  5-token shingles (whole-payload fallback at ≥3 tokens). Content token
  positions covered by any shingle are marked *echoed*; a mention wholly
  inside an echoed span never fires. Parroting surfaced text back is thus
  structurally invisible to the scan.
- **Mask-only surfaced-text channel.** `note_surfaced_text` /
  `note_surfaced_texts_from_response` feed **all** surfaced result/packet
  text (episode content, packet summaries — not just entity snippets) into
  the mask ring with *no entity binding and no fire eligibility*. This closes
  the M1-verifier bypass where echoing a surfaced EPISODE verbatim read as
  "novel" tokens relative to the entity snippets. Scenario (e) exercises it.
- **Novel-mention rule.** An entry fires only when its name token sequence
  appears with at least one non-echoed token. Single-token names must be ≥3
  chars AND share non-echoed context vocabulary (±12-token window, tokens ≥4
  chars) with the surfaced snippet — an incidental common word with no
  topical tie to what was surfaced does not count as reliance.
- **(entity, group) 30-min dedup.** At most one used event per
  `(group_id, entity_id)` per rolling 30-minute window
  (`_USAGE_DEDUP_WINDOW_SECONDS`), so a persistently in-context entity cannot
  accrue runaway used events. Stale dedup keys are pruned each scan.
- **Write path.** Fired entries record `record_access(tier="used")` —
  `w_used=0.3` in the one usage store; hygiene `access_history` is unchanged.
  Everything is gated on `recall_usage_feedback_enabled` (default False).

## Scenarios and numbers

1. **Planted reliance (a):** 10 transcripts; each surfaces the target entity
   plus 2 distractors, then the "agent turn" relies on the target in novel
   wording. **10/10 targets fired; 0 distractors fired.** Rate 1.00 vs
   floor 0.70.
2. **Precision (b):** over all planted positives and negatives (10 reliance
   targets + 20 surfaced distractors + 3 common-word cases + 3 unrelated
   turns + the two echo scenarios): **tp=10, fp=0, fn=0 → precision 1.00**
   vs gate ≥ 0.80.
3. **Adversarial common word (c):** "Python" surfaced, next turn mentions
   python incidentally (a rename one-liner; a zoo snake), plus a <3-char
   name ("Go") in ordinary prose. **0 fires over 3 cases** — the
   single-token context-vocabulary requirement and the short-name reject did
   the work.
4. **Verbatim top-k echo (d):** the full surfaced payload of 5 entities is
   fed back as the synthetic next turn. **0 fires** — every mention sits
   inside echoed shingle spans.
5. **Surfaced-episode-text echo (e):** an episode's text (not an entity
   snippet) is surfaced via the mask-only channel, then echoed verbatim with
   a conversational prefix. **0 fires** — the mask covers all surfaced text,
   not just entity snippets.

## LIMITATIONS (mandated by the M1 verifiers)

- **Paraphrase echo is NOT caught.** The echo mask is token-shingle based:
  an agent that *paraphrases* surfaced text into new tokens ("she does
  ceramics on Saturdays" for the surfaced pottery snippet) reads as a novel
  mention and fires a false `used` event. No semantic/embedding mask is
  attempted. **Why acceptable at these magnitudes:** a false fire is one
  `w_used=0.3` event, capped at one per (entity, group) per 30-minute
  window; `u` log-compresses frequency (`N_cap=50`) and enters ranking only
  as the bounded multiplier `(1 + β·u)`, `β ≤ 0.30` — so even a sustained
  paraphrase-echo stream moves an entity at most within the ≤30% relevance
  band and can never overtake a stronger semantic match (the overtake
  theorem). The loop-risk gradient (surfaced → re-ranked → re-surfaced) is
  further damped because surfaced-tier events themselves carry
  `w_ranking=0` — only the *paraphrased mention* leaks weight, at 0.3, not
  the surfacing itself.
- **Multi-token incidental mentions can still fire.** The
  context-vocabulary requirement applies only to single-token names; a
  multi-token name ("Konner Moshier") mentioned incidentally in novel tokens
  fires without genuine reliance. **Why acceptable:** multi-token name
  collisions with unrelated discourse are far rarer than common single
  words (the case the guard exists for); the misfire cost is identical to
  the paraphrase case (0.3 per 30-min window, log-compressed, β-bounded);
  and the tier ladder already tolerates environmental mention noise by
  design — ingestion mentions carry `w_mentioned=0.1` (F10), so an
  occasional spurious used event is a 3× mentioned event, well inside the
  bounded tiebreaker band, not a new signal class.
- Rig scale: 10 positive + 8 negative transcripts, single synthetic
  catalog — this is a mechanism gate (echo-immunity + precision floor), not
  a live-yield estimate. Live used-event *yield* on organic dogfood traffic
  is a separate observation window and does not gate G7.

## Reproduce

```bash
cd /Users/konnermoshier/Engram/server && \
env HOME=<scratchpad>/fakehome ENGRAM_MODE=lite uv run python \
    <scratchpad>/experiments/rf_echo/run_echo.py
```

Exit 0 iff all gates pass; writes `rf_echo_results.json` beside the rig.
