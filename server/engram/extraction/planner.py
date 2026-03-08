"""Deterministic projection planning for long episodes."""

from __future__ import annotations

import re

from engram.config import ActivationConfig
from engram.extraction.extractor import MAX_EXTRACTION_INPUT_CHARS
from engram.extraction.models import ProjectedSpan, ProjectionPlan
from engram.models.episode import Episode
from engram.models.episode_cue import EpisodeCue

_SENTENCE_RE = re.compile(r".+?(?:[.!?](?:\s+|$)|\n{2,}|$)", re.S)
_CORRECTION_HINTS = (
    "actually",
    "correction",
    "updated",
    "changed",
    "instead",
    "no longer",
    "moved to",
    "moved from",
)


class ProjectionPlanner:
    """Plan which spans of an episode should be projected first."""

    def __init__(self, cfg: ActivationConfig) -> None:
        self._cfg = cfg
        self._max_input_chars = MAX_EXTRACTION_INPUT_CHARS

    def plan(
        self,
        episode: Episode,
        cue: EpisodeCue | None = None,
    ) -> ProjectionPlan:
        """Build a deterministic projection plan for an episode."""
        content = episode.content or ""
        total_chars = len(content)
        if not content:
            return ProjectionPlan(
                episode_id=episode.id,
                strategy="empty_episode",
                spans=[],
                selected_text="",
                selected_chars=0,
                total_chars=0,
            )

        if (
            not self._cfg.projector_v2_enabled
            or not self._cfg.projection_planner_enabled
            or not self._cfg.targeted_projection_enabled
            or total_chars <= self._max_input_chars
        ):
            return self._full_episode_plan(episode.id, content, total_chars)

        candidate_spans = self._segment_content(content)
        if not candidate_spans:
            span = ProjectedSpan(
                span_id="span_0",
                start_char=0,
                end_char=min(total_chars, self._max_input_chars),
                text=content[: self._max_input_chars],
                score=1.0,
                reasons=["fallback_window"],
            )
            return ProjectionPlan(
                episode_id=episode.id,
                strategy="fallback_window",
                spans=[span],
                selected_text=span.text,
                selected_chars=len(span.text),
                total_chars=total_chars,
                was_truncated=total_chars > len(span.text),
                warnings=["planner_fallback"],
            )

        scored_spans = [
            self._score_span(span, idx, len(candidate_spans), cue)
            for idx, span in enumerate(candidate_spans)
        ]
        selected = self._select_spans(scored_spans)
        selected_text = "\n\n".join(span.text.strip() for span in selected if span.text.strip())
        selected_text = selected_text[: self._max_input_chars]
        selected_chars = len(selected_text)
        strategy = "targeted_spans" if len(selected) > 1 else "focused_span"
        warnings: list[str] = []
        if selected_chars < total_chars:
            warnings.append("targeted_projection")

        return ProjectionPlan(
            episode_id=episode.id,
            strategy=strategy,
            spans=selected,
            selected_text=selected_text,
            selected_chars=selected_chars,
            total_chars=total_chars,
            was_truncated=selected_chars < total_chars,
            warnings=warnings,
        )

    def _full_episode_plan(
        self,
        episode_id: str,
        content: str,
        total_chars: int,
    ) -> ProjectionPlan:
        span = ProjectedSpan(
            span_id="span_0",
            start_char=0,
            end_char=total_chars,
            text=content[: self._max_input_chars],
            score=1.0,
            reasons=["full_episode"],
        )
        return ProjectionPlan(
            episode_id=episode_id,
            strategy="full_episode",
            spans=[span],
            selected_text=span.text,
            selected_chars=len(span.text),
            total_chars=total_chars,
            was_truncated=total_chars > len(span.text),
            warnings=["input_truncated"] if total_chars > len(span.text) else [],
        )

    def _segment_content(self, content: str) -> list[ProjectedSpan]:
        pieces = []
        for match in _SENTENCE_RE.finditer(content):
            raw = match.group()
            text = raw.strip()
            if not text:
                continue
            pieces.append((text, match.start(), match.end()))

        if not pieces:
            stripped = content.strip()
            if not stripped:
                return []
            return [
                ProjectedSpan(
                    span_id="span_0",
                    start_char=0,
                    end_char=len(content),
                    text=stripped[: self._cfg.projection_span_max_chars],
                )
            ]

        target_chars = self._cfg.projection_span_target_chars
        max_chars = min(self._cfg.projection_span_max_chars, self._max_input_chars)
        min_chars = self._cfg.projection_min_span_chars

        spans: list[ProjectedSpan] = []
        buffer: list[str] = []
        span_start = 0
        span_end = 0
        char_count = 0

        def flush() -> None:
            nonlocal buffer, span_start, span_end, char_count
            if not buffer:
                return
            text = " ".join(buffer).strip()
            if not text:
                buffer = []
                char_count = 0
                return
            spans.append(
                ProjectedSpan(
                    span_id=f"span_{len(spans)}",
                    start_char=span_start,
                    end_char=span_end,
                    text=text[:max_chars],
                )
            )
            buffer = []
            char_count = 0

        for piece_text, piece_start, piece_end in pieces:
            if len(piece_text) > max_chars:
                if buffer:
                    flush()
                for offset in range(0, len(piece_text), max_chars):
                    window = piece_text[offset : offset + max_chars]
                    spans.append(
                        ProjectedSpan(
                            span_id=f"span_{len(spans)}",
                            start_char=piece_start + offset,
                            end_char=min(piece_start + offset + len(window), piece_end),
                            text=window,
                        )
                    )
                continue

            if not buffer:
                span_start = piece_start

            projected_size = char_count + len(piece_text) + (1 if buffer else 0)
            if buffer and projected_size > target_chars and char_count >= min_chars:
                flush()
                span_start = piece_start

            buffer.append(piece_text)
            span_end = piece_end
            char_count += len(piece_text) + (1 if len(buffer) > 1 else 0)

            if char_count >= max_chars:
                flush()

        flush()
        return spans

    def _score_span(
        self,
        span: ProjectedSpan,
        idx: int,
        total_spans: int,
        cue: EpisodeCue | None,
    ) -> ProjectedSpan:
        score = 0.05
        reasons: list[str] = []
        lowered = span.text.lower()

        if cue is not None:
            mention_hits = 0
            for mention in cue.entity_mentions[:8]:
                text = (mention.get("text") or "").strip().lower()
                if text and text in lowered:
                    mention_hits += 1
            if mention_hits:
                score += min(0.35, mention_hits * 0.12)
                reasons.append("cue_mentions")

            temporal_hits = 0
            for marker in cue.temporal_markers[:6]:
                text = marker.strip().lower()
                if text and text in lowered:
                    temporal_hits += 1
            if temporal_hits:
                score += min(0.20, temporal_hits * 0.08)
                reasons.append("temporal_markers")

            quote_hits = 0
            for quote in cue.quote_spans[:4]:
                text = quote.strip().lower()
                if text and text in lowered:
                    quote_hits += 1
            if quote_hits:
                score += min(0.20, quote_hits * 0.10)
                reasons.append("quoted_span")

            if cue.contradiction_keys:
                hint_hits = sum(1 for hint in _CORRECTION_HINTS if hint in lowered)
                if hint_hits:
                    score += min(0.30, hint_hits * 0.15)
                    reasons.append("correction_hint")

        if any(hint in lowered for hint in _CORRECTION_HINTS):
            score += 0.12
            reasons.append("late_correction_pattern")

        if total_spans > 1:
            position_ratio = idx / max(total_spans - 1, 1)
            score += 0.06 * position_ratio
            if position_ratio > 0.6:
                reasons.append("late_span_bias")

        if self._cfg.emotional_salience_enabled:
            try:
                from engram.extraction.salience import compute_emotional_salience

                score += 0.12 * compute_emotional_salience(span.text).composite
            except Exception:
                pass

        score += min(len(span.text) / max(self._cfg.projection_span_target_chars, 1), 1.0) * 0.05

        return ProjectedSpan(
            span_id=span.span_id,
            start_char=span.start_char,
            end_char=span.end_char,
            text=span.text,
            score=round(score, 4),
            reasons=reasons,
        )

    def _select_spans(self, spans: list[ProjectedSpan]) -> list[ProjectedSpan]:
        if not spans:
            return []

        max_total_chars = self._max_input_chars
        budget = self._cfg.projection_span_budget
        radius = self._cfg.projection_neighbor_span_radius
        selected_indices: set[int] = set()
        selected_chars = 0

        ranked_indices = sorted(
            range(len(spans)),
            key=lambda idx: (spans[idx].score, -spans[idx].start_char),
            reverse=True,
        )

        def try_add(idx: int, reason: str | None = None) -> None:
            nonlocal selected_chars
            if idx < 0 or idx >= len(spans) or idx in selected_indices:
                return
            span = spans[idx]
            if selected_chars + len(span.text) > max_total_chars and selected_indices:
                return
            selected_indices.add(idx)
            selected_chars += len(span.text)
            if reason is not None and reason not in span.reasons:
                span.reasons.append(reason)

        for idx in ranked_indices[:budget]:
            try_add(idx)
            for offset in range(1, radius + 1):
                try_add(idx - offset, reason="neighbor_context")
                try_add(idx + offset, reason="neighbor_context")

        for idx in ranked_indices:
            if selected_chars >= max_total_chars:
                break
            if idx in selected_indices:
                continue
            try_add(idx)

        selected = [spans[idx] for idx in sorted(selected_indices)]
        if not selected:
            return [max(spans, key=lambda span: span.score)]

        # If we substantially under-filled the budget, favor broader coverage.
        if selected_chars < max_total_chars * 0.55 and len(selected) < len(spans):
            remaining = [
                spans[idx]
                for idx in ranked_indices
                if idx not in selected_indices
            ]
            for span in remaining:
                if selected_chars + len(span.text) > max_total_chars:
                    continue
                selected.append(span)
                selected_chars += len(span.text)
                if selected_chars >= max_total_chars * 0.85:
                    break
            selected.sort(key=lambda span: span.start_char)

        return selected


def summarize_plan(plan: ProjectionPlan) -> dict[str, object]:
    """Return compact plan metadata for observability events."""
    avg_score = (
        round(sum(span.score for span in plan.spans) / len(plan.spans), 4)
        if plan.spans
        else 0.0
    )
    return {
        "strategy": plan.strategy,
        "selectedChars": plan.selected_chars,
        "totalChars": plan.total_chars,
        "selectedSpanCount": len(plan.spans),
        "wasTruncated": plan.was_truncated,
        "avgSpanScore": avg_score,
        "coverage": round(plan.selected_chars / max(plan.total_chars, 1), 4),
    }
