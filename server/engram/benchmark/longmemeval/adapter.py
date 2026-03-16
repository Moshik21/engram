"""Engram adapter for LongMemEval: ingest sessions, query, collect evidence.

Uses the real running Engram infrastructure (HelixDB, embeddings)
with a dedicated group_id per question for isolation.

Key design choices for benchmark accuracy:
- Session dates are prepended to episode content during ingestion
  so temporal cues survive extraction and are embedded in vectors.
- Temporal queries get episode-first evidence (raw conversation text
  with dates, not entity-summary abstractions that lose detail).
- Full episode content is fetched post-recall to bypass the 500-char
  truncation in the recall pipeline.
- Zero LLM calls: no reader, no judge.  Evidence is returned raw
  for embedding-based evaluation.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from engram.benchmark.longmemeval.dataset import (
    LongMemEvalInstance,
    LongMemEvalSession,
)
from engram.config import ActivationConfig, EngramConfig
from engram.extraction.factory import create_extractor
from engram.graph_manager import GraphManager
from engram.storage.factory import create_stores
from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex
from engram.storage.resolver import resolve_mode

logger = logging.getLogger(__name__)

# Group ID prefix for all LongMemEval data
GROUP_PREFIX = "longmemeval"

# Temporal query detection patterns
_TEMPORAL_PATTERNS = re.compile(
    r"\b("
    r"first|before|after|when did|how many days|how many months|"
    r"how many weeks|how long ago|how long since|most recent|"
    r"latest|earlier|last time|which came first|chronological|"
    r"ago|since then|prior to|previously|sooner|order of|"
    r"which did .+ first|what was the first|set up first|"
    r"took .+ first|did .+ before|started .+ before"
    r")\b",
    re.IGNORECASE,
)


def _parse_session_date(raw: str) -> datetime | None:
    """Parse a LongMemEval session date string into a datetime.

    LongMemEval uses the format ``"2023/04/10 (Mon) 17:50"``.
    Also handles ISO 8601 and common date-only formats as fallbacks.
    """
    if not raw:
        return None

    # Strip parenthesised day-of-week like "(Mon)" so strptime can match
    cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", raw).strip()

    for fmt in (
        "%Y/%m/%d %H:%M",    # LongMemEval primary format
        "%Y/%m/%d",           # date-only variant
        "%Y-%m-%dT%H:%M:%S",  # ISO with time
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue

    # Last resort: fromisoformat handles many ISO variants
    try:
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


def _is_temporal_query(question: str) -> bool:
    """Detect if a question requires temporal reasoning."""
    return bool(_TEMPORAL_PATTERNS.search(question))


@dataclass
class AdapterStats:
    """Track adapter operations for cost/performance analysis."""

    sessions_ingested: int = 0
    episodes_stored: int = 0
    episodes_extracted: int = 0
    extraction_calls: int = 0
    embedding_calls: int = 0
    recall_calls: int = 0
    total_ingest_ms: float = 0.0
    total_query_ms: float = 0.0


@dataclass
class QueryResult:
    """Result of querying Engram for a LongMemEval question."""

    question_id: str
    hypothesis: str
    evidence: list[str]
    evidence_scores: list[float]
    retrieved_session_ids: list[str]
    latency_ms: float
    num_entities: int = 0
    num_episodes: int = 0


class EngramLongMemEvalAdapter:
    """Uses the real Engram stack with a dedicated group_id per question.

    Connects to the running HelixDB/embedding infrastructure and
    uses group_id isolation to keep benchmark data separate.

    Zero LLM calls: evidence is collected raw for embedding-based evaluation.
    """

    def __init__(
        self,
        cfg: ActivationConfig | None = None,
        *,
        extraction_mode: str = "auto",
        consolidation: bool = False,
        top_k: int = 10,
        shared_group: bool = False,
    ) -> None:
        self._cfg = cfg
        self._extraction_mode = extraction_mode
        self._consolidation = consolidation
        self._top_k = top_k
        self._shared_group = shared_group
        self.stats = AdapterStats()

        # Shared infrastructure (initialized once)
        self._config: EngramConfig | None = None
        self._graph_store: GraphStore | None = None
        self._activation_store: ActivationStore | None = None
        self._search_index: SearchIndex | None = None
        self._initialized = False

        # Per-question state
        self._manager: GraphManager | None = None
        self._current_group_id: str | None = None
        # session_id → date mapping for the current question
        self._session_dates: dict[str, str] = {}

    async def _ensure_initialized(self) -> None:
        """Connect to the real running Engram infrastructure once."""
        if self._initialized:
            return

        # Load .env files so resolver sees ENGRAM_MODE, GEMINI_API_KEY, etc.
        from pathlib import Path

        from dotenv import load_dotenv

        load_dotenv(Path.home() / ".engram" / ".env", override=False)
        load_dotenv()  # Also load ./server/.env or ./.env

        self._config = EngramConfig()
        if self._cfg is not None:
            self._config.activation = self._cfg

        mode = await resolve_mode(self._config.mode)
        logger.info("LongMemEval adapter connecting: mode=%s", mode.value)

        graph, activation, search = create_stores(mode, self._config)

        await graph.initialize()
        await search.initialize()

        self._graph_store = graph
        self._activation_store = activation
        self._search_index = search
        self._initialized = True
        logger.info("LongMemEval adapter initialized on real infrastructure")

    def _group_id_for(self, question_id: str) -> str:
        """Generate a group_id for a question.

        Per-question isolation (default): each question gets its own group,
        matching the benchmark's design where each question has its own
        conversation history (1-6 sessions).

        Shared mode: all episodes in one group, simulating a real user's
        full memory. Set via ``self._shared_group`` constructor parameter.
        """
        if self._shared_group:
            return GROUP_PREFIX
        return f"{GROUP_PREFIX}_{question_id}"

    async def _setup_manager(self, question_id: str) -> None:
        """Create a GraphManager for the given question's group_id."""
        await self._ensure_initialized()
        assert self._graph_store is not None
        assert self._activation_store is not None
        assert self._search_index is not None

        extractor = self._build_extractor()
        self._current_group_id = self._group_id_for(question_id)

        cfg = self._cfg or ActivationConfig()
        # Benchmark: give ALL result slots to episodes.
        cfg.passage_first_entity_budget = 0
        # Disable graph structural embedding search — no graph embeddings
        # are trained during the benchmark.
        cfg.weight_graph_structural = 0.0
        self._manager = GraphManager(
            self._graph_store,
            self._activation_store,
            self._search_index,
            extractor,
            cfg=cfg,
            runtime_mode="benchmark",
        )

    def _build_extractor(self) -> Any:
        """Build the entity extractor based on extraction mode."""
        if self._extraction_mode == "narrow":
            from engram.extraction.narrow_adapter import NarrowExtractorAdapter

            cfg = self._cfg or ActivationConfig()
            return NarrowExtractorAdapter(cfg)

        if self._extraction_mode == "none":
            return _NoopExtractor()

        # auto or anthropic — use the factory with the full config
        assert self._config is not None
        self._config.activation.extraction_provider = self._extraction_mode
        return create_extractor(self._config)

    def _format_session_content(self, session: LongMemEvalSession) -> str:
        """Format session content with date header for temporal context."""
        parts = []
        if session.date:
            parts.append(f"[Conversation from {session.date}]")
        parts.append(session.text)
        return "\n".join(parts)

    async def ingest_instance(self, instance: LongMemEvalInstance) -> None:
        """Ingest all haystack sessions for one question."""
        await self._setup_manager(instance.question_id)
        assert self._manager is not None
        assert self._current_group_id is not None

        # Clean up previous data for this question (safe with per-question groups)
        if not self._shared_group:
            await self.cleanup_group(instance.question_id)

        start = time.perf_counter()
        group_id = self._current_group_id

        # Build session date mapping for query phase
        self._session_dates = {s.session_id: s.date for s in instance.sessions if s.date}

        for session in instance.sessions:
            # Prepend session date to content so temporal cues are
            # embedded in vectors and survive in episode text.
            content = self._format_session_content(session)
            if not content.strip():
                continue

            # Parse session date to a datetime for structured sorting/filtering.
            conv_dt: datetime | None = None
            if session.date:
                conv_dt = _parse_session_date(session.date)

            if self._extraction_mode == "none":
                await self._manager.store_episode(
                    content,
                    group_id=group_id,
                    source=f"lme:{session.session_id}",
                    session_id=session.session_id,
                    conversation_date=conv_dt,
                )
                self.stats.episodes_stored += 1
            else:
                episode_id = await self._manager.store_episode(
                    content,
                    group_id=group_id,
                    source=f"lme:{session.session_id}",
                    session_id=session.session_id,
                    conversation_date=conv_dt,
                )
                try:
                    await self._manager.project_episode(episode_id, group_id=group_id)
                    self.stats.episodes_extracted += 1
                    self.stats.extraction_calls += 1
                except Exception:
                    logger.debug(
                        "Extraction failed for session %s",
                        session.session_id,
                        exc_info=True,
                    )
                self.stats.episodes_stored += 1

            self.stats.sessions_ingested += 1

        if self._consolidation:
            await self._run_consolidation(group_id)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stats.total_ingest_ms += elapsed_ms
        logger.info(
            "Ingested %d sessions for %s in %.0fms (group=%s)",
            len(instance.sessions),
            instance.question_id,
            elapsed_ms,
            group_id,
        )

    async def _run_consolidation(self, group_id: str) -> None:
        """Run consolidation phases after ingestion."""
        try:
            from engram.consolidation.engine import ConsolidationEngine

            assert self._graph_store is not None
            assert self._activation_store is not None
            assert self._search_index is not None

            cons_store: Any = None
            graph = self._graph_store
            if hasattr(graph, "_db_path"):
                from engram.consolidation.store import (
                    SQLiteConsolidationStore,
                )

                cons_store = SQLiteConsolidationStore(graph._db_path)
                await cons_store.initialize()

            if cons_store is None:
                logger.warning("No consolidation store for this mode")
                return

            cfg = self._cfg or ActivationConfig()
            engine = ConsolidationEngine(
                graph_store=self._graph_store,
                activation_store=self._activation_store,
                search_index=self._search_index,
                consolidation_store=cons_store,
                cfg=cfg,
            )
            await engine.run_cycle(group_id=group_id, trigger="benchmark")
        except Exception:
            logger.warning(
                "Consolidation failed, continuing without it",
                exc_info=True,
            )

    async def query_instance(self, instance: LongMemEvalInstance) -> QueryResult:
        """Query Engram with a LongMemEval question.

        Returns raw evidence texts — no LLM reader, no LLM judge.
        """
        assert self._manager is not None
        assert self._current_group_id is not None

        start = time.perf_counter()
        group_id = self._current_group_id
        is_temporal = _is_temporal_query(instance.question)

        # Fetch more results for temporal queries to ensure
        # we get enough episodes (not just entity summaries).
        fetch_limit = self._top_k * 2 if is_temporal else self._top_k

        self.stats.recall_calls += 1
        results = await self._manager.recall(
            query=instance.question,
            group_id=group_id,
            limit=fetch_limit,
            record_access=False,
            interaction_type="benchmark",
        )

        # Separate results by type for evidence composition.
        entity_evidence: list[str] = []
        episode_evidence: list[tuple[str, str]] = []  # (date, text)
        evidence_scores: list[float] = []
        retrieved_session_ids: list[str] = []
        num_entities = 0
        num_episodes = 0

        for r in results:
            score = r.get("score", 0.0)
            evidence_scores.append(score)

            if "entity" in r:
                entity = r["entity"]
                name = entity.get("name", "")
                summary = entity.get("summary", "")
                if summary:
                    entity_evidence.append(f"{name}: {summary}")
                    num_entities += 1
                for rel in entity.get("relationships", []):
                    src = rel.get("source_id") or rel.get("source", "")
                    pred = rel.get("predicate", "")
                    tgt = rel.get("target_id") or rel.get("target", "")
                    rel_text = f"{src} {pred} {tgt}"
                    if rel_text.strip() and pred:
                        entity_evidence.append(rel_text)

            elif r.get("result_type") == "cue_episode":
                ep = r.get("episode", {})
                cue = r.get("cue", {})
                raw_content = ep.get("content", "")
                chunk_ctx = cue.get("compressed_content", "")
                source = ep.get("source", "")
                ep_text = await self._enrich_episode_text(
                    ep.get("id", ""), chunk_ctx or raw_content, source, group_id
                )
                if ep_text:
                    ep_date = ep.get("conversation_date") or ep.get("created_at") or ""
                    episode_evidence.append((ep_date, ep_text))
                    num_episodes += 1
                self._track_session_id(source, retrieved_session_ids)

            elif "episode" in r:
                ep = r["episode"]
                raw_content = ep.get("content", "")
                source = ep.get("source", "")
                ep_text = await self._enrich_episode_text(
                    ep.get("id", ""), raw_content, source, group_id
                )
                if ep_text:
                    ep_date = ep.get("conversation_date") or ep.get("created_at") or ""
                    episode_evidence.append((ep_date, ep_text))
                    num_episodes += 1
                self._track_session_id(source, retrieved_session_ids)

        # Sort episodes chronologically for temporal/knowledge-update queries
        if is_temporal or instance.question_type == "knowledge-update":
            episode_evidence.sort(key=lambda x: x[0])

        # Build flat evidence list
        episode_texts = [text for _, text in episode_evidence]
        if is_temporal:
            evidence_texts = episode_texts + entity_evidence
        else:
            evidence_texts = self._interleave_evidence(entity_evidence, episode_texts)

        # Hypothesis = concatenation of top evidence (no LLM reader)
        hypothesis = self._build_hypothesis(evidence_texts)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stats.total_query_ms += elapsed_ms

        return QueryResult(
            question_id=instance.question_id,
            hypothesis=hypothesis,
            evidence=evidence_texts,
            evidence_scores=evidence_scores,
            retrieved_session_ids=retrieved_session_ids,
            latency_ms=elapsed_ms,
            num_entities=num_entities,
            num_episodes=num_episodes,
        )

    @staticmethod
    def _build_hypothesis(evidence: list[str]) -> str:
        """Build a hypothesis from top evidence (no LLM).

        Returns concatenated top evidence.  Evaluation uses embedding
        containment against gold answer, not text matching against
        this hypothesis.
        """
        if not evidence:
            return "I don't have enough information to answer this question."
        combined = " ".join(e.strip() for e in evidence[:3] if e.strip())
        if len(combined) > 1000:
            combined = combined[:997] + "..."
        return combined

    async def _enrich_episode_text(
        self,
        episode_id: str,
        truncated_content: str,
        source: str,
        group_id: str,
    ) -> str:
        """Fetch full episode content (bypass 500-char recall truncation).

        The recall pipeline truncates episodes to 500 chars which loses
        temporal details. Fetch the full content for benchmark evidence.
        """
        content = truncated_content

        # Try to fetch full content from graph store
        if episode_id and self._graph_store is not None:
            try:
                full_ep = await self._graph_store.get_episode_by_id(episode_id, group_id)
                if full_ep and full_ep.content:
                    content = full_ep.content
            except Exception:
                pass  # Fall back to truncated content

        # Add session date tag if not already present
        if source and source.startswith("lme:"):
            sid = source[4:]
            date = self._session_dates.get(sid, "")
            if date and not content.startswith("[Conversation from"):
                content = f"[Session from {date}]\n{content}"

        return content

    @staticmethod
    def _track_session_id(source: str, retrieved_session_ids: list[str]) -> None:
        """Extract and deduplicate session IDs from episode source."""
        if source and source.startswith("lme:"):
            sid = source[4:]
            if sid not in retrieved_session_ids:
                retrieved_session_ids.append(sid)

    @staticmethod
    def _interleave_evidence(entities: list[str], episodes: list[str]) -> list[str]:
        """Interleave entity and episode evidence by alternating."""
        result: list[str] = []
        ei, pi = 0, 0
        while ei < len(entities) or pi < len(episodes):
            if pi < len(episodes):
                result.append(episodes[pi])
                pi += 1
            if ei < len(entities):
                result.append(entities[ei])
                ei += 1
        return result

    def get_embed_fn(self):
        """Return the embedding function for use by the evaluator."""
        if self._search_index is not None:
            provider = getattr(self._search_index, "_provider", None)
            if provider and provider.dimension() > 0:
                return provider.embed
        return None

    async def cleanup_group(self, question_id: str) -> None:
        """Remove all data for a specific question's group_id."""
        if self._graph_store is None:
            return
        group_id = self._group_id_for(question_id)
        try:
            if hasattr(self._graph_store, "delete_group"):
                await self._graph_store.delete_group(group_id)
        except Exception:
            logger.debug("Could not clean up group %s (graph)", group_id, exc_info=True)
        try:
            if self._search_index is not None and hasattr(self._search_index, "delete_group"):
                await self._search_index.delete_group(group_id)
        except Exception:
            logger.debug("Could not clean up group %s (search)", group_id, exc_info=True)

    async def close(self) -> None:
        """Clean up resources and report embedding statistics."""
        # Report embedding stats if available
        if self._search_index is not None and hasattr(self._search_index, "embed_stats"):
            stats = self._search_index.embed_stats
            total_eps = stats.get("episodes_indexed", 0) + stats.get("episodes_failed", 0)
            logger.info(
                "LongMemEval embedding stats: "
                "episodes=%d/%d indexed (%.1f%%), "
                "chunks=%d/%d, entities=%d/%d, "
                "retries=%d, fallback=%d",
                stats.get("episodes_indexed", 0),
                total_eps,
                100.0 * stats.get("episodes_indexed", 0) / total_eps if total_eps else 0,
                stats.get("chunks_indexed", 0),
                stats.get("chunks_failed", 0),
                stats.get("entities_indexed", 0),
                stats.get("entities_failed", 0),
                stats.get("retries", 0),
                stats.get("fallback_used", 0),
            )
        if self._search_index is not None:
            try:
                await self._search_index.close()
            except Exception:
                pass
        if self._graph_store is not None:
            try:
                await self._graph_store.close()
            except Exception:
                pass
        self._graph_store = None
        self._activation_store = None
        self._search_index = None
        self._manager = None
        self._initialized = False


class _NoopExtractor:
    """No-op extractor for observe-only mode."""

    async def extract(self, text: str):
        from engram.extraction.extractor import ExtractionResult

        return ExtractionResult(entities=[], relationships=[])
