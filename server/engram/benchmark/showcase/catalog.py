"""Baseline catalog for the showcase benchmark."""

from __future__ import annotations

from engram.benchmark.showcase.models import BaselineCatalogEntry

DEFAULT_HEADLINE_BASELINES = [
    "engram_full",
    "langgraph_store_memory",
    "mem0_style_memory",
    "graphiti_temporal_graph",
]
DEFAULT_CONTROL_BASELINES = [
    "context_summary",
    "markdown_canonical",
    "hybrid_rag_temporal",
]
DEFAULT_APPENDIX_BASELINES = [
    "context_window",
    "markdown_memory",
    "vector_rag",
]
DEFAULT_ABLATION_BASELINES = [
    "engram_no_cues",
    "engram_search_only",
]
DEFAULT_SPEC_ONLY_BASELINES = [
    "letta",
    "llamaindex_memory",
    "crewai_memory",
]

BASELINE_CATALOG: dict[str, BaselineCatalogEntry] = {
    "engram_full": BaselineCatalogEntry(
        baseline_id="engram_full",
        display_name="Engram Full",
        comparison_group="headline",
        status="measured",
        family="engram",
        accent="#67e8f9",
        archetype="Cue-first long-term memory with graph recall and prospective triggers.",
        description=(
            "Full Engram memory loop using episodic capture, selective projection, cue recall, "
            "graph-aware retrieval, and prospective memory."
        ),
        fairness_notes="Real GraphManager path in lite mode; no fixture-only shortcuts.",
        known_limitations="Hybrid vector search is optional and not enabled in the default run.",
        why_included="Headline system under test.",
    ),
    "langgraph_store_memory": BaselineCatalogEntry(
        baseline_id="langgraph_store_memory",
        display_name="LangGraph Store Memory",
        comparison_group="headline",
        status="measured",
        family="external_proxy",
        accent="#34d399",
        external_technology_label="LangGraph",
        archetype="Recent thread window plus rolling summary and durable store lookup.",
        description=(
            "Proxy for LangGraph-style persistence with a recent thread window, deterministic "
            "summary, and a store of durable facts keyed by entities/topics."
        ),
        fairness_notes="Offline deterministic proxy, not a direct LangGraph SDK integration.",
        known_limitations="No graph traversal, cue packets, or prospective trigger engine.",
        why_included="Represents framework-native thread persistence plus store memory.",
    ),
    "mem0_style_memory": BaselineCatalogEntry(
        baseline_id="mem0_style_memory",
        display_name="Mem0 Style Memory",
        comparison_group="headline",
        status="measured",
        family="external_proxy",
        accent="#fbbf24",
        external_technology_label="Mem0 / OpenMemory",
        archetype="Extracted durable memory objects with slot updates and compressed recall.",
        description=(
            "Proxy for Mem0/OpenMemory-style memory that extracts, normalizes, updates, and "
            "retrieves compressed memory objects instead of transcript chunks."
        ),
        fairness_notes="Offline deterministic proxy using shared scenario extraction fixtures.",
        known_limitations="No graph walk, cue memory, or proactive intention triggering.",
        why_included="Represents current market memory-layer products for agents.",
    ),
    "graphiti_temporal_graph": BaselineCatalogEntry(
        baseline_id="graphiti_temporal_graph",
        display_name="Graphiti Temporal Graph",
        comparison_group="headline",
        status="measured",
        family="external_proxy",
        accent="#a78bfa",
        external_technology_label="Zep / Graphiti",
        archetype="Temporal graph memory with validity-aware retrieval and 2-hop expansion.",
        description=(
            "Proxy for Graphiti/Zep-style temporal graph memory with deterministic graph expansion, "
            "current-state filtering, and local lexical/vector fusion."
        ),
        fairness_notes="Offline deterministic proxy using local embeddings where available.",
        known_limitations="No cue layer, no consolidation cycles, and no intention trigger model.",
        why_included="Closest architectural peer on temporal and graph-aware memory.",
    ),
    "context_summary": BaselineCatalogEntry(
        baseline_id="context_summary",
        display_name="Context + Summary",
        comparison_group="control",
        status="measured",
        family="control",
        accent="#10b981",
        archetype="Recent-turn context with a rolling deterministic summary.",
        description="Strong context-window control without graph semantics or durable memory objects.",
        fairness_notes="Honors the same probe budgets as every other measured baseline.",
        known_limitations="No persistent graph, cue recall, or associative traversal.",
        why_included="Strong practical non-graph control.",
    ),
    "markdown_canonical": BaselineCatalogEntry(
        baseline_id="markdown_canonical",
        display_name="Markdown Canonical",
        comparison_group="control",
        status="measured",
        family="control",
        accent="#f59e0b",
        archetype="Structured latest-win notebook with lexical retrieval.",
        description="Human-readable canonical notebook with deterministic sections for current facts, corrections, open loops, and intentions.",
        fairness_notes="Compression and retrieval are deterministic and budget-bounded.",
        known_limitations="No graph semantics, cue layer, or activation model.",
        why_included="Fair markdown-file baseline instead of a toy notes file.",
    ),
    "hybrid_rag_temporal": BaselineCatalogEntry(
        baseline_id="hybrid_rag_temporal",
        display_name="Hybrid RAG Temporal",
        comparison_group="control",
        status="measured",
        family="control",
        accent="#8b5cf6",
        archetype="Chunk retrieval with lexical and vector fusion plus temporal filtering.",
        description="Raw-retrieval control with deterministic current-state filtering over vector and lexical matches.",
        fairness_notes="Uses local embeddings for offline reproducibility.",
        known_limitations="No cue recall, no intention triggers, and no graph traversal.",
        why_included="Modern retrieval baseline without long-term memory architecture.",
    ),
    "context_window": BaselineCatalogEntry(
        baseline_id="context_window",
        display_name="Context Window",
        comparison_group="appendix",
        status="measured",
        family="alternative",
        accent="#38bdf8",
        archetype="Recent-turn raw history only.",
        description="Simple working-memory baseline constrained to the most recent history budget.",
        fairness_notes="Included as an appendix control, not the headline comparison.",
        known_limitations="No durable memory outside the visible recent window.",
        why_included="Naive prompt-only baseline.",
    ),
    "markdown_memory": BaselineCatalogEntry(
        baseline_id="markdown_memory",
        display_name="Markdown Memory",
        comparison_group="appendix",
        status="measured",
        family="alternative",
        accent="#f97316",
        archetype="Flat timestamped notebook with lexical retrieval.",
        description="Simple append-only markdown notes retrieved by keyword overlap.",
        fairness_notes="Included as an appendix control, not a canonical notebook competitor.",
        known_limitations="No latest-win correction semantics or structure.",
        why_included="Naive file-memory baseline.",
    ),
    "vector_rag": BaselineCatalogEntry(
        baseline_id="vector_rag",
        display_name="Vector RAG",
        comparison_group="appendix",
        status="measured",
        family="alternative",
        accent="#818cf8",
        archetype="Raw chunk retrieval with local embeddings.",
        description="Simple vector/lexical retrieval over raw note chunks with no higher-level memory model.",
        fairness_notes="Uses local embeddings only; deterministic when available.",
        known_limitations="No temporal invalidation, graph expansion, or prospective behavior.",
        why_included="Naive RAG baseline.",
    ),
    "engram_no_cues": BaselineCatalogEntry(
        baseline_id="engram_no_cues",
        display_name="Engram No Cues",
        comparison_group="ablation",
        status="measured",
        family="ablation",
        accent="#22d3ee",
        archetype="Engram without cue and projection pathways.",
        description="Strict Engram ablation with cue recall and targeted projection disabled.",
        fairness_notes="Same GraphManager path with cue-related features turned off.",
        known_limitations="Does not represent an external system; attribution-only baseline.",
        why_included="Isolates cue-driven latent recall value.",
    ),
    "engram_search_only": BaselineCatalogEntry(
        baseline_id="engram_search_only",
        display_name="Engram Search Only",
        comparison_group="ablation",
        status="measured",
        family="ablation",
        accent="#06b6d4",
        archetype="Engram entity search without activation or prospective features.",
        description="Strict Engram ablation that keeps searchable graph memory but disables activation, cue recall, and prospective retrieval.",
        fairness_notes="Same underlying memory store with retrieval features disabled.",
        known_limitations="Attribution-only baseline rather than a market comparison.",
        why_included="Shows where search alone stops being sufficient.",
    ),
    "engram_full_hybrid": BaselineCatalogEntry(
        baseline_id="engram_full_hybrid",
        display_name="Engram Full Hybrid",
        comparison_group="appendix",
        status="measured",
        family="engram",
        accent="#60a5fa",
        archetype="Engram with hybrid search enabled.",
        description="Optional Engram variant that uses hybrid lexical/vector search inside the same benchmark harness.",
        fairness_notes="Only available when a compatible embedding provider is configured.",
        known_limitations="Optional appendix variant, not part of the default public comparison.",
        why_included="Lets the suite show vector-backed Engram behavior when configured.",
    ),
    "letta": BaselineCatalogEntry(
        baseline_id="letta",
        display_name="Letta",
        comparison_group="spec_only",
        status="spec_only",
        family="spec_only",
        accent="#c084fc",
        external_technology_label="Letta",
        archetype="Pinned memory blocks and editable agent memory state.",
        description="Stateful-agent memory system built around explicit core memory blocks and editable long-term memory.",
        fairness_notes="Tracked on the page but not yet runnable in-suite.",
        known_limitations="A fair proxy is less clear without direct product integration.",
        why_included="Important public comparison target for stateful agents.",
    ),
    "llamaindex_memory": BaselineCatalogEntry(
        baseline_id="llamaindex_memory",
        display_name="LlamaIndex Memory",
        comparison_group="spec_only",
        status="spec_only",
        family="spec_only",
        accent="#fb7185",
        external_technology_label="LlamaIndex",
        archetype="Framework memory queue plus optional long-term extraction.",
        description="Framework-native memory primitives combining short-term buffers and optional long-term extraction/retrieval.",
        fairness_notes="Tracked on the page but not yet runnable in-suite.",
        known_limitations="Less direct as a temporal-memory peer than Mem0 or Graphiti.",
        why_included="Common framework comparison target.",
    ),
    "crewai_memory": BaselineCatalogEntry(
        baseline_id="crewai_memory",
        display_name="CrewAI Memory",
        comparison_group="spec_only",
        status="spec_only",
        family="spec_only",
        accent="#60a5fa",
        external_technology_label="CrewAI",
        archetype="Workflow-oriented short-term, long-term, entity, and contextual memory.",
        description="Built-in memory stack for orchestrated multi-agent workflows.",
        fairness_notes="Tracked on the page but not yet runnable in-suite.",
        known_limitations="More workflow-centric than a direct temporal-memory architecture peer.",
        why_included="Relevant framework comparison for orchestration-heavy agents.",
    ),
}

DEFAULT_PRIMARY_BASELINES = DEFAULT_HEADLINE_BASELINES + DEFAULT_CONTROL_BASELINES


def baseline_entry(baseline_id: str) -> BaselineCatalogEntry:
    return BASELINE_CATALOG[baseline_id]


def display_name(baseline_id: str) -> str:
    entry = BASELINE_CATALOG.get(baseline_id)
    if entry is not None:
        return entry.display_name
    return baseline_id.replace("_", " ").title()
