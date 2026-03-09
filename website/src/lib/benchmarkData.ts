import { useEffect, useState } from "react";

export type BaselineCard = {
  name: string;
  display_name: string;
  family: string;
  comparison_group: string | null;
  status: string | null;
  external_technology_label: string | null;
  accent: string | null;
  archetype: string | null;
  description: string | null;
  fairness_notes: string | null;
  known_limitations: string | null;
  why_included: string | null;
  scenario_pass_rate: number | null;
  false_recall_rate: number | null;
  temporal_correctness: number | null;
  negation_correctness: number | null;
  open_loop_recovery: number | null;
  prospective_trigger_rate: number | null;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
};

export type AblationSummary = {
  name: string;
  display_name: string;
  scenario_pass_rate: number | null;
  false_recall_rate: number | null;
};

export type ScenarioWinner = {
  scenario_id: string;
  title: string;
  winner: string | null;
  winner_score: number;
};

export type SpecOnlyBaseline = {
  baseline_id: string;
  display_name: string;
  comparison_group: string;
  status: string;
  family: string;
  accent: string;
  external_technology_label: string | null;
  archetype: string;
  description: string;
  fairness_notes: string;
  known_limitations: string;
  why_included: string;
};

export type BenchmarkSummary = {
  generated_at: string;
  track: string;
  mode: string;
  seeds: number[];
  headline: {
    engram_full_pass_rate: number | null;
    engram_full_false_recall: number | null;
    best_headline_competitor_pass_rate: number | null;
  };
  headline_baselines: BaselineCard[];
  control_baselines: BaselineCard[];
  primary_baselines: BaselineCard[];
  appendix_baselines: BaselineCard[];
  ablations: AblationSummary[];
  spec_only_baselines: SpecOnlyBaseline[];
  baseline_catalog: Record<string, SpecOnlyBaseline | BaselineCard>;
  scenario_winners: ScenarioWinner[];
};

const FALLBACK_METADATA: Record<string, Omit<BaselineCard, "scenario_pass_rate" | "false_recall_rate" | "temporal_correctness" | "negation_correctness" | "open_loop_recovery" | "prospective_trigger_rate" | "latency_p50_ms" | "latency_p95_ms">> = {
  engram_full: {
    name: "engram_full",
    display_name: "Engram Full",
    family: "engram",
    comparison_group: "headline",
    status: "measured",
    external_technology_label: null,
    accent: "#67e8f9",
    archetype: "Cue-first long-term memory with graph recall and prospective triggers.",
    description: "Full Engram memory loop using episodic capture, selective projection, cue recall, graph-aware retrieval, and prospective memory.",
    fairness_notes: "Real GraphManager path in lite mode; no fixture-only shortcuts.",
    known_limitations: "Hybrid vector search is optional and not enabled in the default run.",
    why_included: "Headline system under test.",
  },
  langgraph_store_memory: {
    name: "langgraph_store_memory",
    display_name: "LangGraph Store Memory",
    family: "external_proxy",
    comparison_group: "headline",
    status: "measured",
    external_technology_label: "LangGraph",
    accent: "#34d399",
    archetype: "Recent thread window plus rolling summary and durable store lookup.",
    description: "Proxy for LangGraph-style persistence with a recent thread window, deterministic summary, and a store of durable facts keyed by entities/topics.",
    fairness_notes: "Offline deterministic proxy, not a direct LangGraph SDK integration.",
    known_limitations: "No graph traversal, cue packets, or prospective trigger engine.",
    why_included: "Represents framework-native thread persistence plus store memory.",
  },
  mem0_style_memory: {
    name: "mem0_style_memory",
    display_name: "Mem0 Style Memory",
    family: "external_proxy",
    comparison_group: "headline",
    status: "measured",
    external_technology_label: "Mem0 / OpenMemory",
    accent: "#fbbf24",
    archetype: "Extracted durable memory objects with slot updates and compressed recall.",
    description: "Proxy for Mem0/OpenMemory-style memory that extracts, normalizes, updates, and retrieves compressed memory objects instead of transcript chunks.",
    fairness_notes: "Offline deterministic proxy using shared scenario extraction fixtures.",
    known_limitations: "No graph walk, cue memory, or proactive intention triggering.",
    why_included: "Represents current market memory-layer products for agents.",
  },
  graphiti_temporal_graph: {
    name: "graphiti_temporal_graph",
    display_name: "Graphiti Temporal Graph",
    family: "external_proxy",
    comparison_group: "headline",
    status: "measured",
    external_technology_label: "Zep / Graphiti",
    accent: "#a78bfa",
    archetype: "Temporal graph memory with validity-aware retrieval and 2-hop expansion.",
    description: "Proxy for Graphiti/Zep-style temporal graph memory with deterministic graph expansion, current-state filtering, and local lexical/vector fusion.",
    fairness_notes: "Offline deterministic proxy using local embeddings where available.",
    known_limitations: "No cue layer, no consolidation cycles, and no intention trigger model.",
    why_included: "Closest architectural peer on temporal and graph-aware memory.",
  },
  context_summary: {
    name: "context_summary",
    display_name: "Context + Summary",
    family: "control",
    comparison_group: "control",
    status: "measured",
    external_technology_label: null,
    accent: "#10b981",
    archetype: "Recent-turn context with a rolling deterministic summary.",
    description: "Strong context-window control without graph semantics or durable memory objects.",
    fairness_notes: "Honors the same probe budgets as every other measured baseline.",
    known_limitations: "No persistent graph, cue recall, or associative traversal.",
    why_included: "Strong practical non-graph control.",
  },
  markdown_canonical: {
    name: "markdown_canonical",
    display_name: "Markdown Canonical",
    family: "control",
    comparison_group: "control",
    status: "measured",
    external_technology_label: null,
    accent: "#f59e0b",
    archetype: "Structured latest-win notebook with lexical retrieval.",
    description: "Human-readable canonical notebook with deterministic sections for current facts, corrections, open loops, and intentions.",
    fairness_notes: "Compression and retrieval are deterministic and budget-bounded.",
    known_limitations: "No graph semantics, cue layer, or activation model.",
    why_included: "Fair markdown-file baseline instead of a toy notes file.",
  },
  hybrid_rag_temporal: {
    name: "hybrid_rag_temporal",
    display_name: "Hybrid RAG Temporal",
    family: "control",
    comparison_group: "control",
    status: "measured",
    external_technology_label: null,
    accent: "#8b5cf6",
    archetype: "Chunk retrieval with lexical and vector fusion plus temporal filtering.",
    description: "Raw-retrieval control with deterministic current-state filtering over vector and lexical matches.",
    fairness_notes: "Uses local embeddings for offline reproducibility.",
    known_limitations: "No cue recall, no intention triggers, and no graph traversal.",
    why_included: "Modern retrieval baseline without long-term memory architecture.",
  },
};

const FALLBACK_SPEC_ONLY: SpecOnlyBaseline[] = [
  {
    baseline_id: "letta",
    display_name: "Letta",
    comparison_group: "spec_only",
    status: "spec_only",
    family: "spec_only",
    accent: "#c084fc",
    external_technology_label: "Letta",
    archetype: "Pinned memory blocks and editable agent memory state.",
    description: "Stateful-agent memory system built around explicit core memory blocks and editable long-term memory.",
    fairness_notes: "Tracked on the page but not yet runnable in-suite.",
    known_limitations: "A fair proxy is less clear without direct product integration.",
    why_included: "Important public comparison target for stateful agents.",
  },
  {
    baseline_id: "llamaindex_memory",
    display_name: "LlamaIndex Memory",
    comparison_group: "spec_only",
    status: "spec_only",
    family: "spec_only",
    accent: "#fb7185",
    external_technology_label: "LlamaIndex",
    archetype: "Framework memory queue plus optional long-term extraction.",
    description: "Framework-native memory primitives combining short-term buffers and optional long-term extraction/retrieval.",
    fairness_notes: "Tracked on the page but not yet runnable in-suite.",
    known_limitations: "Less direct as a temporal-memory peer than Mem0 or Graphiti.",
    why_included: "Common framework comparison target.",
  },
  {
    baseline_id: "crewai_memory",
    display_name: "CrewAI Memory",
    comparison_group: "spec_only",
    status: "spec_only",
    family: "spec_only",
    accent: "#60a5fa",
    external_technology_label: "CrewAI",
    archetype: "Workflow-oriented short-term, long-term, entity, and contextual memory.",
    description: "Built-in memory stack for orchestrated multi-agent workflows.",
    fairness_notes: "Tracked on the page but not yet runnable in-suite.",
    known_limitations: "More workflow-centric than a direct temporal-memory architecture peer.",
    why_included: "Relevant framework comparison for orchestration-heavy agents.",
  },
];

function withFallbackMetrics(id: string): BaselineCard {
  const meta = FALLBACK_METADATA[id];
  return {
    ...meta,
    scenario_pass_rate: null,
    false_recall_rate: null,
    temporal_correctness: null,
    negation_correctness: null,
    open_loop_recovery: null,
    prospective_trigger_rate: null,
    latency_p50_ms: null,
    latency_p95_ms: null,
  };
}

export function fallbackBenchmarkSummary(): BenchmarkSummary {
  const headlineBaselines = [
    withFallbackMetrics("engram_full"),
    withFallbackMetrics("langgraph_store_memory"),
    withFallbackMetrics("mem0_style_memory"),
    withFallbackMetrics("graphiti_temporal_graph"),
  ];
  const controlBaselines = [
    withFallbackMetrics("context_summary"),
    withFallbackMetrics("markdown_canonical"),
    withFallbackMetrics("hybrid_rag_temporal"),
  ];
  return {
    generated_at: "",
    track: "showcase",
    mode: "full",
    seeds: [],
    headline: {
      engram_full_pass_rate: null,
      engram_full_false_recall: null,
      best_headline_competitor_pass_rate: null,
    },
    headline_baselines: headlineBaselines,
    control_baselines: controlBaselines,
    primary_baselines: [...headlineBaselines, ...controlBaselines],
    appendix_baselines: [],
    ablations: [],
    spec_only_baselines: FALLBACK_SPEC_ONLY,
    baseline_catalog: Object.fromEntries(
      Object.keys(FALLBACK_METADATA).map((id) => [id, withFallbackMetrics(id)]),
    ),
    scenario_winners: [],
  };
}

let cachedSummary: BenchmarkSummary | null = null;
let cachedPromise: Promise<BenchmarkSummary> | null = null;

async function loadBenchmarkSummary(): Promise<BenchmarkSummary> {
  if (cachedSummary) {
    return cachedSummary;
  }
  if (!cachedPromise) {
    cachedPromise = fetch("/benchmarks/latest.json").then(async (response) => {
      if (!response.ok) {
        throw new Error("benchmark summary unavailable");
      }
      const json = (await response.json()) as BenchmarkSummary;
      cachedSummary = json;
      return json;
    });
  }
  return cachedPromise;
}

export function useBenchmarkSummary() {
  const [data, setData] = useState<BenchmarkSummary | null>(cachedSummary);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    loadBenchmarkSummary()
      .then((summary) => {
        if (!cancelled) {
          setData(summary);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError("No benchmark export found yet.");
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return { data, error };
}
