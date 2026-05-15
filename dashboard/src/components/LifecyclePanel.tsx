import { useEffect, useMemo } from "react";
import { useEngramStore } from "../store";
import type {
  DashboardView,
  Episode,
  LifecycleStageKey,
  LifecycleStageStatus,
} from "../store/types";

const EMPTY_CUE_METRICS = {
  cueCount: 0,
  episodesWithoutCues: 0,
  cueCoverage: 0,
  cueHitCount: 0,
  cueHitEpisodeCount: 0,
  cueHitEpisodeRate: 0,
  cueSurfacedCount: 0,
  cueSelectedCount: 0,
  cueUsedCount: 0,
  cueNearMissCount: 0,
  avgPolicyScore: 0,
  avgProjectionAttempts: 0,
  projectedCueCount: 0,
  cueToProjectionConversionRate: 0,
};

const EMPTY_PROJECTION_METRICS = {
  stateCounts: {
    queued: 0,
    cued: 0,
    cueOnly: 0,
    scheduled: 0,
    projecting: 0,
    projected: 0,
    failed: 0,
    deadLetter: 0,
  },
  attemptedEpisodeCount: 0,
  totalAttempts: 0,
  failureCount: 0,
  deadLetterCount: 0,
  failureRate: 0,
  avgProcessingDurationMs: 0,
  avgTimeToProjectionMs: 0,
  yield: {
    linkedEntityCount: 0,
    relationshipCount: 0,
    avgLinkedEntitiesPerProjectedEpisode: 0,
    avgRelationshipsPerProjectedEpisode: 0,
  },
};

const EMPTY_INTENTION_SUMMARY = {
  activeCount: 0,
  refreshContextCount: 0,
  afterConsolidationCount: 0,
  pinnedResultCount: 0,
  needsRefreshCount: 0,
  latestRefreshedAt: null,
};

const STAGES = [
  { key: "capture", label: "Capture", accent: "#22d3ee" },
  { key: "cue", label: "Cue", accent: "#facc15" },
  { key: "project", label: "Project", accent: "#818cf8" },
  { key: "recall", label: "Recall", accent: "#34d399" },
  { key: "consolidate", label: "Consolidate", accent: "#f97316" },
] as const;

const STAGE_DRILLDOWNS: Record<
  LifecycleStageKey,
  { view: DashboardView; command: string }
> = {
  capture: { view: "feed", command: "Open Feed" },
  cue: { view: "stats", command: "Open Stats" },
  project: { view: "stats", command: "Open Stats" },
  recall: { view: "knowledge", command: "Open Knowledge" },
  consolidate: { view: "consolidation", command: "Open Consolidation" },
};

function formatRate(value: number) {
  if (!Number.isFinite(value)) return "0%";
  return `${(value * 100).toFixed(value >= 0.995 || value <= 0.005 ? 0 : 1)}%`;
}

function formatNumber(value: number, digits = 1) {
  if (!Number.isFinite(value)) return "0";
  if (Math.abs(value - Math.round(value)) < 0.01) return Math.round(value).toLocaleString();
  return value.toFixed(digits);
}

function formatAge(iso: string | null | undefined) {
  if (!iso) return "none";
  const elapsedMs = Date.now() - new Date(iso).getTime();
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0) return "now";
  const minutes = Math.floor(elapsedMs / 60_000);
  if (minutes < 1) return "now";
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

function latestEpisode(episodes: Episode[]) {
  return episodes.reduce<Episode | null>((latest, episode) => {
    if (!latest) return episode;
    return new Date(episode.createdAt).getTime() > new Date(latest.createdAt).getTime()
      ? episode
      : latest;
  }, null);
}

function stageHealth(status: LifecycleStageStatus) {
  if (status === "active") return { label: "active", color: "var(--accent)" };
  if (status === "attention") return { label: "attention", color: "var(--danger)" };
  return { label: "ready", color: "var(--success)" };
}

function cycleIssueText(
  cycle:
    | {
        error?: unknown;
        phase_issue?: unknown;
        phases?: Array<{ phase?: unknown; status?: unknown; error?: unknown }>;
      }
    | null
    | undefined,
) {
  const error = cycle?.error;
  if (typeof error === "string" && error.trim()) return error;
  const phaseIssueText = cycle?.phase_issue;
  if (typeof phaseIssueText === "string" && phaseIssueText.trim()) {
    return phaseIssueText;
  }
  const phaseIssue = cycle?.phases?.find((phase) => {
    if (phase.status === "error") return true;
    const phaseError = phase.error;
    return typeof phaseError === "string" && phaseError.trim();
  });
  if (!phaseIssue) return null;
  const phaseName = typeof phaseIssue.phase === "string" ? phaseIssue.phase : "phase";
  const phaseError = phaseIssue.error;
  if (typeof phaseError === "string" && phaseError.trim()) {
    return `${phaseName}: ${phaseError}`;
  }
  return `${phaseName}: phase error`;
}

function cycleHasIssue(
      cycle:
    | {
        status?: unknown;
        error?: unknown;
        phase_issue?: unknown;
        phases?: Array<{ phase?: unknown; status?: unknown; error?: unknown }>;
      }
    | null
    | undefined,
) {
  return cycle?.status === "failed" || cycleIssueText(cycle) !== null;
}

function StageCard({
  label,
  accent,
  status,
  primary,
  secondary,
  metrics,
  drilldownLabel,
  onClick,
}: {
  label: string;
  accent: string;
  status: LifecycleStageStatus;
  primary: string;
  secondary: string;
  metrics: ReadonlyArray<{ label: string; value: string }>;
  drilldownLabel: string;
  onClick: () => void;
}) {
  const health = stageHealth(status);
  return (
    <button
      type="button"
      aria-label={`Open ${label} drilldown`}
      className="card"
      onClick={onClick}
      style={{
        minWidth: 0,
        padding: 16,
        borderColor: `${accent}22`,
        background: `linear-gradient(180deg, ${accent}10, rgba(8, 10, 18, 0.72))`,
        display: "flex",
        flexDirection: "column",
        gap: 12,
        color: "inherit",
        cursor: "pointer",
        fontFamily: "inherit",
        textAlign: "left",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
          <span
            aria-hidden="true"
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: accent,
              boxShadow: `0 0 12px ${accent}66`,
              flexShrink: 0,
            }}
          />
          <div
            style={{
              fontSize: 15,
              fontWeight: 600,
              color: "var(--text-primary)",
              lineHeight: 1.1,
            }}
          >
            {label}
          </div>
        </div>
        <span className="label" style={{ color: health.color, fontSize: 9 }}>
          {health.label}
        </span>
      </div>

      <div>
        <div
          className="mono tabular-nums"
          style={{ fontSize: 30, lineHeight: 1, color: "#fff", marginBottom: 6 }}
        >
          {primary}
        </div>
        <div
          style={{
            fontSize: 12,
            color: "var(--text-secondary)",
            minHeight: 36,
            overflowWrap: "anywhere",
          }}
        >
          {secondary}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 8 }}>
        {metrics.map((metric) => (
          <div
            key={metric.label}
            style={{
              minWidth: 0,
              borderTop: "1px solid var(--border)",
              paddingTop: 8,
            }}
          >
            <div className="label" style={{ fontSize: 8, marginBottom: 3 }}>
              {metric.label}
            </div>
            <div
              className="mono tabular-nums"
              style={{ fontSize: 14, color: "var(--text-primary)", overflowWrap: "anywhere" }}
            >
              {metric.value}
            </div>
          </div>
        ))}
      </div>
      <div className="label" style={{ color: accent, fontSize: 9, marginTop: "auto" }}>
        {drilldownLabel}
      </div>
    </button>
  );
}

function PipelineRail() {
  return (
    <div
      aria-label="Brain lifecycle"
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
        gap: 0,
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        overflow: "hidden",
        background: "rgba(255,255,255,0.02)",
      }}
    >
      {STAGES.map((stage, index) => (
        <div
          key={stage.key}
          style={{
            minWidth: 0,
            padding: "10px 12px",
            borderLeft: index === 0 ? "none" : "1px solid var(--border)",
            background: `linear-gradient(180deg, ${stage.accent}0f, transparent)`,
          }}
        >
          <div className="label" style={{ color: stage.accent, fontSize: 9 }}>
            {String(index + 1).padStart(2, "0")}
          </div>
          <div style={{ fontSize: 13, color: "var(--text-primary)", fontWeight: 500 }}>
            {stage.label}
          </div>
        </div>
      ))}
    </div>
  );
}

function RecentQueue({ episodes }: { episodes: Episode[] }) {
  const recent = episodes.slice(0, 5);
  return (
    <section
      className="card"
      style={{
        padding: 16,
        minWidth: 0,
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <div className="label">Recent Memory States</div>
      {recent.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {recent.map((episode) => (
            <div
              key={episode.episodeId}
              style={{
                display: "grid",
                gridTemplateColumns: "minmax(0, 1fr) auto auto",
                gap: 10,
                alignItems: "center",
                padding: "7px 0",
                borderTop: "1px solid var(--border-subtle)",
              }}
            >
              <div
                style={{
                  minWidth: 0,
                  color: "var(--text-secondary)",
                  fontSize: 12,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {episode.content}
              </div>
              <span className="mono" style={{ fontSize: 10, color: "var(--text-muted)" }}>
                {episode.status}
              </span>
              <span className="mono" style={{ fontSize: 10, color: "var(--text-muted)" }}>
                {episode.projectionState ?? episode.cue?.projectionState ?? "uncued"}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ color: "var(--text-muted)", fontSize: 12 }}>No episodes loaded</div>
      )}
    </section>
  );
}

type ActiveRecallEntity = {
  id: string;
  name: string;
  entityType: string;
  activation: number;
};

function TopActivated({ entities }: { entities: ActiveRecallEntity[] }) {
  const top = entities.slice(0, 5);
  return (
    <section
      className="card"
      style={{
        padding: 16,
        minWidth: 0,
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <div className="label">Active Recall Context</div>
      {top.length > 0 ? (
        top.map((entity) => (
          <div
            key={entity.id}
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1fr) auto",
              gap: 10,
              alignItems: "center",
              padding: "7px 0",
              borderTop: "1px solid var(--border-subtle)",
            }}
          >
            <div style={{ minWidth: 0 }}>
              <div
                style={{
                  color: "var(--text-primary)",
                  fontSize: 12,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {entity.name}
              </div>
              <div className="label" style={{ fontSize: 8, marginTop: 2 }}>
                {entity.entityType}
              </div>
            </div>
            <span className="mono tabular-nums" style={{ fontSize: 11, color: "#34d399" }}>
              {formatNumber(entity.activation, 2)}
            </span>
          </div>
        ))
      ) : (
        <div style={{ color: "var(--text-muted)", fontSize: 12 }}>No active entities loaded</div>
      )}
    </section>
  );
}

export function LifecyclePanel() {
  const lifecycleSummary = useEngramStore((s) => s.lifecycleSummary);
  const isLoadingLifecycleSummary = useEngramStore((s) => s.isLoadingLifecycleSummary);
  const loadLifecycleSummary = useEngramStore((s) => s.loadLifecycleSummary);
  const stats = useEngramStore((s) => s.stats);
  const isLoadingStats = useEngramStore((s) => s.isLoadingStats);
  const loadStats = useEngramStore((s) => s.loadStats);
  const episodes = useEngramStore((s) => s.episodes);
  const loadEpisodes = useEngramStore((s) => s.loadEpisodes);
  const knowledgeResults = useEngramStore((s) => s.knowledgeResults);
  const activationLeaderboard = useEngramStore((s) => s.activationLeaderboard);
  const cycles = useEngramStore((s) => s.cycles);
  const isRunning = useEngramStore((s) => s.isRunning);
  const schedulerActive = useEngramStore((s) => s.schedulerActive);
  const pressure = useEngramStore((s) => s.pressure);
  const loadStatus = useEngramStore((s) => s.loadStatus);
  const loadCycles = useEngramStore((s) => s.loadCycles);
  const selectCycle = useEngramStore((s) => s.selectCycle);
  const setCurrentView = useEngramStore((s) => s.setCurrentView);
  const setLifecycleDrilldownStage = useEngramStore((s) => s.setLifecycleDrilldownStage);

  useEffect(() => {
    if (!lifecycleSummary && !isLoadingLifecycleSummary) {
      void loadLifecycleSummary();
    }
    if (!lifecycleSummary) {
      if (!stats && !isLoadingStats) void loadStats();
      if (episodes.length === 0) void loadEpisodes();
      void loadStatus();
      void loadCycles();
    }
  }, [
    episodes.length,
    isLoadingLifecycleSummary,
    isLoadingStats,
    lifecycleSummary,
    loadCycles,
    loadEpisodes,
    loadLifecycleSummary,
    loadStats,
    loadStatus,
    stats,
  ]);

  const summary = useMemo(() => {
    if (lifecycleSummary) {
      const latest = lifecycleSummary.capture.latestEpisode;
      const latestCycle = lifecycleSummary.consolidate.latestCycle;
      const latestCycleError = cycleIssueText(latestCycle);
      const completedPhases =
        latestCycle?.phases.filter((phase) => phase.status === "success").length ?? 0;
      const projectFailures =
        lifecycleSummary.project.failedCount + lifecycleSummary.project.deadLetterCount;
      const cueRecallResults = knowledgeResults.filter(
        (result) => result.resultType === "cue_episode",
      ).length;
      const recallIntentions = lifecycleSummary.recall.intentions ?? EMPTY_INTENTION_SUMMARY;

      return {
        capture: {
          primary: lifecycleSummary.capture.episodeCount.toLocaleString(),
          secondary: latest
            ? `${latest.source} capture ${formatAge(latest.createdAt)} ago`
            : "No captured episodes loaded",
          status: lifecycleSummary.capture.status,
          metrics: [
            { label: "queued", value: lifecycleSummary.capture.activeCount.toLocaleString() },
            { label: "latest", value: latest?.status ?? "none" },
          ],
        },
        cue: {
          primary: formatRate(lifecycleSummary.cue.coverage),
          secondary: `${lifecycleSummary.cue.cueCount.toLocaleString()} cueable episodes, ${lifecycleSummary.cue.episodesWithoutCues.toLocaleString()} without cues`,
          status: lifecycleSummary.cue.status,
          metrics: [
            { label: "hits", value: lifecycleSummary.cue.hitCount.toLocaleString() },
            { label: "used", value: lifecycleSummary.cue.usedCount.toLocaleString() },
          ],
        },
        project: {
          primary: lifecycleSummary.project.projectedCount.toLocaleString(),
          secondary: `${lifecycleSummary.project.activeCount.toLocaleString()} active projection states, ${projectFailures.toLocaleString()} failed or dead-lettered`,
          status: lifecycleSummary.project.status,
          metrics: [
            {
              label: "conversion",
              value: formatRate(lifecycleSummary.cue.projectionConversionRate),
            },
            { label: "failure", value: formatRate(lifecycleSummary.project.failureRate) },
          ],
        },
        recall: {
          primary: (
            lifecycleSummary.recall.activeEntityCount ||
            lifecycleSummary.recall.topActivated.length
          ).toLocaleString(),
          secondary: `${knowledgeResults.length.toLocaleString()} current recall results, ${recallIntentions.activeCount.toLocaleString()} active intentions`,
          status:
            knowledgeResults.length > 0 || recallIntentions.activeCount > 0
              ? "active"
              : lifecycleSummary.recall.status,
          metrics: [
            { label: "top score", value: formatNumber(lifecycleSummary.recall.topScore, 2) },
            { label: "pinned", value: recallIntentions.pinnedResultCount.toLocaleString() },
            { label: "cue recalls", value: cueRecallResults.toLocaleString() },
          ],
        },
        consolidate: {
          primary: latestCycle
            ? latestCycle.status
            : lifecycleSummary.consolidate.isRunning
              ? "running"
              : "idle",
          secondary: latestCycle
            ? latestCycleError
              ? `${completedPhases}/${latestCycle.phases.length} phases on latest ${latestCycle.trigger} cycle; ${latestCycleError}`
              : `${completedPhases}/${latestCycle.phases.length} phases on latest ${latestCycle.trigger} cycle`
            : lifecycleSummary.consolidate.schedulerActive
              ? "Scheduler active"
              : "No consolidation cycles loaded",
          status: lifecycleSummary.consolidate.status,
          metrics: [
            { label: "cycles", value: lifecycleSummary.consolidate.cycleCount.toLocaleString() },
            {
              label: "pressure",
              value: lifecycleSummary.consolidate.pressure
                ? formatNumber(lifecycleSummary.consolidate.pressure.value, 2)
                : "0",
            },
          ],
        },
      } as const;
    }

    const cueMetrics = stats?.cueMetrics ?? EMPTY_CUE_METRICS;
    const projectionMetrics = stats?.projectionMetrics ?? EMPTY_PROJECTION_METRICS;
    const latest = latestEpisode(episodes);
    const totalEpisodes = stats?.totalEpisodes ?? episodes.length;
    const queuedStatuses = new Set(["queued", "pending", "processing", "extracting"]);
    const capturedActive = episodes.filter((episode) => queuedStatuses.has(episode.status)).length;
    const projectActive =
      projectionMetrics.stateCounts.queued +
      projectionMetrics.stateCounts.cued +
      projectionMetrics.stateCounts.scheduled +
      projectionMetrics.stateCounts.projecting;
    const projectFailures =
      projectionMetrics.stateCounts.failed + projectionMetrics.stateCounts.deadLetter;
    const cueRecallResults = knowledgeResults.filter(
      (result) => result.resultType === "cue_episode",
    ).length;
    const latestCycle = cycles[0] ?? null;
    const latestCycleError = cycleIssueText(latestCycle);
    const completedPhases = latestCycle?.phases.filter((phase) => phase.status === "success").length ?? 0;

    return {
      capture: {
        primary: totalEpisodes.toLocaleString(),
        secondary: latest
          ? `${latest.source} capture ${formatAge(latest.createdAt)} ago`
          : "No captured episodes loaded",
        status: capturedActive > 0 ? "active" : "ready",
        metrics: [
          { label: "queued", value: capturedActive.toLocaleString() },
          { label: "latest", value: latest?.status ?? "none" },
        ],
      },
      cue: {
        primary: formatRate(cueMetrics.cueCoverage),
        secondary: `${cueMetrics.cueCount.toLocaleString()} cueable episodes, ${cueMetrics.episodesWithoutCues.toLocaleString()} without cues`,
        status: cueMetrics.episodesWithoutCues > 0 ? "attention" : "ready",
        metrics: [
          { label: "hits", value: cueMetrics.cueHitCount.toLocaleString() },
          { label: "used", value: cueMetrics.cueUsedCount.toLocaleString() },
        ],
      },
      project: {
        primary: projectionMetrics.stateCounts.projected.toLocaleString(),
        secondary: `${projectActive.toLocaleString()} active projection states, ${projectFailures.toLocaleString()} failed or dead-lettered`,
        status: projectFailures > 0 ? "attention" : projectActive > 0 ? "active" : "ready",
        metrics: [
          { label: "conversion", value: formatRate(cueMetrics.cueToProjectionConversionRate) },
          { label: "failure", value: formatRate(projectionMetrics.failureRate) },
        ],
      },
      recall: {
        primary: (stats?.topActivated.length ?? activationLeaderboard.length).toLocaleString(),
        secondary: `${knowledgeResults.length.toLocaleString()} current recall results, ${cueRecallResults.toLocaleString()} from cues`,
        status: knowledgeResults.length > 0 || activationLeaderboard.length > 0 ? "active" : "ready",
        metrics: [
          { label: "top score", value: formatNumber(stats?.topActivated[0]?.activation ?? activationLeaderboard[0]?.currentActivation ?? 0, 2) },
          { label: "cue recalls", value: cueRecallResults.toLocaleString() },
        ],
      },
      consolidate: {
        primary: latestCycle ? latestCycle.status : isRunning ? "running" : "idle",
        secondary: latestCycle
          ? latestCycleError
            ? `${completedPhases}/${latestCycle.phases.length} phases on latest ${latestCycle.trigger} cycle; ${latestCycleError}`
            : `${completedPhases}/${latestCycle.phases.length} phases on latest ${latestCycle.trigger} cycle`
          : schedulerActive
            ? "Scheduler active"
            : "No consolidation cycles loaded",
        status: cycleHasIssue(latestCycle) ? "attention" : isRunning ? "active" : "ready",
        metrics: [
          { label: "cycles", value: cycles.length.toLocaleString() },
          { label: "pressure", value: pressure ? formatNumber(pressure.value, 2) : "0" },
        ],
      },
    } as const;
  }, [
    activationLeaderboard,
    cycles,
    episodes,
    isRunning,
    knowledgeResults,
    lifecycleSummary,
    pressure,
    schedulerActive,
    stats,
  ]);

  const activeRecallEntities = useMemo<ActiveRecallEntity[]>(() => {
    if (lifecycleSummary) return lifecycleSummary.recall.topActivated;
    if (stats) return stats.topActivated;
    return activationLeaderboard.map((item) => ({
      id: item.entityId,
      name: item.name,
      entityType: item.entityType,
      activation: item.currentActivation,
    }));
  }, [activationLeaderboard, lifecycleSummary, stats]);

  const recentEpisodes = lifecycleSummary?.recentEpisodes ?? episodes;
  const totals = lifecycleSummary?.totals;

  const stageData = [
    { ...STAGES[0], ...summary.capture },
    { ...STAGES[1], ...summary.cue },
    { ...STAGES[2], ...summary.project },
    { ...STAGES[3], ...summary.recall },
    { ...STAGES[4], ...summary.consolidate },
  ];

  const openStageDrilldown = (stageKey: LifecycleStageKey) => {
    const drilldown = STAGE_DRILLDOWNS[stageKey];
    if (stageKey === "consolidate") {
      const cycleId = lifecycleSummary?.consolidate.latestCycle?.id ?? cycles[0]?.id ?? null;
      if (cycleId) selectCycle(cycleId);
    }
    setCurrentView(drilldown.view);
    setLifecycleDrilldownStage(stageKey);
  };

  return (
    <div
      className="animate-fade-in"
      style={{
        height: "100%",
        overflowY: "auto",
        padding: "10px 14px 20px",
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <header
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: 12,
          alignItems: "stretch",
        }}
      >
        <section
          className="card"
          style={{
            padding: 18,
            minWidth: 0,
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            gap: 16,
          }}
        >
          <div>
            <div className="label" style={{ marginBottom: 8 }}>
              Brain Runtime
            </div>
            <h1
              style={{
                fontSize: 24,
                lineHeight: 1.1,
                fontWeight: 600,
                color: "#fff",
                marginBottom: 8,
              }}
            >
              One continuous memory loop
            </h1>
            <div style={{ color: "var(--text-secondary)", fontSize: 13, maxWidth: 760 }}>
              {(totals?.episodes ?? stats?.totalEpisodes ?? 0).toLocaleString()} episodes ·{" "}
              {(totals?.cues ?? stats?.cueMetrics?.cueCount ?? 0).toLocaleString()} cues ·{" "}
              {(totals?.projected ?? stats?.projectionMetrics?.stateCounts.projected ?? 0).toLocaleString()} projected ·{" "}
              {(totals?.cycles ?? cycles.length).toLocaleString()} cycles
            </div>
          </div>
          <PipelineRail />
        </section>

        <section
          className="card"
          style={{
            padding: 18,
            minWidth: 0,
            display: "grid",
            gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
            gap: 12,
          }}
        >
          <div>
            <div className="label" style={{ marginBottom: 6 }}>
              Graph Yield
            </div>
            <div className="mono tabular-nums" style={{ fontSize: 28, color: "#fff", lineHeight: 1 }}>
              {(totals?.entities ?? stats?.totalEntities ?? 0).toLocaleString()}
            </div>
            <div style={{ color: "var(--text-muted)", fontSize: 11, marginTop: 5 }}>
              entities
            </div>
          </div>
          <div>
            <div className="label" style={{ marginBottom: 6 }}>
              Relationships
            </div>
            <div className="mono tabular-nums" style={{ fontSize: 28, color: "#fff", lineHeight: 1 }}>
              {(totals?.relationships ?? stats?.totalRelationships ?? 0).toLocaleString()}
            </div>
            <div style={{ color: "var(--text-muted)", fontSize: 11, marginTop: 5 }}>
              durable edges
            </div>
          </div>
          <div style={{ gridColumn: "1 / -1" }}>
            <div className="metric-bar" style={{ height: 5 }}>
              <div
                className="metric-bar-fill"
                style={{
                  width: `${Math.max(0, Math.min(100, (lifecycleSummary?.cue.coverage ?? stats?.cueMetrics?.cueCoverage ?? 0) * 100))}%`,
                  background: "linear-gradient(90deg, #22d3ee, #34d399, #f97316)",
                }}
              />
            </div>
            <div className="label" style={{ marginTop: 8 }}>
              cue coverage {formatRate(lifecycleSummary?.cue.coverage ?? stats?.cueMetrics?.cueCoverage ?? 0)}
            </div>
          </div>
        </section>
      </header>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: 10,
        }}
      >
        {stageData.map((stage) => (
          <StageCard
            key={stage.key}
            label={stage.label}
            accent={stage.accent}
            status={stage.status}
            primary={stage.primary}
            secondary={stage.secondary}
            metrics={stage.metrics}
            drilldownLabel={STAGE_DRILLDOWNS[stage.key].command}
            onClick={() => openStageDrilldown(stage.key)}
          />
        ))}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: 10,
          minHeight: 0,
        }}
      >
        <RecentQueue episodes={recentEpisodes} />
        <TopActivated entities={activeRecallEntities} />
      </div>
    </div>
  );
}
