import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { useEngramStore } from "../store";
import type {
  BrainLoopEvaluationReport,
  BrainLoopEvaluationSignalKey,
  BrainLoopReportDegradation,
} from "../store/types";

function pct(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "n/a";
  return `${(value * 100).toFixed(value > 0.99 || value < 0.01 ? 0 : 1)}%`;
}

function num(value: number | null | undefined, digits = 1) {
  if (value == null || !Number.isFinite(value)) return "n/a";
  if (Math.abs(value - Math.round(value)) < 0.01) return Math.round(value).toLocaleString();
  return value.toFixed(digits);
}

function duration(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value) || value < 0) return "n/a";
  if (value < 1000) return `${value.toFixed(0)}ms`;
  if (value < 60_000) return `${(value / 1000).toFixed(1)}s`;
  return `${(value / 60_000).toFixed(1)}m`;
}

function formatAge(iso: string | null | undefined) {
  if (!iso) return "not run";
  const elapsedMs = Date.now() - new Date(iso).getTime();
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0) return "now";
  const minutes = Math.floor(elapsedMs / 60_000);
  if (minutes < 1) return "now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

const EVALUATION_SIGNAL_LABELS: Array<[BrainLoopEvaluationSignalKey, string]> = [
  ["cueUsefulness", "Cue usefulness"],
  ["projectionYield", "Projection yield"],
  ["recallQuality", "Recall quality"],
  ["falseRecall", "False recall"],
  ["triageCalibration", "Triage calibration"],
  ["consolidationEffect", "Consolidation effect"],
];

function formatSignalMetric(
  key: BrainLoopEvaluationSignalKey,
  metric: number | null | undefined,
) {
  if (key === "projectionYield" || key === "triageCalibration") return num(metric, 3);
  return pct(metric);
}

function statusAccent(status: string | null | undefined) {
  if (status === "measured" || status === "passed" || status === "ready") return "#34d399";
  if (status === "failed" || status === "error") return "#fb7185";
  if (status?.startsWith("needs")) return "#facc15";
  return "var(--text-muted)";
}

function compactList(values: string[], fallback = "none") {
  return values.length > 0 ? values.join(", ") : fallback;
}

function evidenceFailureSummary(failures: string[]) {
  if (failures.length === 0) return "none";
  if (failures.length === 1) return failures[0];
  return `${failures.length} failures`;
}

function fallbackReleaseEvidenceStatus(report: BrainLoopEvaluationReport) {
  const measuredCount = [
    report.humanLabelEvidence?.status,
    report.adoptionEvidence?.status,
    report.adoptionClientEvidence?.status,
  ].filter((status) => status === "measured").length;
  return `${measuredCount}/3 measured`;
}

function formatDegradation(item: BrainLoopReportDegradation) {
  const parts = [
    item.stage,
    item.skipReason,
    item.timeoutMs != null ? duration(item.timeoutMs) : null,
  ].filter(Boolean);
  return parts.join(" · ");
}

function EvidenceColumn({
  title,
  status,
  children,
}: {
  title: string;
  status: string;
  children: ReactNode;
}) {
  const accent = statusAccent(status);
  return (
    <div
      style={{
        minWidth: 0,
        borderTop: `1px solid ${accent}44`,
        paddingTop: 10,
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
        <div className="label" style={{ fontSize: 8 }}>
          {title}
        </div>
        <span className="label" style={{ color: accent, fontSize: 8 }}>
          {status}
        </span>
      </div>
      {children}
    </div>
  );
}

function EvidenceLine({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div style={{ minWidth: 0 }}>
      <div className="label" style={{ fontSize: 8, marginBottom: 3 }}>
        {label}
      </div>
      <div
        className="mono tabular-nums"
        style={{
          color: accent ?? "var(--text-secondary)",
          fontSize: 12,
          lineHeight: 1.3,
          overflowWrap: "anywhere",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  accent = "var(--accent)",
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div
      style={{
        minWidth: 0,
        borderTop: "1px solid var(--border-subtle)",
        paddingTop: 10,
      }}
    >
      <div className="label" style={{ fontSize: 8, marginBottom: 4 }}>
        {label}
      </div>
      <div
        className="mono tabular-nums"
        style={{
          fontSize: 20,
          lineHeight: 1,
          color: accent,
          overflowWrap: "anywhere",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function Section({
  title,
  status,
  accent,
  children,
}: {
  title: string;
  status: string;
  accent: string;
  children: ReactNode;
}) {
  return (
    <section
      className="card"
      style={{
        padding: 16,
        minWidth: 0,
        display: "flex",
        flexDirection: "column",
        gap: 14,
        borderColor: `${accent}24`,
        background: `linear-gradient(180deg, ${accent}0f, rgba(8, 10, 18, 0.72))`,
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
          <h2
            style={{
              margin: 0,
              fontSize: 15,
              fontWeight: 600,
              color: "var(--text-primary)",
              lineHeight: 1.1,
            }}
          >
            {title}
          </h2>
        </div>
        <span className="label" style={{ color: accent, fontSize: 9 }}>
          {status}
        </span>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
          gap: 12,
        }}
      >
        {children}
      </div>
    </section>
  );
}

function clampInt(value: string) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return 0;
  return Math.max(0, parsed);
}

function clampScore(value: string) {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) return 0;
  return Math.min(1, Math.max(0, parsed));
}

function ToggleField({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        minWidth: 0,
        color: "var(--text-secondary)",
        fontSize: 12,
      }}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.currentTarget.checked)}
      />
      <span>{label}</span>
    </label>
  );
}

function CompactInput({
  label,
  value,
  onChange,
  type = "text",
  min,
  max,
  step,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: "text" | "number";
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 5, minWidth: 0 }}>
      <span className="label" style={{ fontSize: 8 }}>
        {label}
      </span>
      <input
        type={type}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(event.currentTarget.value)}
        style={{
          width: "100%",
          boxSizing: "border-box",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-sm)",
          background: "rgba(255,255,255,0.03)",
          color: "var(--text-primary)",
          padding: "8px 9px",
          fontSize: 12,
          outline: "none",
        }}
      />
    </label>
  );
}

function LabelFormSection({
  title,
  status,
  children,
}: {
  title: string;
  status: string;
  children: ReactNode;
}) {
  return (
    <section
      className="card"
      style={{
        padding: 16,
        minWidth: 0,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
        <div className="label">{title}</div>
        <span className="label" style={{ color: status === "failed" ? "#fb7185" : "var(--text-muted)" }}>
          {status}
        </span>
      </div>
      {children}
    </section>
  );
}

export function EvaluationPanel() {
  const report = useEngramStore((s) => s.evaluationReport);
  const isLoading = useEngramStore((s) => s.isLoadingEvaluationReport);
  const isSavingRecallEvaluation = useEngramStore((s) => s.isSavingRecallEvaluation);
  const isSavingSessionEvaluation = useEngramStore((s) => s.isSavingSessionEvaluation);
  const loadEvaluationReport = useEngramStore((s) => s.loadEvaluationReport);
  const recordRecallEvaluation = useEngramStore((s) => s.recordRecallEvaluation);
  const recordSessionContinuityEvaluation = useEngramStore((s) => s.recordSessionContinuityEvaluation);
  const initialLoadAttempted = useRef(false);
  const [recallStatus, setRecallStatus] = useState<"idle" | "stored" | "failed">("idle");
  const [sessionStatus, setSessionStatus] = useState<"idle" | "stored" | "failed">("idle");
  const [recallLabel, setRecallLabel] = useState({
    recallTriggered: true,
    recallHelped: true,
    recallNeeded: true,
    packetsSurfaced: "0",
    packetsUsed: "0",
    falseRecalls: "0",
    stalePackets: "0",
    correctedPackets: "0",
    query: "",
    notes: "",
  });
  const [sessionLabel, setSessionLabel] = useState({
    baselineScore: "0",
    memoryScore: "1",
    openLoopExpected: false,
    openLoopRecovered: false,
    temporalExpected: false,
    temporalCorrect: false,
    scenario: "",
    notes: "",
  });

  useEffect(() => {
    if (!report && !initialLoadAttempted.current) {
      initialLoadAttempted.current = true;
      void loadEvaluationReport();
    }
  }, [loadEvaluationReport, report]);

  const topFamilies = useMemo(() => {
    const families = Object.entries(report?.recall.familyContributions ?? {});
    return families
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(([name, count]) => `${name} ${count}`)
      .join(" · ");
  }, [report]);

  const topCalibrationPhase = useMemo(() => {
    const phases = Object.entries(report?.consolidate.calibration.phaseTotals ?? {});
    return phases
      .sort((a, b) => b[1].labeledExamples - a[1].labeledExamples)
      .at(0);
  }, [report]);
  const latestCycleIssue = useMemo(() => {
    const error = report?.consolidate.latestCycle?.error;
    if (typeof error === "string" && error.trim()) return error;
    const phaseIssue = report?.consolidate.latestCycle?.phase_issue;
    return typeof phaseIssue === "string" && phaseIssue.trim() ? phaseIssue : null;
  }, [report]);
  const evaluationSignalRows = useMemo(
    () =>
      EVALUATION_SIGNAL_LABELS.map(([key, label]) => ({
        key,
        label,
        signal: report?.evaluationSignals[key],
      })),
    [report],
  );
  const releaseBlockers = useMemo(() => {
    const components = report?.releaseEvidence?.components;
    return Array.from(
      new Set([
        ...(components?.adoption.blockers ?? []),
        ...(components?.adoptionClients.blockers ?? []),
      ]),
    );
  }, [report]);
  const releaseMcpServerFailures = useMemo(() => {
    const components = report?.releaseEvidence?.components;
    return Array.from(
      new Set([
        ...(components?.adoption.mcpServerFailures ?? []),
        ...(components?.adoptionClients.mcpServerFailures ?? []),
      ]),
    );
  }, [report]);
  const reportDegradations = report?.degradations ?? [];

  const submitRecallLabel = async (event: FormEvent) => {
    event.preventDefault();
    setRecallStatus("idle");
    try {
      await recordRecallEvaluation({
        recallTriggered: recallLabel.recallTriggered,
        recallHelped: recallLabel.recallHelped,
        recallNeeded: recallLabel.recallNeeded,
        packetsSurfaced: clampInt(recallLabel.packetsSurfaced),
        packetsUsed: clampInt(recallLabel.packetsUsed),
        falseRecalls: clampInt(recallLabel.falseRecalls),
        stalePackets: clampInt(recallLabel.stalePackets),
        correctedPackets: clampInt(recallLabel.correctedPackets),
        query: recallLabel.query.trim() || null,
        notes: recallLabel.notes.trim() || null,
      });
      setRecallStatus("stored");
      setRecallLabel((current) => ({
        ...current,
        packetsSurfaced: "0",
        packetsUsed: "0",
        falseRecalls: "0",
        stalePackets: "0",
        correctedPackets: "0",
        query: "",
        notes: "",
      }));
    } catch {
      setRecallStatus("failed");
    }
  };

  const submitSessionLabel = async (event: FormEvent) => {
    event.preventDefault();
    setSessionStatus("idle");
    try {
      await recordSessionContinuityEvaluation({
        baselineScore: clampScore(sessionLabel.baselineScore),
        memoryScore: clampScore(sessionLabel.memoryScore),
        openLoopExpected: sessionLabel.openLoopExpected,
        openLoopRecovered: sessionLabel.openLoopRecovered,
        temporalExpected: sessionLabel.temporalExpected,
        temporalCorrect: sessionLabel.temporalCorrect,
        scenario: sessionLabel.scenario.trim() || null,
        notes: sessionLabel.notes.trim() || null,
      });
      setSessionStatus("stored");
      setSessionLabel((current) => ({
        ...current,
        scenario: "",
        notes: "",
      }));
    } catch {
      setSessionStatus("failed");
    }
  };

  if (!report && isLoading) {
    return (
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="card" style={{ padding: "12px 18px" }}>
          <span className="label">Loading evaluation...</span>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="absolute inset-0 flex items-center justify-center">
        <button className="card" style={{ padding: "12px 18px" }} onClick={() => void loadEvaluationReport()}>
          <span className="label">Load Evaluation</span>
        </button>
      </div>
    );
  }

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
        className="card"
        style={{
          padding: 18,
          display: "grid",
          gridTemplateColumns: "minmax(0, 1fr) auto",
          gap: 12,
          alignItems: "start",
        }}
      >
        <div style={{ minWidth: 0 }}>
          <div className="label" style={{ marginBottom: 8 }}>
            Brain Loop Evaluation
          </div>
          <h1
            style={{
              fontSize: 24,
              lineHeight: 1.1,
              fontWeight: 600,
              color: "#fff",
              margin: 0,
            }}
          >
            Runtime quality signals
          </h1>
          <div style={{ color: "var(--text-secondary)", fontSize: 12, marginTop: 8 }}>
            {report.totals.episodes.toLocaleString()} episodes ·{" "}
            {report.totals.entities.toLocaleString()} entities · refreshed{" "}
            {formatAge(report.generatedAt)}
          </div>
        </div>
        <button
          type="button"
          onClick={() => void loadEvaluationReport()}
          className="label"
          style={{
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            padding: "7px 10px",
            background: "rgba(255,255,255,0.03)",
            color: "var(--text-primary)",
            cursor: "pointer",
          }}
        >
          Refresh
        </button>
      </header>

      {report.degraded ? (
        <section
          className="card"
          style={{
            padding: 14,
            borderColor: "rgba(250, 204, 21, 0.28)",
            background: "rgba(250, 204, 21, 0.06)",
            display: "flex",
            flexDirection: "column",
            gap: 8,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
            <div className="label" style={{ color: "#facc15" }}>
              Report degraded
            </div>
            <span className="label" style={{ color: "var(--text-muted)" }}>
              {reportDegradations.length.toLocaleString()} fallback
              {reportDegradations.length === 1 ? "" : "s"}
            </span>
          </div>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 8,
              color: "var(--text-secondary)",
              fontSize: 12,
            }}
          >
            {reportDegradations.length > 0 ? (
              reportDegradations.map((item, index) => (
                <span
                  key={`${item.stage}-${item.skipReason ?? "none"}-${index}`}
                  className="mono"
                  style={{
                    border: "1px solid rgba(250, 204, 21, 0.22)",
                    borderRadius: "var(--radius-sm)",
                    padding: "4px 7px",
                    background: "rgba(0,0,0,0.18)",
                  }}
                >
                  {formatDegradation(item)}
                </span>
              ))
            ) : (
              <span className="mono">degraded fallback used</span>
            )}
          </div>
        </section>
      ) : null}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: 10,
        }}
      >
        <Section title="Cue" status={report.cue.status} accent="#facc15">
          <Metric label="coverage" value={pct(report.cue.coverage)} accent="#facc15" />
          <Metric label="surfaced" value={report.cue.surfacedCount.toLocaleString()} />
          <Metric label="selected rate" value={pct(report.cue.selectedRate)} />
          <Metric label="used rate" value={pct(report.cue.usedRate)} accent="#34d399" />
          <Metric label="projection" value={pct(report.cue.projectionConversionRate)} accent="#818cf8" />
          <Metric label="near miss" value={pct(report.cue.nearMissRate)} accent="#fb7185" />
        </Section>

        <Section title="Project" status={report.project.status} accent="#818cf8">
          <Metric label="projected" value={report.project.projectedCount.toLocaleString()} accent="#818cf8" />
          <Metric label="backlog" value={pct(report.project.backlogRate)} accent="#facc15" />
          <Metric label="failure" value={pct(report.project.failureRate)} accent="#fb7185" />
          <Metric label="latency" value={duration(report.project.avgTimeToProjectionMs)} accent="#22d3ee" />
          <Metric label="processing" value={duration(report.project.avgProcessingDurationMs)} />
          <Metric label="entities/episode" value={num(report.project.yield.avgLinkedEntitiesPerProjectedEpisode)} />
          <Metric label="relationships" value={report.project.yield.relationshipCount.toLocaleString()} />
        </Section>

        <Section title="Recall" status={report.recall.evaluation.status} accent="#34d399">
          <Metric label="precision" value={pct(report.recall.evaluation.memoryNeedPrecision)} accent="#34d399" />
          <Metric label="need recall" value={pct(report.recall.evaluation.memoryNeedRecall)} accent="#22d3ee" />
          <Metric label="missed need" value={pct(report.recall.evaluation.missedRecallRate)} accent="#fb7185" />
          <Metric label="false recall" value={pct(report.recall.evaluation.falseRecallRate)} accent="#fb7185" />
          <Metric label="useful packets" value={pct(report.recall.evaluation.usefulPacketRate)} />
          <Metric label="stale packets" value={pct(report.recall.evaluation.stalePacketRate)} accent="#facc15" />
          <Metric label="corrected packets" value={pct(report.recall.evaluation.correctedPacketRate)} accent="#fb7185" />
          <Metric label="analysis p95" value={duration(report.recall.latency.analyzerMs.p95Ms)} accent="#22d3ee" />
          <Metric label="probe p95" value={duration(report.recall.latency.probeMs.p95Ms)} />
          <Metric label="labels" value={report.recall.evaluation.sampleCount.toLocaleString()} />
        </Section>

        <Section title="Memory Value" status={report.memoryValue.status} accent="#14b8a6">
          <Metric label="operations" value={report.memoryValue.cost.operationCount.toLocaleString()} accent="#14b8a6" />
          <Metric label="skipped" value={report.memoryValue.cost.skippedCount.toLocaleString()} accent="#facc15" />
          <Metric label="errors" value={report.memoryValue.cost.errorCount.toLocaleString()} accent="#fb7185" />
          <Metric label="avg added" value={duration(report.memoryValue.cost.avgAddedLatencyMs)} />
          <Metric label="p95 added" value={duration(report.memoryValue.cost.p95AddedLatencyMs)} accent="#22d3ee" />
          <Metric label="budget p95" value={duration(report.memoryValue.cost.p95BudgetMs)} />
          <Metric label="timeout" value={pct(report.memoryValue.cost.timeoutRate)} accent="#fb7185" />
          <Metric label="degraded" value={pct(report.memoryValue.cost.degradedRate)} accent="#fb7185" />
          <Metric label="budget miss" value={pct(report.memoryValue.cost.budgetMissRate)} accent="#facc15" />
          <Metric label="cache hit" value={pct(report.memoryValue.cost.cacheHitRate)} accent="#34d399" />
          <Metric label="stale packet" value={pct(report.memoryValue.benefit.stalePacketRate)} accent="#facc15" />
          <Metric label="corrected packet" value={pct(report.memoryValue.benefit.correctedPacketRate)} accent="#fb7185" />
          <Metric label="dismissed" value={report.recall.control.dismissedCount.toLocaleString()} accent="#fb7185" />
          <Metric label="corrected" value={report.recall.control.correctedCount.toLocaleString()} accent="#facc15" />
          <Metric label="benefit labels" value={report.memoryValue.benefit.recallSampleCount.toLocaleString()} />
          <Metric label="continuity lift" value={num(report.memoryValue.benefit.sessionContinuityLift, 3)} />
        </Section>

        <Section
          title="Recall Gate"
          status={report.recall.control.adaptiveThresholdsEnabled ? "adaptive" : report.recall.status}
          accent="#a78bfa"
        >
          <Metric label="analyses" value={report.recall.totalAnalyses.toLocaleString()} accent="#a78bfa" />
          <Metric label="triggers" value={report.recall.triggerCount.toLocaleString()} />
          <Metric label="runtime used" value={report.recall.control.usedCount.toLocaleString()} accent="#34d399" />
          <Metric label="dismissed" value={report.recall.control.dismissedCount.toLocaleString()} accent="#fb7185" />
          <Metric label="graph lift" value={pct(report.recall.graphLiftRate)} />
          <Metric label="probe trigger" value={pct(report.recall.probeTriggerRate)} />
          <Metric label="graph override" value={report.recall.control.graphOverrideCount.toLocaleString()} />
          <Metric label="resonance" value={num(report.recall.control.thresholds.resonance, 2)} accent="#facc15" />
        </Section>

        <Section title="Continuity Scorecard" status={report.recall.continuity.status} accent="#22d3ee">
          <Metric label="lift" value={num(report.recall.continuity.sessionContinuityLift, 3)} accent="#22d3ee" />
          <Metric label="open loops" value={pct(report.recall.continuity.openLoopRecoveryRate)} />
          <Metric label="temporal" value={pct(report.recall.continuity.temporalCorrectness)} />
          <Metric label="labels" value={report.recall.continuity.sampleCount.toLocaleString()} />
          <div
            style={{
              gridColumn: "1 / -1",
              minWidth: 0,
              borderTop: "1px solid rgba(34, 211, 238, 0.18)",
              paddingTop: 10,
              color: "var(--text-muted)",
              fontSize: 11,
              lineHeight: 1.35,
            }}
          >
            North star: cold Decision hit rate (fresh session surfaces ≥1 prior
            Decision). This panel measures labeled continuity lift — open work
            below is secondary hygiene, not product success.
          </div>
        </Section>

        <Section title="Consolidate" status={report.consolidate.status} accent="#f97316">
          <Metric label="cycles" value={report.consolidate.cycleCount.toLocaleString()} accent="#f97316" />
          <Metric label="affected" value={report.consolidate.itemsAffected.toLocaleString()} />
          <Metric label="effect" value={pct(report.consolidate.effectRate)} accent="#34d399" />
          <Metric label="adjudication" value={pct(report.consolidate.adjudication.effectRate)} accent="#22d3ee" />
          <Metric label="unaffected" value={report.consolidate.adjudication.itemsUnaffected.toLocaleString()} />
          <Metric
            label="open work"
            value={(report.consolidate.adjudication.openWorkCount ?? 0).toLocaleString()}
            accent="#fb7185"
          />
          <Metric
            label="approved"
            value={(report.consolidate.adjudication.approvedEvidenceCount ?? 0).toLocaleString()}
            accent="#34d399"
          />
          <Metric
            label="deferred"
            value={(report.consolidate.adjudication.deferredEvidenceCount ?? 0).toLocaleString()}
            accent="#facc15"
          />
          <Metric label="errors" value={report.consolidate.errorCount.toLocaleString()} accent="#fb7185" />
          <Metric label="snapshots" value={report.consolidate.calibration.snapshotCount.toLocaleString()} />
          <Metric label="accuracy" value={pct(topCalibrationPhase?.[1].accuracy)} accent="#34d399" />
          <Metric label="ECE" value={num(topCalibrationPhase?.[1].expectedCalibrationError, 3)} accent="#facc15" />
          <div
            style={{
              gridColumn: "1 / -1",
              minWidth: 0,
              borderTop: "1px solid rgba(250, 204, 21, 0.18)",
              paddingTop: 10,
              color: "var(--text-muted)",
              fontSize: 11,
              lineHeight: 1.35,
            }}
          >
            Open work / deferred evidence is graph hygiene. Product success is cold
            session continuity (Decision recall), not a zero adjudication queue.
          </div>
          {latestCycleIssue ? (
            <div
              style={{
                gridColumn: "1 / -1",
                minWidth: 0,
                borderTop: "1px solid rgba(251, 113, 133, 0.28)",
                paddingTop: 10,
              }}
            >
              <div className="label" style={{ fontSize: 8, marginBottom: 4, color: "#fb7185" }}>
                latest issue
              </div>
              <div
                style={{
                  color: "var(--text-secondary)",
                  fontSize: 12,
                  lineHeight: 1.35,
                  overflowWrap: "anywhere",
                }}
              >
                {latestCycleIssue}
              </div>
            </div>
          ) : null}
        </Section>
      </div>

      <section className="card" style={{ padding: 16, minWidth: 0 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 10,
            marginBottom: 12,
          }}
        >
          <div className="label">Signal Readiness</div>
          <span className="label" style={{ color: "var(--text-muted)" }}>
            {evaluationSignalRows.filter(({ signal }) => signal?.status === "measured").length}/
            {evaluationSignalRows.length} measured
          </span>
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: 8,
          }}
        >
          {evaluationSignalRows.map(({ key, label, signal }) => (
            <div
              key={key}
              style={{
                minWidth: 0,
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-sm)",
                padding: 10,
                background: "rgba(255,255,255,0.025)",
              }}
            >
              <div className="label" style={{ fontSize: 8, marginBottom: 6 }}>
                {label}
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: 8,
                  alignItems: "baseline",
                }}
              >
                <span style={{ color: "var(--text-primary)", fontSize: 13 }}>
                  {signal?.status ?? "needs_data"}
                </span>
                <span className="mono tabular-nums" style={{ color: "var(--accent)", fontSize: 13 }}>
                  {formatSignalMetric(key, signal?.metric)}
                </span>
              </div>
              <div style={{ color: "var(--text-muted)", fontSize: 11, marginTop: 6 }}>
                {(signal?.evidenceCount ?? 0).toLocaleString()} evidence
              </div>
              {signal?.gap ? (
                <div
                  style={{
                    color: "#fb7185",
                    fontSize: 11,
                    lineHeight: 1.3,
                    marginTop: 6,
                    overflowWrap: "anywhere",
                  }}
                >
                  {signal.gap}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      </section>

      <section className="card" style={{ padding: 16, minWidth: 0 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 10,
            marginBottom: 12,
          }}
        >
          <div className="label">Release Evidence</div>
          <span
            className="label"
            style={{
              color: statusAccent(report.releaseEvidence?.status),
            }}
          >
            {report.releaseEvidence?.status ?? fallbackReleaseEvidenceStatus(report)}
          </span>
        </div>
        {report.releaseEvidence ||
        report.humanLabelEvidence ||
        report.adoptionEvidence ||
        report.adoptionClientEvidence ? (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
              gap: 12,
            }}
          >
            {report.releaseEvidence ? (
              <EvidenceColumn title="Readiness" status={report.releaseEvidence.status}>
                <EvidenceLine
                  label="missing"
                  value={compactList(report.releaseEvidence.missing)}
                  accent={
                    report.releaseEvidence.missing.length > 0
                      ? "#facc15"
                      : "var(--text-muted)"
                  }
                />
                <EvidenceLine
                  label="failures"
                  value={evidenceFailureSummary(report.releaseEvidence.failures)}
                  accent={
                    report.releaseEvidence.failures.length > 0
                      ? "#fb7185"
                      : "var(--text-muted)"
                  }
                />
                <EvidenceLine
                  label="signals"
                  value={report.releaseEvidence.components.evaluationSignals.status}
                  accent={statusAccent(
                    report.releaseEvidence.components.evaluationSignals.status,
                  )}
                />
                <EvidenceLine
                  label="clients"
                  value={report.releaseEvidence.components.adoptionClients.status}
                  accent={statusAccent(
                    report.releaseEvidence.components.adoptionClients.status,
                  )}
                />
                {releaseBlockers.length > 0 ? (
                  <EvidenceLine
                    label="blockers"
                    value={compactList(releaseBlockers)}
                    accent="#fb7185"
                  />
                ) : null}
                {releaseMcpServerFailures.length > 0 ? (
                  <EvidenceLine
                    label="mcp failures"
                    value={compactList(releaseMcpServerFailures)}
                    accent="#fb7185"
                  />
                ) : null}
              </EvidenceColumn>
            ) : null}

            <EvidenceColumn
              title="Human labels"
              status={report.humanLabelEvidence?.status ?? "missing"}
            >
              <EvidenceLine
                label="client"
                value={report.humanLabelEvidence?.client ?? "unknown"}
                accent={statusAccent(report.humanLabelEvidence?.status)}
              />
              <EvidenceLine
                label="recall samples"
                value={`${report.humanLabelEvidence?.recallSampleCount ?? 0}/${
                  report.humanLabelEvidence?.minRecallSamples ?? 0
                }`}
              />
              <EvidenceLine
                label="session samples"
                value={`${report.humanLabelEvidence?.sessionSampleCount ?? 0}/${
                  report.humanLabelEvidence?.minSessionSamples ?? 0
                }`}
              />
              <EvidenceLine
                label="failures"
                value={evidenceFailureSummary(report.humanLabelEvidence?.failures ?? [])}
                accent={
                  (report.humanLabelEvidence?.failures.length ?? 0) > 0
                    ? "#fb7185"
                    : "var(--text-muted)"
                }
              />
            </EvidenceColumn>

            <EvidenceColumn
              title="Adoption"
              status={report.adoptionEvidence?.status ?? "missing"}
            >
              <EvidenceLine
                label="client"
                value={report.adoptionEvidence?.client ?? "unknown"}
                accent={statusAccent(report.adoptionEvidence?.status)}
              />
              <EvidenceLine
                label="calls"
                value={(report.adoptionEvidence?.callCount ?? 0).toLocaleString()}
              />
              {(report.adoptionEvidence?.blockers?.length ?? 0) > 0 ? (
                <EvidenceLine
                  label="blockers"
                  value={compactList(report.adoptionEvidence?.blockers ?? [])}
                  accent="#fb7185"
                />
              ) : null}
              {(report.adoptionEvidence?.mcpServerFailures?.length ?? 0) > 0 ? (
                <EvidenceLine
                  label="mcp failures"
                  value={compactList(report.adoptionEvidence?.mcpServerFailures ?? [])}
                  accent="#fb7185"
                />
              ) : null}
              <EvidenceLine
                label="required tools"
                value={`${report.adoptionEvidence?.requiredTools.observed.length ?? 0}/${
                  report.adoptionEvidence?.requiredTools.expected.length ?? 0
                }`}
              />
              <EvidenceLine
                label="failures"
                value={evidenceFailureSummary(report.adoptionEvidence?.failures ?? [])}
                accent={
                  (report.adoptionEvidence?.failures.length ?? 0) > 0
                    ? "#fb7185"
                    : "var(--text-muted)"
                }
              />
            </EvidenceColumn>

            <EvidenceColumn
              title="Clients"
              status={report.adoptionClientEvidence?.status ?? "missing"}
            >
              <EvidenceLine
                label="required"
                value={compactList(report.adoptionClientEvidence?.requiredClients ?? [])}
              />
              <EvidenceLine
                label="observed"
                value={compactList(report.adoptionClientEvidence?.observedClients ?? [])}
                accent={statusAccent(report.adoptionClientEvidence?.status)}
              />
              <EvidenceLine
                label="reports"
                value={(report.adoptionClientEvidence?.reportCount ?? 0).toLocaleString()}
              />
              {(report.adoptionClientEvidence?.blockers?.length ?? 0) > 0 ? (
                <EvidenceLine
                  label="blockers"
                  value={compactList(report.adoptionClientEvidence?.blockers ?? [])}
                  accent="#fb7185"
                />
              ) : null}
              {(report.adoptionClientEvidence?.mcpServerFailures?.length ?? 0) > 0 ? (
                <EvidenceLine
                  label="mcp failures"
                  value={compactList(
                    report.adoptionClientEvidence?.mcpServerFailures ?? [],
                  )}
                  accent="#fb7185"
                />
              ) : null}
              <EvidenceLine
                label="failures"
                value={evidenceFailureSummary(report.adoptionClientEvidence?.failures ?? [])}
                accent={
                  (report.adoptionClientEvidence?.failures.length ?? 0) > 0
                    ? "#fb7185"
                    : "var(--text-muted)"
                }
              />
            </EvidenceColumn>

            {(report.additionalAdoptionEvidence?.length ?? 0) > 0 ? (
              <EvidenceColumn title="Additional adoption" status="measured">
                <EvidenceLine
                  label="reports"
                  value={(report.additionalAdoptionEvidence ?? [])
                    .map((evidence) => evidence.client ?? evidence.artifactPath ?? "unknown")
                    .join(", ")}
                />
                <EvidenceLine
                  label="failures"
                  value={evidenceFailureSummary(
                    (report.additionalAdoptionEvidence ?? []).flatMap(
                      (evidence) => evidence.failures,
                    ),
                  )}
                  accent={
                    (report.additionalAdoptionEvidence ?? []).some(
                      (evidence) => evidence.failures.length > 0,
                    )
                      ? "#fb7185"
                      : "var(--text-muted)"
                  }
                />
                {(report.additionalAdoptionEvidence ?? []).some(
                  (evidence) => (evidence.blockers?.length ?? 0) > 0,
                ) ? (
                  <EvidenceLine
                    label="blockers"
                    value={compactList(
                      (report.additionalAdoptionEvidence ?? []).flatMap(
                        (evidence) => evidence.blockers ?? [],
                      ),
                    )}
                    accent="#fb7185"
                  />
                ) : null}
              </EvidenceColumn>
            ) : null}
          </div>
        ) : (
          <div style={{ color: "var(--text-muted)", fontSize: 13 }}>
            No release evidence attached
          </div>
        )}
      </section>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: 10,
        }}
      >
        <LabelFormSection title="Recall Label" status={isSavingRecallEvaluation ? "saving" : recallStatus}>
          <form onSubmit={submitRecallLabel} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 8 }}>
              <ToggleField
                label="Triggered"
                checked={recallLabel.recallTriggered}
                onChange={(checked) => setRecallLabel((current) => ({ ...current, recallTriggered: checked }))}
              />
              <ToggleField
                label="Helped"
                checked={recallLabel.recallHelped}
                onChange={(checked) => setRecallLabel((current) => ({ ...current, recallHelped: checked }))}
              />
              <ToggleField
                label="Needed"
                checked={recallLabel.recallNeeded}
                onChange={(checked) => setRecallLabel((current) => ({ ...current, recallNeeded: checked }))}
              />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(86px, 1fr))", gap: 8 }}>
              <CompactInput
                label="Surfaced"
                type="number"
                min={0}
                step={1}
                value={recallLabel.packetsSurfaced}
                onChange={(value) => setRecallLabel((current) => ({ ...current, packetsSurfaced: value }))}
              />
              <CompactInput
                label="Used"
                type="number"
                min={0}
                step={1}
                value={recallLabel.packetsUsed}
                onChange={(value) => setRecallLabel((current) => ({ ...current, packetsUsed: value }))}
              />
              <CompactInput
                label="False"
                type="number"
                min={0}
                step={1}
                value={recallLabel.falseRecalls}
                onChange={(value) => setRecallLabel((current) => ({ ...current, falseRecalls: value }))}
              />
              <CompactInput
                label="Stale"
                type="number"
                min={0}
                step={1}
                value={recallLabel.stalePackets}
                onChange={(value) => setRecallLabel((current) => ({ ...current, stalePackets: value }))}
              />
              <CompactInput
                label="Corrected"
                type="number"
                min={0}
                step={1}
                value={recallLabel.correctedPackets}
                onChange={(value) => setRecallLabel((current) => ({ ...current, correctedPackets: value }))}
              />
            </div>
            <CompactInput
              label="Query"
              value={recallLabel.query}
              onChange={(value) => setRecallLabel((current) => ({ ...current, query: value }))}
            />
            <CompactInput
              label="Notes"
              value={recallLabel.notes}
              onChange={(value) => setRecallLabel((current) => ({ ...current, notes: value }))}
            />
            <button
              type="submit"
              disabled={isSavingRecallEvaluation}
              className="label"
              style={{
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)",
                padding: "8px 10px",
                background: "rgba(52, 211, 153, 0.12)",
                color: "var(--text-primary)",
                cursor: isSavingRecallEvaluation ? "wait" : "pointer",
              }}
            >
              Store Recall
            </button>
          </form>
        </LabelFormSection>

        <LabelFormSection title="Continuity Label" status={isSavingSessionEvaluation ? "saving" : sessionStatus}>
          <form onSubmit={submitSessionLabel} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 8 }}>
              <CompactInput
                label="Baseline"
                type="number"
                min={0}
                max={1}
                step={0.1}
                value={sessionLabel.baselineScore}
                onChange={(value) => setSessionLabel((current) => ({ ...current, baselineScore: value }))}
              />
              <CompactInput
                label="Memory"
                type="number"
                min={0}
                max={1}
                step={0.1}
                value={sessionLabel.memoryScore}
                onChange={(value) => setSessionLabel((current) => ({ ...current, memoryScore: value }))}
              />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 8 }}>
              <ToggleField
                label="Open loop"
                checked={sessionLabel.openLoopExpected}
                onChange={(checked) => setSessionLabel((current) => ({ ...current, openLoopExpected: checked }))}
              />
              <ToggleField
                label="Recovered"
                checked={sessionLabel.openLoopRecovered}
                onChange={(checked) => setSessionLabel((current) => ({ ...current, openLoopRecovered: checked }))}
              />
              <ToggleField
                label="Temporal"
                checked={sessionLabel.temporalExpected}
                onChange={(checked) => setSessionLabel((current) => ({ ...current, temporalExpected: checked }))}
              />
              <ToggleField
                label="Correct"
                checked={sessionLabel.temporalCorrect}
                onChange={(checked) => setSessionLabel((current) => ({ ...current, temporalCorrect: checked }))}
              />
            </div>
            <CompactInput
              label="Scenario"
              value={sessionLabel.scenario}
              onChange={(value) => setSessionLabel((current) => ({ ...current, scenario: value }))}
            />
            <CompactInput
              label="Notes"
              value={sessionLabel.notes}
              onChange={(value) => setSessionLabel((current) => ({ ...current, notes: value }))}
            />
            <button
              type="submit"
              disabled={isSavingSessionEvaluation}
              className="label"
              style={{
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)",
                padding: "8px 10px",
                background: "rgba(34, 211, 238, 0.12)",
                color: "var(--text-primary)",
                cursor: isSavingSessionEvaluation ? "wait" : "pointer",
              }}
            >
              Store Continuity
            </button>
          </form>
        </LabelFormSection>

        <section className="card" style={{ padding: 16, minWidth: 0 }}>
          <div className="label" style={{ marginBottom: 10 }}>
            Recall Families
          </div>
          <div style={{ color: "var(--text-secondary)", fontSize: 13 }}>
            {topFamilies || "No trigger families recorded"}
          </div>
        </section>

        <section className="card" style={{ padding: 16, minWidth: 0 }}>
          <div className="label" style={{ marginBottom: 10 }}>
            Calibration
          </div>
          {report.consolidate.calibration.status === "needs_quality" ? (
            <div style={{ color: "var(--text-muted)", fontSize: 13 }}>Needs labeled decisions</div>
          ) : topCalibrationPhase ? (
            <div style={{ color: "var(--text-secondary)", fontSize: 13 }}>
              {topCalibrationPhase[0]} · {topCalibrationPhase[1].labeledExamples.toLocaleString()} labels · accuracy{" "}
              {pct(topCalibrationPhase[1].accuracy)} · ECE {num(topCalibrationPhase[1].expectedCalibrationError, 3)}
            </div>
          ) : (
            <div style={{ color: "var(--text-muted)", fontSize: 13 }}>No calibration snapshots</div>
          )}
        </section>
      </div>

      {report.coverageGaps.length > 0 && (
        <section className="card" style={{ padding: 16 }}>
          <div className="label" style={{ marginBottom: 10 }}>
            Coverage Gaps
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {report.coverageGaps.map((gap) => (
              <div
                key={gap}
                style={{
                  color: "var(--text-secondary)",
                  fontSize: 12,
                  borderTop: "1px solid var(--border-subtle)",
                  paddingTop: 7,
                }}
              >
                {gap}
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
