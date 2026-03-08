import { useEffect, useState, type CSSProperties } from "react";

type BaselineSummary = {
  name: string;
  family: string;
  scenario_pass_rate: number;
  false_recall_rate: number;
  temporal_correctness: number;
  negation_correctness: number;
  open_loop_recovery: number;
  prospective_trigger_rate: number;
};

type AblationSummary = {
  name: string;
  scenario_pass_rate: number;
  false_recall_rate: number;
};

type ScenarioWinner = {
  scenario_id: string;
  title: string;
  winner: string | null;
  winner_score: number;
};

type BenchmarkSummary = {
  generated_at: string;
  track: string;
  mode: string;
  seeds: number[];
  headline: {
    engram_full_pass_rate: number | null;
    engram_full_false_recall: number | null;
  };
  primary_baselines: BaselineSummary[];
  ablations: AblationSummary[];
  scenario_winners: ScenarioWinner[];
};

const serif: CSSProperties = { fontFamily: '"Instrument Serif", serif', fontStyle: "italic" };
const body: CSSProperties = { fontFamily: '"Outfit", sans-serif' };
const mono: CSSProperties = { fontFamily: '"JetBrains Mono", monospace' };
const tabularNums: CSSProperties = { fontVariantNumeric: "tabular-nums" };

function displayName(name: string) {
  return name.replace(/_/g, " ");
}

function fmtDate(iso: string) {
  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

function fmt(value: number | null | undefined) {
  if (value == null) {
    return "n/a";
  }
  return value.toFixed(3);
}

export function BenchmarkShowcase() {
  const [data, setData] = useState<BenchmarkSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    fetch("/benchmarks/latest.json")
      .then(async (response) => {
        if (!response.ok) {
          throw new Error("benchmark summary unavailable");
        }
        return response.json();
      })
      .then((json: BenchmarkSummary) => {
        if (!cancelled) {
          setData(json);
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

  const shellStyle: CSSProperties = {
    borderRadius: 24,
    border: "1px solid rgba(103,232,249,0.18)",
    background:
      "linear-gradient(180deg, rgba(103,232,249,0.05), rgba(255,255,255,0.02))",
    backdropFilter: "blur(22px)",
    boxShadow: "0 0 40px rgba(103,232,249,0.08)",
    padding: "2rem",
  };

  if (!data) {
    return (
      <div style={shellStyle}>
        <div style={{ ...mono, fontSize: 11, letterSpacing: "0.2em", textTransform: "uppercase", color: "#67e8f9", marginBottom: 12 }}>
          Benchmark Results
        </div>
        <h2 style={{ ...serif, fontSize: "clamp(1.5rem, 3vw, 2rem)", marginBottom: 12 }}>
          Measured, not asserted.
        </h2>
        <p style={{ ...body, color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: 0 }}>
          {error ?? "Loading benchmark summary\u2026"} Run the showcase benchmark with
          {" "}
          <code style={{ ...mono, color: "var(--text-primary)" }}>
            --website-export-path website/public/benchmarks/latest.json
          </code>
          {" "}
          to publish fresh numbers on the site.
        </p>
      </div>
    );
  }

  const engram = data.primary_baselines.find((item) => item.name === "engram_full");
  const competitors = data.primary_baselines.filter((item) => item.name !== "engram_full");
  const prospectiveGap = data.ablations.find((item) => item.name === "engram_search_only");

  return (
    <div style={shellStyle}>
      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", gap: 16, marginBottom: 24 }}>
        <div>
          <div style={{ ...mono, fontSize: 11, letterSpacing: "0.2em", textTransform: "uppercase", color: "#67e8f9", marginBottom: 12 }}>
            Benchmark Results
          </div>
          <h2 style={{ ...serif, fontSize: "clamp(1.5rem, 3vw, 2rem)", marginBottom: 10 }}>
            Engram against stronger fair baselines.
          </h2>
          <p style={{ ...body, color: "var(--text-secondary)", lineHeight: 1.7, maxWidth: 680, marginBottom: 0 }}>
            Fixed retrieval budgets, deterministic scenario grading, and real system paths. This section renders the latest exported showcase benchmark instead of static claims.
          </p>
        </div>
        <div style={{ minWidth: 220 }}>
          <div style={{ ...mono, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 8 }}>
            Latest Export
          </div>
          <div style={{ ...body, color: "var(--text-primary)", fontSize: 14 }} suppressHydrationWarning>
            {fmtDate(data.generated_at)}
          </div>
          <div style={{ ...body, color: "var(--text-secondary)", fontSize: 14 }}>
            track={data.track} • mode={data.mode} • seeds={data.seeds.join(", ")}
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16, marginBottom: 28 }}>
        <div style={{ border: "1px solid rgba(103,232,249,0.14)", borderRadius: 18, padding: 18, background: "rgba(255,255,255,0.02)" }}>
          <div style={{ ...mono, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 8 }}>
            Engram Pass Rate
          </div>
          <div style={{ ...serif, ...tabularNums, fontSize: 36, color: "var(--text-primary)" }}>
            {fmt(data.headline.engram_full_pass_rate)}
          </div>
        </div>
        <div style={{ border: "1px solid rgba(103,232,249,0.14)", borderRadius: 18, padding: 18, background: "rgba(255,255,255,0.02)" }}>
          <div style={{ ...mono, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 8 }}>
            Engram False Recall
          </div>
          <div style={{ ...serif, ...tabularNums, fontSize: 36, color: "var(--text-primary)" }}>
            {fmt(data.headline.engram_full_false_recall)}
          </div>
        </div>
        <div style={{ border: "1px solid rgba(103,232,249,0.14)", borderRadius: 18, padding: 18, background: "rgba(255,255,255,0.02)" }}>
          <div style={{ ...mono, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 8 }}>
            Best Non-Engram
          </div>
          <div style={{ ...serif, ...tabularNums, fontSize: 36, color: "var(--text-primary)" }}>
            {competitors.length > 0 ? fmt(Math.max(...competitors.map((item) => item.scenario_pass_rate))) : "n/a"}
          </div>
        </div>
        <div style={{ border: "1px solid rgba(103,232,249,0.14)", borderRadius: 18, padding: 18, background: "rgba(255,255,255,0.02)" }}>
          <div style={{ ...mono, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 8 }}>
            Search-Only Ablation
          </div>
          <div style={{ ...serif, ...tabularNums, fontSize: 36, color: "var(--text-primary)" }}>
            {fmt(prospectiveGap?.scenario_pass_rate)}
          </div>
        </div>
      </div>

      <div style={{ overflowX: "auto", marginBottom: 24 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", ...body }}>
          <caption style={{ ...mono, textAlign: "left", fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 12 }}>
            Primary baseline comparison across key memory metrics
          </caption>
          <thead>
            <tr style={{ color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", fontSize: 11 }}>
              <th scope="col" style={{ textAlign: "left", padding: "0 0 12px" }}>Baseline</th>
              <th scope="col" style={{ textAlign: "right", padding: "0 0 12px" }}>Pass</th>
              <th scope="col" style={{ textAlign: "right", padding: "0 0 12px" }}>False Recall</th>
              <th scope="col" style={{ textAlign: "right", padding: "0 0 12px" }}>Temporal</th>
              <th scope="col" style={{ textAlign: "right", padding: "0 0 12px" }}>Prospective</th>
            </tr>
          </thead>
          <tbody>
            {data.primary_baselines.map((item) => (
              <tr key={item.name} style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                <td style={{ padding: "14px 0", color: item.name === "engram_full" ? "var(--accent)" : "var(--text-primary)" }}>
                  {displayName(item.name)}
                </td>
                <td style={{ padding: "14px 0", textAlign: "right", color: "var(--text-primary)", ...tabularNums }}>{fmt(item.scenario_pass_rate)}</td>
                <td style={{ padding: "14px 0", textAlign: "right", color: "var(--text-primary)", ...tabularNums }}>{fmt(item.false_recall_rate)}</td>
                <td style={{ padding: "14px 0", textAlign: "right", color: "var(--text-primary)", ...tabularNums }}>{fmt(item.temporal_correctness)}</td>
                <td style={{ padding: "14px 0", textAlign: "right", color: "var(--text-primary)", ...tabularNums }}>{fmt(item.prospective_trigger_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16 }}>
        <div style={{ borderRadius: 18, border: "1px solid rgba(255,255,255,0.06)", padding: 18, background: "rgba(255,255,255,0.015)" }}>
          <div style={{ ...mono, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 12 }}>
            What This Shows
          </div>
          <p style={{ ...body, margin: 0, lineHeight: 1.7, color: "var(--text-secondary)" }}>
            Engram is measured against stronger fair baselines with fixed budgets, not hand-waved against a toy notebook. The headline here is system behavior under constrained recall, not bigger prompts.
          </p>
        </div>
        <div style={{ borderRadius: 18, border: "1px solid rgba(255,255,255,0.06)", padding: 18, background: "rgba(255,255,255,0.015)" }}>
          <div style={{ ...mono, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 12 }}>
            Scenario Winners
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {data.scenario_winners.slice(0, 5).map((item) => (
              <div key={item.scenario_id} style={{ ...body, fontSize: 14, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                <span style={{ color: "var(--text-primary)" }}>{item.title}</span>
                {" "}
                <span style={{ color: "var(--text-muted)" }}>{"\u2192"}</span>
                {" "}
                <span style={{ color: "var(--accent)" }}>{item.winner ? displayName(item.winner) : "no pass"}</span>
              </div>
            ))}
            {data.scenario_winners.length > 5 && (
              <div style={{ ...mono, fontSize: 12, color: "var(--text-muted)" }}>
                +{data.scenario_winners.length - 5} more scenarios
              </div>
            )}
          </div>
        </div>
      </div>

      {engram == null ? null : (
        <p style={{ ...body, marginTop: 18, marginBottom: 0, color: "var(--text-muted)", fontSize: 14, lineHeight: 1.7 }}>
          Measured against equal retrieval budgets. Showcase track is the primary credibility track; answer and external tracks are supporting layers.
        </p>
      )}
    </div>
  );
}
