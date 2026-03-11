import { Link } from "react-router-dom";

import { BenchmarkShowcase } from "../components/BenchmarkShowcase";
import { ScrollReveal } from "../components/ScrollReveal";
import {
  fallbackBenchmarkSummary,
  useBenchmarkSummary,
} from "../lib/benchmarkData";

const serif = { fontFamily: '"Instrument Serif", Georgia, serif', fontStyle: "italic" as const };
const mono = { fontFamily: '"JetBrains Mono", monospace' };
const body = { fontFamily: '"Outfit", sans-serif' };

const METHODOLOGY = [
  "Equal retrieval budgets per scenario and the same transcript order for every measured baseline.",
  "Deterministic scenario grading on surfaced evidence rather than anecdotal chat demos.",
  "Real Engram system paths through GraphManager rather than fixture-only shortcuts.",
  "External technology comparisons are clearly split between measured proxies and spec-only targets.",
  "Website numbers come from exported benchmark artifacts, not hand-edited copy.",
] as const;

function Label({ children }: { children: string }) {
  return (
    <span
      style={{
        ...mono,
        display: "block",
        marginBottom: 16,
        fontSize: 11,
        fontWeight: 500,
        letterSpacing: "0.18em",
        textTransform: "uppercase" as const,
        color: "#67e8f9",
      }}
    >
      {children}
    </span>
  );
}

function Badge({
  children,
  accent,
}: {
  children: string;
  accent: string;
}) {
  return (
    <span
      style={{
        ...mono,
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        fontSize: 10,
        letterSpacing: "0.12em",
        textTransform: "uppercase" as const,
        padding: "5px 10px",
        borderRadius: 9999,
        border: `1px solid ${accent}33`,
        background: `${accent}14`,
        color: accent,
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: accent,
          boxShadow: `0 0 12px ${accent}66`,
        }}
      />
      {children}
    </span>
  );
}

function BaselineGrid({
  title,
  description,
  items,
}: {
  title: string;
  description: string;
  items: Array<{
    name?: string;
    display_name: string;
    status: string | null;
    accent: string | null;
    archetype: string | null;
    description: string | null;
    fairness_notes: string | null;
    known_limitations: string | null;
    why_included: string | null;
    external_technology_label?: string | null;
  }>;
}) {
  if (items.length === 0) {
    return null;
  }
  return (
    <section style={{ padding: "0 24px 88px" }}>
      <div style={{ maxWidth: 1080, margin: "0 auto" }}>
        <ScrollReveal>
          <Label>{title}</Label>
          <p style={{ ...body, color: "var(--text-secondary)", lineHeight: 1.75, maxWidth: 760, marginBottom: 28 }}>
            {description}
          </p>
        </ScrollReveal>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 18 }}>
          {items.map((item, index) => (
            <ScrollReveal key={item.display_name} delay={index * 60}>
              <article
                style={{
                  borderRadius: 22,
                  border: "1px solid rgba(255,255,255,0.06)",
                  background:
                    "linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015))",
                  padding: 22,
                  height: "100%",
                }}
              >
                <div style={{ marginBottom: 16, display: "flex", flexWrap: "wrap", gap: 10 }}>
                  <Badge accent={item.accent ?? "#67e8f9"}>
                    {item.status === "spec_only" ? "Spec Only" : "Measured"}
                  </Badge>
                  {item.external_technology_label ? (
                    <Badge accent="#94a3b8">{item.external_technology_label}</Badge>
                  ) : null}
                </div>
                <h3 style={{ ...body, fontSize: 20, fontWeight: 500, marginBottom: 10 }}>{item.display_name}</h3>
                <p style={{ ...body, color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: 14 }}>
                  {item.archetype ?? item.description}
                </p>
                <div style={{ display: "grid", gap: 10 }}>
                  <div>
                    <div style={{ ...mono, fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", color: item.accent ?? "#67e8f9", marginBottom: 6 }}>
                      Why Included
                    </div>
                    <p style={{ ...body, color: "var(--text-secondary)", lineHeight: 1.65, margin: 0 }}>
                      {item.why_included ?? item.description}
                    </p>
                  </div>
                  <div>
                    <div style={{ ...mono, fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 6 }}>
                      Limitation
                    </div>
                    <p style={{ ...body, color: "var(--text-muted)", lineHeight: 1.65, margin: 0 }}>
                      {item.known_limitations ?? item.fairness_notes}
                    </p>
                  </div>
                </div>
              </article>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  );
}

export function BenchmarksPage() {
  const { data, error } = useBenchmarkSummary();
  const summary = data ?? fallbackBenchmarkSummary();

  return (
    <div style={{ background: "var(--void)", color: "var(--text-primary)" }}>
      <section style={{ paddingTop: 156, paddingBottom: 72, position: "relative" }}>
        <div
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            background:
              "radial-gradient(circle at 20% 20%, rgba(103,232,249,0.10), transparent 32%), radial-gradient(circle at 80% 18%, rgba(167,139,250,0.10), transparent 28%)",
          }}
        />
        <div style={{ maxWidth: 1080, margin: "0 auto", padding: "0 24px", position: "relative" }}>
          <ScrollReveal>
            <Label>Benchmarks</Label>
          </ScrollReveal>
          <ScrollReveal delay={80}>
            <h1
              style={{
                ...serif,
                fontSize: "clamp(2.6rem, 5vw, 4.1rem)",
                lineHeight: 1.08,
                maxWidth: 860,
                marginBottom: 20,
                textWrap: "balance",
              }}
            >
              Benchmarks that measure memory behavior,
              <br />
              not prompt theater.
            </h1>
          </ScrollReveal>
          <ScrollReveal delay={140}>
            <p
              style={{
                ...body,
                maxWidth: 760,
                fontSize: 18,
                lineHeight: 1.75,
                color: "var(--text-secondary)",
                marginBottom: 32,
              }}
            >
              Engram is benchmarked against stronger fair baselines with equal retrieval budgets,
              deterministic scenario grading, and exported artifacts that drive the site directly.
              The headline measured set now includes external memory-system proxies rather than only
              generic notebook and RAG controls.
            </p>
          </ScrollReveal>
          <ScrollReveal delay={200}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
              <Link to="/docs" className="btn btn-primary">
                See Repro Command
              </Link>
              <a
                href="https://github.com/engram-labs/engram"
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-secondary"
              >
                Open Repo
              </a>
            </div>
          </ScrollReveal>
        </div>
      </section>

      {error ? (
        <section style={{ padding: "0 24px 40px" }}>
          <div style={{ maxWidth: 1080, margin: "0 auto" }}>
            <div
              style={{
                borderRadius: 18,
                border: "1px solid rgba(251,191,36,0.24)",
                background: "rgba(251,191,36,0.08)",
                padding: 18,
              }}
            >
              <div style={{ ...mono, fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: "#fbbf24", marginBottom: 8 }}>
                Fallback Metadata
              </div>
              <p style={{ ...body, color: "var(--text-secondary)", margin: 0, lineHeight: 1.7 }}>
                Live benchmark export is missing, so this page is showing bundled baseline metadata
                and architecture descriptions only. Publish fresh results with the export command below.
              </p>
            </div>
          </div>
        </section>
      ) : null}

      <section style={{ padding: "0 24px 88px" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto" }}>
          <ScrollReveal>
            <BenchmarkShowcase data={summary} error={error} />
          </ScrollReveal>
        </div>
      </section>

      <section style={{ padding: "0 24px 88px" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto" }}>
          <ScrollReveal>
            <Label>Methodology</Label>
            <h2 style={{ ...serif, fontSize: "clamp(1.8rem, 3.5vw, 2.8rem)", marginBottom: 18, textWrap: "balance" }}>
              Fairness contract and benchmark shape.
            </h2>
            <p style={{ ...body, color: "var(--text-secondary)", lineHeight: 1.75, maxWidth: 720, marginBottom: 28 }}>
              The benchmark is designed to avoid the usual failure mode of memory demos:
              convenient anecdotes, hidden prompt stuffing, and incomparable baselines.
            </p>
          </ScrollReveal>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 24 }}>
            <ScrollReveal delay={80}>
              <div
                style={{
                  borderRadius: 22,
                  border: "1px solid var(--border)",
                  background: "rgba(255,255,255,0.02)",
                  padding: 24,
                }}
              >
                <ul style={{ listStyle: "none", display: "grid", gap: 14 }}>
                  {METHODOLOGY.map((item) => (
                    <li key={item} style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
                      <span
                        style={{
                          width: 8,
                          height: 8,
                          borderRadius: "50%",
                          background: "var(--accent)",
                          marginTop: 10,
                          flexShrink: 0,
                          boxShadow: "0 0 14px rgba(103,232,249,0.35)",
                        }}
                      />
                      <span style={{ ...body, color: "var(--text-secondary)", lineHeight: 1.7 }}>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </ScrollReveal>
            <ScrollReveal delay={120}>
              <div
                style={{
                  borderRadius: 22,
                  border: "1px solid rgba(103,232,249,0.18)",
                  background:
                    "linear-gradient(180deg, rgba(103,232,249,0.08), rgba(255,255,255,0.02))",
                  padding: 24,
                }}
              >
                <div style={{ ...mono, fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 14 }}>
                  Reproduce
                </div>
                <pre
                  style={{
                    ...mono,
                    margin: 0,
                    whiteSpace: "pre-wrap",
                    color: "var(--text-primary)",
                    fontSize: 13,
                    lineHeight: 1.8,
                  }}
                >
{`cd server
uv run python scripts/benchmark_showcase.py \\
  --track showcase \\
  --mode full \\
  --strict-fairness \\
  --emit-readme-snippet \\
  --website-export-path ../website/public/benchmarks/latest.json`}
                </pre>
              </div>
            </ScrollReveal>
          </div>
        </div>
      </section>

      <BaselineGrid
        title="Headline Measured Competitors"
        description="The systems that anchor the public comparison: Engram plus three measured external memory shapes."
        items={summary.headline_baselines}
      />

      <BaselineGrid
        title="Measured Control Baselines"
        description="These stay visible so the benchmark still shows notebook, summary, and RAG-style controls alongside the headline competitors."
        items={summary.control_baselines}
      />

      <BaselineGrid
        title="Spec-Only Comparison Targets"
        description="These systems matter enough to track on the page, but they are still architectural comparison targets rather than runnable in-suite baselines in this wave."
        items={summary.spec_only_baselines}
      />
    </div>
  );
}
