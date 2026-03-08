import { Link } from "react-router-dom";
import { BenchmarkShowcase } from "../components/BenchmarkShowcase";
import { ScrollReveal } from "../components/ScrollReveal";

const serif = { fontFamily: '"Instrument Serif", Georgia, serif', fontStyle: "italic" as const };
const mono = { fontFamily: '"JetBrains Mono", monospace' };
const body = { fontFamily: '"Outfit", sans-serif' };

const MEASURED_BASELINES = [
  {
    name: "Engram Full",
    status: "Measured",
    accent: "#67e8f9",
    model: "Cue-first long-term memory with episodic capture, graph recall, prospective memory, and consolidation.",
    why: "This is the full system under test and the benchmark headline.",
  },
  {
    name: "Context + Summary",
    status: "Measured",
    accent: "#34d399",
    model: "Recent-turn window plus deterministic rolling summary of older turns.",
    why: "Represents the strongest practical context-window baseline without graph memory.",
  },
  {
    name: "Markdown Canonical",
    status: "Measured",
    accent: "#fbbf24",
    model: "Structured latest-win notebook with lexical retrieval over current facts, corrections, open loops, and intentions.",
    why: "Represents human-readable memory files done as fairly as possible.",
  },
  {
    name: "Hybrid RAG Temporal",
    status: "Measured",
    accent: "#a78bfa",
    model: "Chunk retrieval with lexical + vector fusion and deterministic temporal filtering.",
    why: "Represents modern retrieval memory without a cue layer or prospective graph model.",
  },
] as const;

const EXTERNAL_SPECS = [
  {
    name: "LangGraph Memory",
    status: "Spec Baseline",
    accent: "#67e8f9",
    model: "Thread persistence plus long-term store-backed memory across sessions.",
    strengths: "Strong agent framework default, practical persistence primitives, easy adoption.",
    gaps: "Not inherently cue-driven, contradiction-aware, or graph-consolidated.",
  },
  {
    name: "Mem0 / OpenMemory",
    status: "Spec Baseline",
    accent: "#34d399",
    model: "Agent memory layer focused on extraction, compression, and retrieval over durable user/project memory.",
    strengths: "Direct market competitor in agent memory, strong MCP relevance, easy mental model.",
    gaps: "Closer to managed memory retrieval than consolidation-heavy graph memory.",
  },
  {
    name: "Zep / Graphiti",
    status: "Spec Baseline",
    accent: "#fbbf24",
    model: "Temporal memory API and graph-oriented long-term memory infrastructure.",
    strengths: "Most relevant architectural peer on temporal memory and graph reasoning.",
    gaps: "Benchmark parity requires either direct integration or a closer proxy implementation.",
  },
  {
    name: "Letta",
    status: "Spec Baseline",
    accent: "#a78bfa",
    model: "Pinned memory blocks plus editable agent memory state.",
    strengths: "Well-known stateful-agent design; strong explicit-memory UX.",
    gaps: "Harder to reproduce fairly inside this harness without direct Letta integration.",
  },
  {
    name: "LlamaIndex Memory",
    status: "Appendix Candidate",
    accent: "#fb7185",
    model: "Framework memory queue plus optional long-term extraction and retrieval.",
    strengths: "Widely used in agent stacks, credible framework-native comparison.",
    gaps: "Less direct as a memory-system peer than Mem0 or Graphiti.",
  },
  {
    name: "CrewAI Memory",
    status: "Appendix Candidate",
    accent: "#60a5fa",
    model: "Built-in short-term, long-term, entity, and contextual memory for multi-agent workflows.",
    strengths: "Relevant for orchestration-focused agents and practical production users.",
    gaps: "More workflow-centric than a direct temporal-memory architecture peer.",
  },
] as const;

const METHODOLOGY = [
  "Equal retrieval budgets per scenario and the same transcript order for every baseline.",
  "Deterministic scenario grading on surfaced evidence rather than anecdotal chat demos.",
  "Real Engram system paths through GraphManager rather than fixture-only shortcuts.",
  "Ablation baselines included to isolate cues and search-only behavior.",
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

export function BenchmarksPage() {
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
                maxWidth: 820,
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
                maxWidth: 720,
                fontSize: 18,
                lineHeight: 1.75,
                color: "var(--text-secondary)",
                marginBottom: 32,
              }}
            >
              Engram is benchmarked against stronger fair baselines with equal retrieval budgets,
              deterministic scenario grading, and exported artifacts that drive the site directly.
              The goal is one honest claim: long-horizon agent memory should behave better than raw
              context, notebooks, and retrieval-only systems.
            </p>
          </ScrollReveal>
          <ScrollReveal delay={200}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
              <Link to="/docs" className="btn btn-primary">
                See Repro Command
              </Link>
              <a
                href="https://github.com/Moshik21/engram"
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

      <section style={{ padding: "0 24px 88px" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto" }}>
          <ScrollReveal>
            <BenchmarkShowcase />
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

      <section style={{ padding: "0 24px 88px" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto" }}>
          <ScrollReveal>
            <Label>Measured Baselines</Label>
            <h2 style={{ ...serif, fontSize: "clamp(1.8rem, 3.5vw, 2.8rem)", marginBottom: 18, textWrap: "balance" }}>
              The systems currently measured in-suite.
            </h2>
          </ScrollReveal>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 18 }}>
            {MEASURED_BASELINES.map((item, index) => (
              <ScrollReveal key={item.name} delay={index * 70}>
                <article
                  style={{
                    borderRadius: 22,
                    border: "1px solid var(--border)",
                    background: "rgba(255,255,255,0.02)",
                    padding: 22,
                    height: "100%",
                  }}
                >
                  <div style={{ marginBottom: 16 }}>
                    <Badge accent={item.accent}>{item.status}</Badge>
                  </div>
                  <h3 style={{ ...body, fontSize: 20, fontWeight: 500, marginBottom: 10 }}>{item.name}</h3>
                  <p style={{ ...body, color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: 14 }}>
                    {item.model}
                  </p>
                  <p style={{ ...body, color: "var(--text-muted)", lineHeight: 1.65, marginBottom: 0 }}>
                    {item.why}
                  </p>
                </article>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      <section style={{ padding: "0 24px 112px" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto" }}>
          <ScrollReveal>
            <Label>External Specs</Label>
            <h2 style={{ ...serif, fontSize: "clamp(1.8rem, 3.5vw, 2.8rem)", marginBottom: 18, textWrap: "balance" }}>
              Major memory systems we should compare against.
            </h2>
            <p style={{ ...body, color: "var(--text-secondary)", lineHeight: 1.75, maxWidth: 760, marginBottom: 28 }}>
              These are tracked explicitly so the benchmark page stays honest about what is already
              measured versus what is currently an architectural comparison target.
            </p>
          </ScrollReveal>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 18 }}>
            {EXTERNAL_SPECS.map((item, index) => (
              <ScrollReveal key={item.name} delay={index * 60}>
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
                  <div style={{ marginBottom: 16 }}>
                    <Badge accent={item.accent}>{item.status}</Badge>
                  </div>
                  <h3 style={{ ...body, fontSize: 20, fontWeight: 500, marginBottom: 10 }}>{item.name}</h3>
                  <p style={{ ...body, color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: 14 }}>
                    {item.model}
                  </p>
                  <div style={{ display: "grid", gap: 10 }}>
                    <div>
                      <div style={{ ...mono, fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", color: item.accent, marginBottom: 6 }}>
                        Strengths
                      </div>
                      <p style={{ ...body, color: "var(--text-secondary)", lineHeight: 1.65, margin: 0 }}>
                        {item.strengths}
                      </p>
                    </div>
                    <div>
                      <div style={{ ...mono, fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 6 }}>
                        Gaps vs Engram
                      </div>
                      <p style={{ ...body, color: "var(--text-muted)", lineHeight: 1.65, margin: 0 }}>
                        {item.gaps}
                      </p>
                    </div>
                  </div>
                </article>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
