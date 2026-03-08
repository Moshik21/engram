import { Link } from "react-router-dom";
import { BrainVisualization } from "../components/BrainVisualization";
import { ScrollReveal } from "../components/ScrollReveal";
import { MemoryFlowDiagram } from "../components/MemoryFlowDiagram";
import { ComparisonTable } from "../components/ComparisonTable";
import { BenchmarkShowcase } from "../components/BenchmarkShowcase";
import { FeatureCard } from "../components/FeatureCard";
import { type CSSProperties } from "react";

/* ───────────────────────────── constants ───────────────────────────── */

const PROOF_PILLS = [
  "Private by default",
  "Works with MCP agents",
  "Lite mode or full graph stack",
  "Cue-based memory",
] as const;

const PROBLEM_CARDS = [
  {
    heading: "Chat history is not durable memory.",
    body: "Scrollback disappears between sessions, gets truncated to fit context limits, and carries no structure. It is a log, not a brain.",
  },
  {
    heading: "Bigger context windows are expensive and indiscriminate.",
    body: "Stuffing 200k tokens into every call burns money, slows inference, and treats every past utterance as equally important. Most of it is noise.",
  },
  {
    heading: "Retrieval alone does not create continuity.",
    body: "Vector search finds similar text, but similarity is not relevance. Without cue-based activation, temporal context, and background consolidation, retrieval is just keyword search with extra steps.",
  },
] as const;

const HOW_CARDS = [
  {
    title: "Observe fast",
    description:
      "Store conversations immediately without forcing expensive extraction. Most turns are low-signal \u2014 capture them cheaply and let consolidation decide what matters later.",
  },
  {
    title: "Recall with cues",
    description:
      "Surface latent memory traces before full projection. Cue-based recall activates relevant memories using spreading activation, not just embedding similarity.",
  },
  {
    title: "Learn from actual use",
    description:
      "Distinguish surfaced from used memory. When the agent actually references a recalled fact, that feedback strengthens the trace. Ignored recalls decay naturally.",
  },
  {
    title: "Consolidate offline",
    description:
      "Triage, merge, infer, mature, and prune in the background. Twelve consolidation phases run on a tiered schedule \u2014 sharpening the graph without blocking inference.",
  },
] as const;

const CONTINUITY_ITEMS = [
  "The agent remembers relevant facts without you repeating them.",
  "Lower-value turns stay latent until they prove useful.",
  "Important memories become structured and durable.",
  "Old noise fades instead of polluting every future recall.",
  "Memory quality improves over weeks, not just within one session.",
] as const;

const USE_CASES = [
  {
    title: "Coding agents",
    description:
      "Architecture decisions, naming conventions, past bugs, library preferences \u2014 recalled automatically across sessions instead of re-explained every time.",
  },
  {
    title: "Personal AI",
    description:
      "Health context, relationship notes, goals, ongoing threads. One private brain that accumulates understanding of your life over months.",
  },
  {
    title: "Research & writing",
    description:
      "Source material, evolving arguments, reviewer feedback, style preferences. Memory that matures alongside the work itself.",
  },
  {
    title: "Team assistants",
    description:
      "Project timelines, decision history, stakeholder preferences, recurring patterns. Continuity that survives member turnover.",
  },
] as const;

/* ────────────────────────────── styles ─────────────────────────────── */

const serif: CSSProperties = { fontFamily: '"Instrument Serif", serif', fontStyle: "italic" };
const body: CSSProperties = { fontFamily: '"Outfit", sans-serif' };
const mono: CSSProperties = { fontFamily: '"JetBrains Mono", monospace' };

const section: CSSProperties = {
  position: "relative",
  padding: "7rem 1.5rem",
};

const narrowContainer: CSSProperties = {
  maxWidth: 720,
  marginInline: "auto",
};

const wideContainer: CSSProperties = {
  maxWidth: 1100,
  marginInline: "auto",
};

const heading: CSSProperties = {
  ...serif,
  color: "var(--text-primary)",
  lineHeight: 1.2,
  marginBottom: "1.5rem",
};

const bodyText: CSSProperties = {
  fontSize: "1.125rem",
  lineHeight: 1.7,
  color: "var(--text-secondary)",
};

const cardBase: CSSProperties = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-lg)",
  padding: "2rem",
  height: "100%",
  transition: "border-color 300ms ease, box-shadow 300ms ease, transform 300ms ease",
};

const glowCard: CSSProperties = {
  background: "var(--surface)",
  border: "1px solid rgba(103, 232, 249, 0.2)",
  borderRadius: "var(--radius-lg)",
  backdropFilter: "blur(24px) saturate(1.2)",
  boxShadow: "0 0 24px rgba(103, 232, 249, 0.06)",
  padding: "2rem 2.5rem",
};

const btnPrimary: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: "0.5rem",
  padding: "0.75rem 1.75rem",
  ...body,
  fontSize: "0.9375rem",
  fontWeight: 500,
  lineHeight: 1.4,
  borderRadius: "var(--radius-md)",
  background: "var(--accent)",
  color: "var(--text-inverse)",
  border: "1px solid transparent",
  boxShadow: "0 1px 2px rgba(0,0,0,0.3), 0 0 16px rgba(103,232,249,0.06)",
  textDecoration: "none",
  whiteSpace: "nowrap",
};

const btnSecondary: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: "0.5rem",
  padding: "0.75rem 1.75rem",
  ...body,
  fontSize: "0.9375rem",
  fontWeight: 500,
  lineHeight: 1.4,
  borderRadius: "var(--radius-md)",
  background: "transparent",
  color: "var(--text-primary)",
  border: "1px solid rgba(255,255,255,0.08)",
  textDecoration: "none",
  whiteSpace: "nowrap",
};

/* ────────────────────────────── helpers ────────────────────────────── */

function ChevronDown() {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ color: "var(--text-muted)" }}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

function AccentCheck() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      style={{ flexShrink: 0, marginTop: 2 }}
    >
      <circle cx="10" cy="10" r="10" fill="var(--accent)" fillOpacity="0.12" />
      <path
        d="M6 10.5l2.5 2.5L14 7.5"
        stroke="var(--accent)"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   HOMEPAGE
   ═══════════════════════════════════════════════════════════════════════ */

export function HomePage() {
  return (
    <main style={{ position: "relative", overflowX: "hidden", background: "var(--void)", color: "var(--text-primary)" }}>
      {/* ── SECTION 1: HERO ──────────────────────────────────────────── */}
      <section style={{ position: "relative", minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", overflow: "hidden" }}>
        {/* Brain visualization background */}
        <div style={{ position: "absolute", inset: 0, zIndex: 0 }}>
          <BrainVisualization />
        </div>

        {/* Vignette overlay */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            zIndex: 1,
            pointerEvents: "none",
            background: "radial-gradient(ellipse 80% 70% at 50% 45%, transparent 0%, var(--void) 100%)",
          }}
        />

        {/* Content */}
        <div style={{ position: "relative", zIndex: 2, display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", padding: "0 1.5rem", maxWidth: 780, marginInline: "auto" }}>
          <ScrollReveal>
            <span className="pill" style={{ ...mono, fontSize: "0.75rem", letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 24, display: "inline-block" }}>
              Private Memory for AI Agents
            </span>
          </ScrollReveal>

          <ScrollReveal delay={100}>
            <h1 style={{ ...serif, fontSize: "clamp(2.5rem, 5.5vw, 4rem)", lineHeight: 1.12, marginBottom: 24, color: "var(--text-primary)" }}>
              AI needs a brain,
              <br style={{ display: "none" }} /> not just a context window.
            </h1>
          </ScrollReveal>

          <ScrollReveal delay={200}>
            <p style={{ ...body, fontSize: "clamp(1.125rem, 2vw, 1.25rem)", lineHeight: 1.7, color: "var(--text-secondary)", maxWidth: 640, marginBottom: 40 }}>
              Engram gives AI agents private long-term memory: episodic capture,
              cue-based recall, structured knowledge, and background
              consolidation that improves continuity over time.
            </p>
          </ScrollReveal>

          <ScrollReveal delay={300}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 16, justifyContent: "center", marginBottom: 48 }}>
              <Link to="/docs" style={btnPrimary}>
                Get Started
              </Link>
              <Link to="/science" style={btnSecondary}>
                Read The Architecture
              </Link>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={400}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12, justifyContent: "center" }}>
              {PROOF_PILLS.map((label) => (
                <span
                  key={label}
                  className="pill"
                  style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}
                >
                  {label}
                </span>
              ))}
            </div>
          </ScrollReveal>
        </div>

        {/* Scroll indicator */}
        <div className="animate-bounce" style={{ position: "absolute", bottom: 32, zIndex: 2, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
          <ChevronDown />
          <ChevronDown />
        </div>
      </section>

      {/* ── SECTION 2: THE PROBLEM ───────────────────────────────────── */}
      <section style={section}>
        <div style={{ ...narrowContainer, marginBottom: 64 }}>
          <ScrollReveal>
            <h2 style={{ ...heading, fontSize: "clamp(1.75rem, 3.5vw, 2.75rem)" }}>
              Most AI is smart in the moment and forgetful across time.
            </h2>
          </ScrollReveal>
          <ScrollReveal delay={100}>
            <p style={bodyText}>
              Today&rsquo;s agents lose everything between sessions. They
              re-discover your preferences, re-learn your codebase, and
              re-ask questions you answered last week. The continuity
              problem is not about intelligence &mdash; it is about
              memory architecture.
            </p>
          </ScrollReveal>
        </div>

        <div style={wideContainer}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 24 }}>
            {PROBLEM_CARDS.map((card, i) => (
              <ScrollReveal key={card.heading} delay={i * 120}>
                <div style={{ ...cardBase, display: "flex", flexDirection: "column" }}>
                  <div className="accent-bar" style={{ marginBottom: 16 }} />
                  <h3 style={{ ...body, fontSize: "1.125rem", fontWeight: 500, marginBottom: 12, color: "var(--text-primary)" }}>
                    {card.heading}
                  </h3>
                  <p style={{ fontSize: "0.9375rem", lineHeight: 1.7, color: "var(--text-secondary)", flex: 1 }}>
                    {card.body}
                  </p>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* ── SECTION 3: THE THESIS ────────────────────────────────────── */}
      <section style={section}>
        <div style={narrowContainer}>
          <ScrollReveal>
            <h2 style={{ ...heading, fontSize: "clamp(2rem, 4vw, 3rem)", marginBottom: 32 }}>
              One brain per person.
            </h2>
          </ScrollReveal>

          <ScrollReveal delay={100}>
            <p style={{ ...bodyText, marginBottom: 40 }}>
              Engram is built around a simple model: each person gets one
              private brain for their AI. Work, projects, goals,
              preferences, relationships, health context, and ongoing
              threads belong to the same private memory graph. Projects
              become neighborhoods in memory, not hard silos that block
              useful connections.
            </p>
          </ScrollReveal>

          <ScrollReveal delay={200}>
            <div style={glowCard}>
              <p style={{ ...serif, fontSize: "clamp(1.25rem, 2.5vw, 1.5rem)", lineHeight: 1.4, color: "var(--accent)" }}>
                &ldquo;Projects are topology, not tenancy.&rdquo;
              </p>
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* ── SECTION 4: HOW IT WORKS ──────────────────────────────────── */}
      <section style={section}>
        <div style={{ ...narrowContainer, marginBottom: 64 }}>
          <ScrollReveal>
            <h2 style={{ ...heading, fontSize: "clamp(1.75rem, 3.5vw, 2.5rem)" }}>
              Memory that starts cheap and becomes structured only when it
              earns it.
            </h2>
          </ScrollReveal>
        </div>

        <div style={{ ...wideContainer, marginBottom: 64 }}>
          <ScrollReveal delay={100}>
            <MemoryFlowDiagram />
          </ScrollReveal>
        </div>

        <div style={wideContainer}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 24 }}>
            {HOW_CARDS.map((card, i) => (
              <ScrollReveal key={card.title} delay={i * 100}>
                <FeatureCard
                  title={card.title}
                  description={card.description}
                />
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* ── SECTION 5: COMPARISON ────────────────────────────────────── */}
      <section style={section}>
        <div style={{ ...narrowContainer, marginBottom: 64 }}>
          <ScrollReveal>
            <h2 style={{ ...heading, fontSize: "clamp(1.75rem, 3.5vw, 2.5rem)" }}>
              Built like a memory system, not a prompt stuffing trick.
            </h2>
          </ScrollReveal>
        </div>

        <div style={wideContainer}>
          <ScrollReveal delay={100}>
            <ComparisonTable />
          </ScrollReveal>
        </div>
      </section>

      {/* ── SECTION 6: BENCHMARK ─────────────────────────────────────── */}
      <section style={section}>
        <div style={{ ...narrowContainer, marginBottom: 48 }}>
          <ScrollReveal>
            <h2 style={{ ...heading, fontSize: "clamp(1.75rem, 3.5vw, 2.5rem)" }}>
              Measured against fair baselines, not just described.
            </h2>
          </ScrollReveal>
          <ScrollReveal delay={80}>
            <p style={{ ...bodyText, marginBottom: 0 }}>
              The benchmark section below renders exported benchmark artifacts from the real suite: stronger baseline comparisons, fixed retrieval budgets, and deterministic scenario scoring.
            </p>
          </ScrollReveal>
          <ScrollReveal delay={140}>
            <div style={{ marginTop: 20 }}>
              <Link
                to="/benchmarks"
                style={{
                  ...body,
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 10,
                  color: "var(--accent)",
                  textDecoration: "none",
                  fontSize: "0.98rem",
                }}
              >
                Open the full benchmark page
                <span style={{ ...mono, fontSize: 12, opacity: 0.75 }}>methodology + baseline specs</span>
              </Link>
            </div>
          </ScrollReveal>
        </div>

        <div style={wideContainer}>
          <ScrollReveal delay={120}>
            <BenchmarkShowcase />
          </ScrollReveal>
        </div>
      </section>

      {/* ── SECTION 7: WHAT YOU GET ──────────────────────────────────── */}
      <section style={section}>
        <div style={narrowContainer}>
          <ScrollReveal>
            <h2 style={{ ...heading, fontSize: "clamp(1.75rem, 3.5vw, 2.5rem)", marginBottom: 48 }}>
              What continuity feels like in practice.
            </h2>
          </ScrollReveal>

          <ul style={{ listStyle: "none", display: "flex", flexDirection: "column", gap: 24 }}>
            {CONTINUITY_ITEMS.map((item, i) => (
              <ScrollReveal key={item} delay={i * 80}>
                <li style={{ display: "flex", alignItems: "flex-start", gap: 16 }}>
                  <AccentCheck />
                  <span style={{ ...bodyText }}>
                    {item}
                  </span>
                </li>
              </ScrollReveal>
            ))}
          </ul>
        </div>
      </section>

      {/* ── SECTION 8: USE CASES ─────────────────────────────────────── */}
      <section style={section}>
        <div style={{ ...narrowContainer, marginBottom: 64 }}>
          <ScrollReveal>
            <h2 style={{ ...heading, fontSize: "clamp(1.75rem, 3.5vw, 2.5rem)" }}>
              For agents that need continuity, not just recall.
            </h2>
          </ScrollReveal>
        </div>

        <div style={wideContainer}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 24 }}>
            {USE_CASES.map((uc, i) => (
              <ScrollReveal key={uc.title} delay={i * 100}>
                <div style={cardBase}>
                  <h3 style={{ ...body, fontSize: "1.125rem", fontWeight: 500, marginBottom: 12, color: "var(--text-primary)" }}>
                    {uc.title}
                  </h3>
                  <p style={{ fontSize: "0.9375rem", lineHeight: 1.7, color: "var(--text-secondary)" }}>
                    {uc.description}
                  </p>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* ── SECTION 8: PRIVACY ───────────────────────────────────────── */}
      <section style={section}>
        <div style={narrowContainer}>
          <ScrollReveal>
            <h2 style={{ ...heading, fontSize: "clamp(1.75rem, 3.5vw, 2.5rem)", marginBottom: 32 }}>
              If memory gets more useful, privacy gets more important.
            </h2>
          </ScrollReveal>

          <ScrollReveal delay={100}>
            <p style={{ ...bodyText, marginBottom: 24 }}>
              Engram runs locally or on infrastructure you control. Each
              brain is sovereign &mdash; owned by its person, stored where
              they choose, deletable at any time. There is no shared
              memory pool, no cross-user training, no silent data
              harvesting.
            </p>
          </ScrollReveal>

          <ScrollReveal delay={150}>
            <p style={{ ...bodyText, marginBottom: 40 }}>
              Future federated learning should work the same way
              differential privacy works in other domains: aggregate
              patterns from many brains without ever reading individual
              memories. The policy layer travels with the brain, not the
              platform.
            </p>
          </ScrollReveal>

          <ScrollReveal delay={250}>
            <div style={glowCard}>
              <p style={{ ...serif, fontSize: "clamp(1.25rem, 2.5vw, 1.5rem)", lineHeight: 1.4, color: "var(--accent)" }}>
                &ldquo;Learn from many private brains without reading
                their thoughts.&rdquo;
              </p>
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* ── SECTION 9: CTA ───────────────────────────────────────────── */}
      <section style={section}>
        {/* Accent glow */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            background: "radial-gradient(ellipse 60% 50% at 50% 60%, rgba(103,232,249,0.06) 0%, transparent 70%)",
          }}
        />

        <div style={{ ...narrowContainer, position: "relative", textAlign: "center" }}>
          <ScrollReveal>
            <h2 style={{ ...heading, fontSize: "clamp(2rem, 4.5vw, 3.25rem)", lineHeight: 1.15, marginBottom: 24 }}>
              Give your AI continuity.
            </h2>
          </ScrollReveal>

          <ScrollReveal delay={100}>
            <p style={{ ...bodyText, marginBottom: 40, maxWidth: 540, marginInline: "auto" }}>
              Start in lite mode with a single SQLite file. Scale to
              FalkorDB and Redis when you need the full graph stack. No
              vendor lock-in, no cloud dependency, no shared memory pool.
            </p>
          </ScrollReveal>

          <ScrollReveal delay={200}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 16, justifyContent: "center" }}>
              <Link to="/docs" style={btnPrimary}>
                Get Started
              </Link>
              <Link to="/vision" style={btnSecondary}>
                Read The Vision
              </Link>
              <Link to="/roadmap" style={btnSecondary}>
                View Roadmap
              </Link>
            </div>
          </ScrollReveal>
        </div>
      </section>
    </main>
  );
}
