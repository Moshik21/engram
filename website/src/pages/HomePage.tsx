import { lazy, Suspense, useEffect, useState, type CSSProperties } from "react";
import { Link } from "react-router-dom";
import { BenchmarkShowcase } from "../components/BenchmarkShowcase";
import { ShowcaseTheater } from "../components/ShowcaseTheater";
import { BrainVisualization } from "../components/BrainVisualization";
import { MemoryFlowDiagram } from "../components/MemoryFlowDiagram";
import { ScrollReveal } from "../components/ScrollReveal";

const BrainScene = lazy(async () => ({ default: BrainVisualization }));

const LIFECYCLE = "Capture -> Cue -> Project -> Recall -> Consolidate";

const AGENTS = ["Claude Code", "Cursor", "Windsurf", "Claude Desktop", "OpenClaw"] as const;

const OPERATOR_STEPS = [
  {
    label: "Install",
    command: "curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- helix",
    detail: "Native Helix through PyO3. Full graph, vector, and BM25 memory without Docker.",
  },
  {
    label: "Start",
    command: "engramctl start",
    detail: "Runs the local Engram API and MCP runtime on your machine.",
  },
  {
    label: "Connect",
    command: "engramctl connect claude-code",
    detail: "Writes MCP client config for the local streamable HTTP endpoint.",
  },
  {
    label: "Bootstrap",
    command: "engramctl bootstrap /path/to/project",
    detail: "Indexes user-approved project docs, notes, and exports as cueable context.",
  },
  {
    label: "Inspect",
    command: "engramctl storage",
    detail: "Shows resolved storage paths, disk usage, graph counts, and growth.",
  },
  {
    label: "Verify",
    command: "engramctl doctor",
    detail: "Checks startup readiness, lifecycle state, and local smoke evidence.",
  },
] as const;

const TRUST_BLOCKS = [
  {
    label: "Default backend",
    title: "Native Helix first",
    body: "The public path is PyO3 native Helix: graph, vector, and BM25 in-process with no Docker requirement.",
    evidence: "engramctl quickstart --mode helix",
  },
  {
    label: "Fallbacks",
    title: "Lite and Docker are explicit",
    body: "Lite is the SQLite fallback/demo path. Docker full is the compatibility lane when you specifically need it.",
    evidence: "bash -s -- lite | bash -s -- full",
  },
  {
    label: "Storage",
    title: "You can see where memory lives",
    body: "Native Helix defaults to ~/.helix/engram-native. Lite defaults to ~/.engram/engram.db.",
    evidence: "engramctl storage",
  },
  {
    label: "Extraction",
    title: "No API key for deterministic basics",
    body: "Anthropic improves extraction quality, but the narrow deterministic pipeline can run without an LLM key.",
    evidence: "extraction_provider=auto|narrow",
  },
  {
    label: "Control",
    title: "Removal is visible",
    body: "Uninstall preserves data by default. Purge mode removes local Engram data when the user chooses it.",
    evidence: "engramctl uninstall --purge-data",
  },
  {
    label: "Lifecycle",
    title: "Memory is auditable",
    body: "Doctor, lifecycle, evaluation, storage, and dashboard surfaces show what the brain is doing.",
    evidence: LIFECYCLE,
  },
] as const;

const MEMORY_STEPS = [
  {
    title: "Capture",
    body: "observe and remember write episodes without forcing every turn into durable knowledge.",
  },
  {
    title: "Cue",
    body: "latent traces keep fresh observations searchable before full projection.",
  },
  {
    title: "Project",
    body: "triage and projection extract evidence into entities, facts, and relationships.",
  },
  {
    title: "Recall",
    body: "agents search memories, artifacts, and cues through MCP before answering.",
  },
  {
    title: "Consolidate",
    body: "background phases merge, calibrate, prune, immunize, mature, and reinforce the graph.",
  },
] as const;

const EVIDENCE_PANELS = [
  {
    title: "Storage diagnostics",
    label: "Dashboard / CLI",
    content: [
      "Backend: helix_native",
      "Data: ~/.helix/engram-native",
      "SQLite companion: ~/.engram/engram.db",
      "Growth: +4 episodes, +7 entities, +12 edges",
      "Command: engramctl storage",
    ],
  },
  {
    title: "Doctor readiness",
    label: "Operator gate",
    content: [
      "Mode: helix",
      "API: ready",
      "MCP: ready",
      "Lifecycle: capture/cue/project/recall/consolidate ready",
      "Command: engramctl doctor",
    ],
  },
  {
    title: "MCP connection",
    label: "Agent config",
    content: [
      '"url": "http://127.0.0.1:8100/mcp"',
      '"transport": "streamable-http"',
      "Clients: Claude Code, Cursor, Windsurf, OpenClaw",
      "Command: engramctl connect claude-code",
    ],
  },
] as const;

const OPENCLAW_COMMANDS = [
  "openclaw skills install engram-brain",
  "curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- openclaw",
  "engramctl doctor",
] as const;

const serif: CSSProperties = { fontFamily: '"Instrument Serif", serif', fontStyle: "italic" };
const body: CSSProperties = { fontFamily: '"Outfit", sans-serif' };
const mono: CSSProperties = { fontFamily: '"JetBrains Mono", monospace' };

const section: CSSProperties = {
  position: "relative",
  padding: "clamp(4.5rem, 9vw, 7rem) 1.5rem",
};

const wideContainer: CSSProperties = {
  maxWidth: 1180,
  marginInline: "auto",
};

const narrowContainer: CSSProperties = {
  maxWidth: 760,
  marginInline: "auto",
};

const heading: CSSProperties = {
  ...serif,
  color: "var(--text-primary)",
  lineHeight: 1.14,
  marginBottom: "1.2rem",
};

const bodyText: CSSProperties = {
  ...body,
  fontSize: "1.06rem",
  lineHeight: 1.75,
  color: "var(--text-secondary)",
};

const cardBase: CSSProperties = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-lg)",
  padding: "1.35rem",
};

const codeCard: CSSProperties = {
  ...mono,
  overflow: "hidden",
  borderRadius: "var(--radius-lg)",
  border: "1px solid rgba(103,232,249,0.16)",
  background: "linear-gradient(180deg, rgba(10,12,22,0.96), rgba(3,4,8,0.94))",
  boxShadow: "0 18px 70px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04)",
};

const btnPrimary: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: "0.5rem",
  padding: "0.8rem 1.2rem",
  ...body,
  fontSize: "0.94rem",
  fontWeight: 600,
  lineHeight: 1.35,
  borderRadius: "var(--radius-md)",
  background: "var(--accent)",
  color: "var(--text-inverse)",
  border: "1px solid transparent",
  textDecoration: "none",
  whiteSpace: "nowrap",
};

const btnSecondary: CSSProperties = {
  ...btnPrimary,
  background: "rgba(255,255,255,0.025)",
  color: "var(--text-primary)",
  border: "1px solid rgba(255,255,255,0.08)",
};

function Label({ children }: { children: string }) {
  return (
    <span
      style={{
        ...mono,
        display: "block",
        marginBottom: 14,
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: "0.14em",
        textTransform: "uppercase",
        color: "#67e8f9",
      }}
    >
      {children}
    </span>
  );
}

function BrainBackdropFallback() {
  return (
    <div
      aria-hidden="true"
      style={{
        position: "absolute",
        inset: 0,
        background:
          "linear-gradient(145deg, rgba(103,232,249,0.12), rgba(3,4,8,0.18) 38%, rgba(3,4,8,0.82)), linear-gradient(180deg, rgba(3,4,8,0.0), rgba(3,4,8,0.8))",
        filter: "blur(4px)",
      }}
    />
  );
}

function TerminalBlock({
  label,
  commands,
  footer,
}: {
  label: string;
  commands: readonly string[];
  footer?: string;
}) {
  return (
    <div style={codeCard}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "12px 16px",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
        }}
      >
        <span style={{ fontSize: 10, letterSpacing: "0.13em", textTransform: "uppercase", color: "var(--text-muted)" }}>
          {label}
        </span>
        <span style={{ display: "flex", gap: 5 }}>
          {[0, 1, 2].map((dot) => (
            <span key={dot} style={{ width: 7, height: 7, borderRadius: 999, background: "rgba(255,255,255,0.10)" }} />
          ))}
        </span>
      </div>
      <div style={{ padding: "18px 18px 20px", display: "grid", gap: 10 }}>
        {commands.map((command) => (
          <div key={command} style={{ fontSize: 12.5, lineHeight: 1.65, color: "#67e8f9", wordBreak: "break-word" }}>
            <span style={{ color: "var(--text-muted)", userSelect: "none" }}>$ </span>
            {command}
          </div>
        ))}
      </div>
      {footer ? (
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", padding: "12px 18px", color: "var(--text-muted)", fontSize: 11, lineHeight: 1.6 }}>
          {footer}
        </div>
      ) : null}
    </div>
  );
}

function StatStrip() {
  return (
    <div className="operator-stat-strip">
      {[
        ["Default", "Native Helix / PyO3"],
        ["Fallback", "SQLite lite"],
        ["Agent path", "MCP + OpenClaw"],
        ["Lifecycle", LIFECYCLE],
      ].map(([label, value]) => (
        <div key={label} style={{ minWidth: 0 }}>
          <div style={{ ...mono, fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 4 }}>
            {label}
          </div>
          <div style={{ ...body, fontSize: 14, color: "var(--text-primary)", lineHeight: 1.35 }}>
            {value}
          </div>
        </div>
      ))}
    </div>
  );
}

function StepCard({ step, index }: { step: (typeof OPERATOR_STEPS)[number]; index: number }) {
  return (
    <article style={{ ...cardBase, display: "grid", gap: 14 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span
          style={{
            ...mono,
            display: "grid",
            placeItems: "center",
            width: 30,
            height: 30,
            borderRadius: 8,
            background: "rgba(103,232,249,0.10)",
            color: "#67e8f9",
            fontSize: 12,
          }}
        >
          {String(index + 1).padStart(2, "0")}
        </span>
        <h3 style={{ ...body, fontSize: 17, fontWeight: 600 }}>{step.label}</h3>
      </div>
      <code style={{ ...mono, display: "block", fontSize: 12, lineHeight: 1.6, color: "#67e8f9", wordBreak: "break-word" }}>
        {step.command}
      </code>
      <p style={{ ...body, color: "var(--text-secondary)", fontSize: 14, lineHeight: 1.65, margin: 0 }}>
        {step.detail}
      </p>
    </article>
  );
}

function TrustBlock({ item }: { item: (typeof TRUST_BLOCKS)[number] }) {
  return (
    <article style={{ ...cardBase, minHeight: "100%" }}>
      <div style={{ ...mono, fontSize: 10, letterSpacing: "0.13em", textTransform: "uppercase", color: "#67e8f9", marginBottom: 10 }}>
        {item.label}
      </div>
      <h3 style={{ ...body, fontSize: 18, fontWeight: 600, marginBottom: 10 }}>{item.title}</h3>
      <p style={{ ...body, color: "var(--text-secondary)", lineHeight: 1.65, fontSize: 14, marginBottom: 14 }}>
        {item.body}
      </p>
      <code style={{ ...mono, fontSize: 11.5, lineHeight: 1.6, color: "var(--text-primary)", wordBreak: "break-word" }}>
        {item.evidence}
      </code>
    </article>
  );
}

function EvidencePanel({ panel }: { panel: (typeof EVIDENCE_PANELS)[number] }) {
  return (
    <article style={{ ...codeCard, height: "100%" }}>
      <div style={{ padding: "14px 16px", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
        <div style={{ ...mono, fontSize: 10, color: "#67e8f9", letterSpacing: "0.13em", textTransform: "uppercase", marginBottom: 5 }}>
          {panel.label}
        </div>
        <h3 style={{ ...body, fontSize: 17, fontWeight: 600 }}>{panel.title}</h3>
      </div>
      <div style={{ padding: 16, display: "grid", gap: 10 }}>
        {panel.content.map((line) => (
          <div key={line} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
            <span style={{ width: 7, height: 7, marginTop: 8, borderRadius: 99, background: "var(--accent)", boxShadow: "0 0 12px rgba(103,232,249,0.35)", flexShrink: 0 }} />
            <code style={{ ...mono, color: "var(--text-secondary)", fontSize: 12, lineHeight: 1.6, wordBreak: "break-word" }}>
              {line}
            </code>
          </div>
        ))}
      </div>
    </article>
  );
}

export function HomePage() {
  const [shouldLoadScene, setShouldLoadScene] = useState(false);

  useEffect(() => {
    const onIdle = () => setShouldLoadScene(true);

    if (typeof window === "undefined") return;

    const idleWindow = window as Window & {
      requestIdleCallback?: (callback: IdleRequestCallback, options?: IdleRequestOptions) => number;
      cancelIdleCallback?: (handle: number) => void;
    };

    if (idleWindow.requestIdleCallback) {
      const idleId = idleWindow.requestIdleCallback(onIdle, { timeout: 1000 });
      return () => idleWindow.cancelIdleCallback?.(idleId);
    }

    const timeoutId = globalThis.setTimeout(onIdle, 250);
    return () => globalThis.clearTimeout(timeoutId);
  }, []);

  return (
    <main style={{ position: "relative", overflowX: "hidden", background: "var(--void)", color: "var(--text-primary)" }}>
      <section className="operator-hero">
        <div style={{ position: "absolute", inset: 0, zIndex: 0, opacity: 0.62 }}>
          {shouldLoadScene ? (
            <Suspense fallback={<BrainBackdropFallback />}>
              <BrainScene />
            </Suspense>
          ) : (
            <BrainBackdropFallback />
          )}
        </div>
        <div
          aria-hidden="true"
          style={{
            position: "absolute",
            inset: 0,
            zIndex: 1,
            pointerEvents: "none",
            background:
              "linear-gradient(90deg, rgba(3,4,8,0.96) 0%, rgba(3,4,8,0.84) 48%, rgba(3,4,8,0.52) 100%), linear-gradient(150deg, rgba(103,232,249,0.08), transparent 46%)",
          }}
        />

        <div className="operator-hero-grid" style={{ position: "relative", zIndex: 2 }}>
          <div>
            <ScrollReveal>
              <span className="pill pill-accent" style={{ marginBottom: 22 }}>
                Local long-term memory for AI agents
              </span>
            </ScrollReveal>
            <ScrollReveal delay={80}>
              <h1 className="operator-hero-title" style={{ ...serif, lineHeight: 1.02, marginBottom: 22, maxWidth: 780 }}>
                Install a local brain your agents can actually use.
              </h1>
            </ScrollReveal>
            <ScrollReveal delay={140}>
              <p style={{ ...bodyText, fontSize: "1.15rem", maxWidth: 680, marginBottom: 28 }}>
                Engram runs on your machine, connects through MCP, stores memory in local paths you can inspect, and keeps the memory loop visible from capture through consolidation.
              </p>
            </ScrollReveal>
            <ScrollReveal delay={200}>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 24 }}>
                <Link to="/docs" style={btnPrimary}>
                  Install Engram
                </Link>
                <Link to="/docs#openclaw" style={btnSecondary}>
                  OpenClaw setup
                </Link>
                <Link to="/benchmarks" style={btnSecondary}>
                  Read benchmark method
                </Link>
              </div>
            </ScrollReveal>
            <ScrollReveal delay={260}>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                {AGENTS.map((agent) => (
                  <span key={agent} className="pill">
                    {agent}
                  </span>
                ))}
              </div>
            </ScrollReveal>
          </div>

          <ScrollReveal delay={170}>
            <div className="operator-hero-terminal">
              <TerminalBlock
                label="Recommended startup"
                commands={[
                  "engramctl quickstart --mode helix",
                  "engramctl start",
                  "engramctl status",
                  "engramctl storage",
                  "engramctl doctor",
                  "engramctl connect claude-code",
                  "engramctl bootstrap /path/to/project",
                ]}
                footer="Native Helix/PyO3 is the default no-Docker path. Lite is fallback/demo. Docker full is compatibility."
              />
              <StatStrip />
            </div>
          </ScrollReveal>
        </div>
      </section>

      <section style={{ padding: "1.4rem 1.5rem 4.5rem" }}>
        <div style={wideContainer}>
          <ScrollReveal>
            <div className="mode-band">
              {[
                ["Native Helix", "Recommended", "PyO3 in-process graph, vector, and BM25 with no Docker."],
                ["Lite", "Fallback/demo", "SQLite path for disposable local testing and smoke checks."],
                ["Docker full", "Compatibility", "Explicit FalkorDB + Redis lane when that stack is required."],
                ["OpenClaw", "First-class", "Install the engram-brain skill and connect to the local MCP runtime."],
              ].map(([title, badge, description]) => (
                <div key={title}>
                  <div style={{ ...mono, fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", color: "#67e8f9", marginBottom: 8 }}>{badge}</div>
                  <h2 style={{ ...body, fontSize: 18, fontWeight: 600, marginBottom: 6 }}>{title}</h2>
                  <p style={{ ...body, color: "var(--text-secondary)", fontSize: 13, lineHeight: 1.6, margin: 0 }}>{description}</p>
                </div>
              ))}
            </div>
          </ScrollReveal>
        </div>
      </section>

      <section style={section}>
        <div style={{ ...wideContainer, display: "grid", gap: 34 }}>
          <ScrollReveal>
            <div style={narrowContainer}>
              <Label>Operator path</Label>
              <h2 className="operator-section-title" style={heading}>
                From zero install to inspected memory in six commands.
              </h2>
              <p style={bodyText}>
                Start with the path a new operator actually needs: install the local runtime, connect an agent, bootstrap context, inspect storage, and verify readiness.
              </p>
            </div>
          </ScrollReveal>
          <div className="operator-step-grid">
            {OPERATOR_STEPS.map((step, index) => (
              <ScrollReveal key={step.label} delay={index * 50}>
                <StepCard step={step} index={index} />
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      <section style={section}>
        <div style={wideContainer}>
          <ScrollReveal>
            <div style={{ ...narrowContainer, marginBottom: 34 }}>
              <Label>Trust surface</Label>
              <h2 className="operator-section-title" style={heading}>
                Memory is only useful if the operator can inspect and control it.
              </h2>
              <p style={bodyText}>
                Engram keeps the mechanics visible: backend mode, storage paths, startup checks, project bootstrap, no-Docker defaults, deterministic fallback extraction, and delete-data paths.
              </p>
            </div>
          </ScrollReveal>
          <div className="trust-grid">
            {TRUST_BLOCKS.map((item, index) => (
              <ScrollReveal key={item.title} delay={index * 45}>
                <TrustBlock item={item} />
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      <section style={section}>
        <div style={wideContainer}>
          <ScrollReveal>
            <div style={{ ...narrowContainer, marginBottom: 34 }}>
              <Label>Product evidence</Label>
              <h2 className="operator-section-title" style={heading}>
                Show the runtime, not just the promise.
              </h2>
              <p style={bodyText}>
                Engram exposes concrete runtime signals: dashboard-style storage diagnostics, doctor readiness, and the MCP config agents use to reach the local brain.
              </p>
            </div>
          </ScrollReveal>
          <div className="evidence-grid">
            {EVIDENCE_PANELS.map((panel, index) => (
              <ScrollReveal key={panel.title} delay={index * 80}>
                <EvidencePanel panel={panel} />
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      <section style={section}>
        <div style={wideContainer}>
          <div className="openclaw-band">
            <ScrollReveal>
              <div>
                <Label>OpenClaw</Label>
                <h2 className="operator-section-title" style={heading}>
                  First-class memory for OpenClaw agents.
                </h2>
                <p style={{ ...bodyText, marginBottom: 22 }}>
                  The OpenClaw path installs the public `engram-brain` skill, configures MCP at `http://127.0.0.1:8100/mcp`, uses native Helix by default, and runs `engramctl doctor`.
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
                  <Link to="/docs#openclaw" style={btnPrimary}>
                    OpenClaw install docs
                  </Link>
                  <a href="https://github.com/Moshik21/engram" target="_blank" rel="noopener noreferrer" style={btnSecondary}>
                    View skill source
                  </a>
                </div>
              </div>
            </ScrollReveal>
            <ScrollReveal delay={100}>
              <TerminalBlock label="OpenClaw commands" commands={OPENCLAW_COMMANDS} />
            </ScrollReveal>
          </div>
        </div>
      </section>

      <section style={section}>
        <div style={{ ...wideContainer, display: "grid", gap: 42 }}>
          <ScrollReveal>
            <div style={narrowContainer}>
              <Label>Memory loop</Label>
              <h2 className="operator-section-title" style={heading}>
                {LIFECYCLE}
              </h2>
              <p style={bodyText}>
                Engram starts with cheap episodic capture, uses cues before projection, recalls through MCP, and consolidates offline so the graph improves without blocking active agent work.
              </p>
            </div>
          </ScrollReveal>
          <ScrollReveal delay={100}>
            <MemoryFlowDiagram />
          </ScrollReveal>
          <div className="memory-step-grid">
            {MEMORY_STEPS.map((step, index) => (
              <ScrollReveal key={step.title} delay={index * 50}>
                <article style={cardBase}>
                  <div style={{ ...mono, fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 8 }}>
                    {String(index + 1).padStart(2, "0")}
                  </div>
                  <h3 style={{ ...body, fontSize: 17, fontWeight: 600, marginBottom: 8 }}>{step.title}</h3>
                  <p style={{ ...body, color: "var(--text-secondary)", fontSize: 14, lineHeight: 1.65 }}>{step.body}</p>
                </article>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      <section id="try-it" style={section}>
        <div style={{ ...wideContainer, display: "grid", gap: 32 }}>
          <ScrollReveal>
            <div style={narrowContainer}>
              <Label>Try it</Label>
              <h2 className="operator-section-title" style={heading}>
                Watch memory fire on a seeded brain.
              </h2>
              <p style={bodyText}>
                The public showcase replays a bundled lite demo.db with three beats: Liam continuity, a correction, and a cross-session briefing. Run the same script locally with <code style={mono}>engram showcase run</code>.
              </p>
            </div>
          </ScrollReveal>
          <ScrollReveal delay={80}>
            <div
              style={{
                borderRadius: 16,
                overflow: "hidden",
                border: "1px solid rgba(139, 92, 246, 0.22)",
                boxShadow: "0 24px 60px rgba(8, 8, 20, 0.45)",
                background: "#0c0c18",
              }}
            >
              <video
                autoPlay
                loop
                muted
                playsInline
                poster="/showcase-demo.gif"
                style={{ display: "block", width: "100%", height: "auto" }}
              >
                <source src="/showcase-demo.mp4" type="video/mp4" />
              </video>
            </div>
          </ScrollReveal>
          <ScrollReveal delay={120}>
            <ShowcaseTheater />
          </ScrollReveal>
        </div>
      </section>

      <section style={section}>
        <div style={{ ...wideContainer, display: "grid", gap: 32 }}>
          <ScrollReveal>
            <div style={narrowContainer}>
              <Label>Benchmarks</Label>
              <h2 className="operator-section-title" style={heading}>
                Read the method before the headline.
              </h2>
              <p style={bodyText}>
                Benchmarks use equal retrieval budgets, deterministic scoring, measured controls, and clear spec-only targets so users can read the evidence without taking marketing copy on faith.
              </p>
              <Link to="/benchmarks" style={{ ...btnSecondary, marginTop: 20 }}>
                Read methodology and caveats
              </Link>
            </div>
          </ScrollReveal>
          <ScrollReveal delay={120}>
            <BenchmarkShowcase />
          </ScrollReveal>
        </div>
      </section>

      <section style={section}>
        <div style={narrowContainer}>
          <ScrollReveal>
            <Label>Architecture stays available</Label>
            <h2 className="operator-section-title" style={heading}>
              The science supports onboarding; it does not replace it.
            </h2>
            <p style={{ ...bodyText, marginBottom: 24 }}>
              Engram still has the cognitive architecture story: ACT-R activation, cue-dependent recall, complementary learning systems, reconsolidation, and offline consolidation. The redesign puts installation and trust first, then lets users go deeper when they want the theory.
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
              <Link to="/science" style={btnSecondary}>
                Science and architecture
              </Link>
              <Link to="/roadmap" style={btnSecondary}>
                Roadmap
              </Link>
            </div>
          </ScrollReveal>
        </div>
      </section>

      <section style={{ ...section, paddingBottom: "8rem" }}>
        <div style={{ ...narrowContainer, textAlign: "center" }}>
          <ScrollReveal>
            <h2 className="operator-section-title" style={heading}>
              Install the local brain. Inspect it. Then let agents use it.
            </h2>
            <p style={{ ...bodyText, marginBottom: 32 }}>
              Start with native Helix, connect an MCP client, bootstrap project context, inspect storage, and run doctor before trusting the runtime.
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12, justifyContent: "center" }}>
              <Link to="/docs" style={btnPrimary}>
                Start with docs
              </Link>
              <Link to="/docs#openclaw" style={btnSecondary}>
                Install for OpenClaw
              </Link>
            </div>
          </ScrollReveal>
        </div>
      </section>
    </main>
  );
}
