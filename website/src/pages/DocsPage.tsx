import { useState } from "react";
import { Link } from "react-router-dom";
import { ScrollReveal } from "../components/ScrollReveal";

/* ───────────────────────────── constants ───────────────────────────── */

const INSTALL_TABS = [
  {
    id: "mcp",
    label: "MCP Server",
    badge: "Recommended",
    heading: "MCP SERVER",
    commands: ["cd server", "uv sync", "uv run engram mcp"],
    description:
      "Starts Engram in lite mode with SQLite. Zero dependencies beyond Python. Connect any MCP-compatible agent and start building memory immediately.",
  },
  {
    id: "docker",
    label: "Docker Full Stack",
    badge: null,
    heading: "DOCKER",
    commands: ["docker compose up -d --build"],
    description:
      "Full mode with FalkorDB, Redis, and the real-time dashboard at localhost:3000. Best for production deployments and graph exploration.",
  },
  {
    id: "rest",
    label: "REST API",
    badge: null,
    heading: "REST API",
    commands: ["cd server", "uv run engram serve"],
    description:
      "Lightweight HTTP server for non-MCP integrations. Interactive docs available at localhost:8100/docs.",
  },
] as const;

const MCP_TOOLS = [
  { name: "observe", desc: "Store raw text for background processing", cat: "capture" },
  { name: "remember", desc: "Store important information with full extraction", cat: "capture" },
  { name: "adjudicate_evidence", desc: "Resolve ambiguous entity or relationship evidence", cat: "capture" },
  { name: "forget", desc: "Remove outdated information", cat: "capture" },
  { name: "bootstrap_project", desc: "Auto-observe key project files", cat: "capture" },
  { name: "recall", desc: "Retrieve relevant memories", cat: "retrieval" },
  { name: "search_entities", desc: "Look up specific entities", cat: "retrieval" },
  { name: "search_facts", desc: "Find specific facts and relationships", cat: "retrieval" },
  { name: "search_artifacts", desc: "Search bootstrapped project artifacts", cat: "retrieval" },
  { name: "get_context", desc: "Get broad overview of what you know", cat: "retrieval" },
  { name: "route_question", desc: "Classify questions for epistemic routing", cat: "retrieval" },
  { name: "mark_identity_core", desc: "Protect important entities from pruning", cat: "management" },
  { name: "intend", desc: "Create prospective memory intentions", cat: "management" },
  { name: "dismiss_intention", desc: "Disable an active intention", cat: "management" },
  { name: "list_intentions", desc: "List active intentions with warmth info", cat: "management" },
  { name: "get_consolidation_status", desc: "Check consolidation state", cat: "system" },
  { name: "trigger_consolidation", desc: "Run consolidation manually", cat: "system" },
  { name: "get_graph_state", desc: "Inspect the knowledge graph", cat: "system" },
  { name: "get_runtime_state", desc: "Check effective mode, profiles, and flags", cat: "system" },
] as const;

const CATEGORIES: { key: string; label: string; color: string }[] = [
  { key: "capture", label: "Capture", color: "#67e8f9" },
  { key: "retrieval", label: "Retrieval", color: "#34d399" },
  { key: "management", label: "Management", color: "#a78bfa" },
  { key: "system", label: "System", color: "#fbbf24" },
];

const CONCEPT_CARDS = [
  { title: "Episodes", desc: "Raw conversation turns stored as temporal events. Low-cost capture first, extraction later." },
  { title: "Entities", desc: "People, projects, concepts extracted into a knowledge graph that matures over time." },
  { title: "Relationships", desc: "Typed edges between entities: WORKS_AT, PREFERS, KNOWS. ~25 canonical predicates." },
  { title: "Activation", desc: "ACT-R inspired recency/frequency ranking. Computed lazily from access history." },
  { title: "Consolidation", desc: "15 offline phases on a three-tier schedule. Triage, merge, infer, adjudicate, replay, prune, compact, mature, semanticize, schema, reindex, graph embed, microglia, dream." },
  { title: "Cues", desc: "Lightweight latent memory traces that surface relevant context before full extraction." },
] as const;

const PHASES = [
  { name: "triage", tier: "Hot", interval: "15 min", desc: "Score queued episodes, promote top ~35% for extraction" },
  { name: "merge", tier: "Warm", interval: "2 hr", desc: "Fuzzy-match duplicate entities via multi-signal scoring" },
  { name: "infer", tier: "Warm", interval: "2 hr", desc: "Create edges for co-occurring entities via PMI" },
  { name: "evidence_adjudicate", tier: "Warm", interval: "2 hr", desc: "Resolve ambiguous entity and relationship evidence" },
  { name: "edge_adjudicate", tier: "Warm", interval: "2 hr", desc: "Budgeted offline adjudication of unresolved edges" },
  { name: "replay", tier: "Cold", interval: "6 hr", desc: "Re-extract recent episodes to find missed entities" },
  { name: "prune", tier: "Cold", interval: "6 hr", desc: "Soft-delete dead entities with low access" },
  { name: "compact", tier: "Warm", interval: "2 hr", desc: "Logarithmic bucketing of access history" },
  { name: "mature", tier: "Warm", interval: "2 hr", desc: "Graduate entities through memory tiers" },
  { name: "semanticize", tier: "Warm", interval: "2 hr", desc: "Promote episodes based on entity coverage" },
  { name: "schema", tier: "Cold", interval: "6 hr", desc: "Detect recurring structural motifs" },
  { name: "reindex", tier: "Warm", interval: "2 hr", desc: "Re-embed entities affected by earlier phases" },
  { name: "graph_embed", tier: "Cold", interval: "6 hr", desc: "Train structural embeddings (Node2Vec, TransE)" },
  { name: "microglia", tier: "Warm", interval: "2 hr", desc: "Graph immune surveillance — prune bad edges, fix summaries" },
  { name: "dream", tier: "Cold", interval: "6 hr", desc: "Spreading activation + cross-domain connections" },
] as const;

const CONFIG_FIELDS = [
  { field: "consolidation_profile", values: "off | observe | conservative | standard", desc: "Controls which consolidation phases run. Off by default." },
  { field: "recall_profile", values: "off | wave1 | wave2 | wave3 | wave4 | all", desc: "Controls retrieval pipeline depth. Each wave adds more signal." },
  { field: "extraction_provider", values: "auto | anthropic | ollama | narrow", desc: "Extraction backend. Auto tries Anthropic, then Ollama, then deterministic narrow pipeline." },
  { field: "ANTHROPIC_API_KEY", values: "sk-ant-...", desc: "Optional. Enables LLM-backed extraction. Without it, the narrow deterministic pipeline handles extraction." },
] as const;

/* ──────────────────────────── sub-components ──────────────────────── */

const serif = { fontFamily: '"Instrument Serif", Georgia, serif', fontStyle: "italic" as const };
const mono = { fontFamily: '"JetBrains Mono", monospace' };
const body = { fontFamily: '"Outfit", sans-serif' };

function Label({ children }: { children: string }) {
  return (
    <span
      style={{ ...mono, fontSize: 11, fontWeight: 500, letterSpacing: "0.12em", textTransform: "uppercase" as const, color: "var(--text-muted)", display: "block", marginBottom: 20 }}
    >
      {children}
    </span>
  );
}

function Heading({ children }: { children: React.ReactNode }) {
  return (
    <h2
      style={{ ...serif, fontSize: "clamp(1.75rem, 3.5vw, 2.5rem)", lineHeight: 1.2, marginBottom: 20 }}
    >
      {children}
    </h2>
  );
}

function TierBadge({ tier }: { tier: string }) {
  const c = tier === "Hot" ? "#f97316" : tier === "Warm" ? "#fbbf24" : "#67e8f9";
  const bg = tier === "Hot" ? "rgba(249,115,22,0.12)" : tier === "Warm" ? "rgba(251,191,36,0.10)" : "rgba(103,232,249,0.10)";
  return (
    <span
      style={{ ...mono, fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase" as const, padding: "3px 8px", borderRadius: 99, background: bg, color: c }}
    >
      {tier}
    </span>
  );
}

/* ════════════════════════════ DOCS PAGE ════════════════════════════ */

export function DocsPage() {
  const [tab, setTab] = useState("mcp");
  const active = INSTALL_TABS.find((t) => t.id === tab) ?? INSTALL_TABS[0];

  return (
    <main style={{ background: "var(--void)", color: "var(--text-primary)", overflowX: "hidden" }}>

      {/* ── HERO ─────────────────────────────────────────────── */}
      <section style={{ paddingTop: 160, paddingBottom: 80 }}>
        <div style={{ maxWidth: 800, margin: "0 auto", padding: "0 24px" }}>
          <ScrollReveal>
            <Label>DOCUMENTATION</Label>
          </ScrollReveal>
          <ScrollReveal delay={80}>
            <h1 style={{ ...serif, fontSize: "clamp(2.5rem, 5vw, 3.5rem)", lineHeight: 1.15, marginBottom: 16 }}>
              Get Started with Engram
            </h1>
          </ScrollReveal>
          <ScrollReveal delay={160}>
            <p style={{ ...body, fontSize: 18, lineHeight: 1.7, color: "var(--text-secondary)", maxWidth: 520 }}>
              Set up private AI memory in minutes. Three options, zero complexity.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* ── QUICK START ──────────────────────────────────────── */}
      <section style={{ paddingTop: 40, paddingBottom: 96 }}>
        <div style={{ maxWidth: 800, margin: "0 auto", padding: "0 24px" }}>
          <ScrollReveal>
            <Label>QUICK START</Label>
            <Heading>Three ways to run Engram</Heading>
          </ScrollReveal>

          {/* Tabs */}
          <ScrollReveal delay={80}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 24 }}>
              {INSTALL_TABS.map((t) => {
                const isActive = tab === t.id;
                return (
                  <button
                    key={t.id}
                    onClick={() => setTab(t.id)}
                    style={{
                      ...mono,
                      fontSize: 12,
                      fontWeight: 500,
                      padding: "10px 20px",
                      borderRadius: 10,
                      border: `1px solid ${isActive ? "rgba(103,232,249,0.3)" : "var(--border)"}`,
                      background: isActive ? "rgba(103,232,249,0.08)" : "var(--surface)",
                      color: isActive ? "#67e8f9" : "var(--text-secondary)",
                      cursor: "pointer",
                      transition: "all 0.2s",
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                    }}
                  >
                    {t.label}
                    {t.badge && (
                      <span
                        style={{
                          ...mono,
                          fontSize: 9,
                          letterSpacing: "0.08em",
                          textTransform: "uppercase",
                          padding: "2px 7px",
                          borderRadius: 99,
                          background: "rgba(103,232,249,0.15)",
                          color: "#67e8f9",
                        }}
                      >
                        {t.badge}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </ScrollReveal>

          {/* Code card */}
          <ScrollReveal delay={120}>
            <div
              style={{
                borderRadius: 14,
                border: "1px solid var(--border)",
                overflow: "hidden",
                background: "var(--surface-solid)",
              }}
            >
              {/* Code header */}
              <div
                style={{
                  padding: "12px 20px",
                  borderBottom: "1px solid var(--border)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <span style={{ ...mono, fontSize: 10, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--text-muted)" }}>
                  {active.heading}
                </span>
                <div style={{ display: "flex", gap: 5 }}>
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: "rgba(255,255,255,0.06)" }} />
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: "rgba(255,255,255,0.06)" }} />
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: "rgba(255,255,255,0.06)" }} />
                </div>
              </div>

              {/* Code body */}
              <div style={{ padding: "24px 24px 28px" }}>
                {active.commands.map((cmd, i) => (
                  <div key={i} style={{ ...mono, fontSize: 14, lineHeight: 2.2 }}>
                    <span style={{ color: "var(--text-muted)", userSelect: "none" }}>$ </span>
                    <span style={{ color: "#67e8f9" }}>{cmd}</span>
                  </div>
                ))}
              </div>

              {/* Description */}
              <div
                style={{
                  padding: "16px 24px 20px",
                  borderTop: "1px solid var(--border)",
                  background: "rgba(255,255,255,0.01)",
                }}
              >
                <p style={{ ...body, fontSize: 14, lineHeight: 1.7, color: "var(--text-secondary)", margin: 0 }}>
                  {active.description}
                </p>
              </div>
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* ── MCP TOOLS ────────────────────────────────────────── */}
      <section style={{ paddingTop: 64, paddingBottom: 96 }}>
        <div style={{ maxWidth: 800, margin: "0 auto", padding: "0 24px" }}>
          <ScrollReveal>
            <Label>MCP TOOLS</Label>
            <Heading>19 tools for memory management</Heading>
            <p style={{ ...body, fontSize: 15, lineHeight: 1.7, color: "var(--text-secondary)", marginBottom: 40, maxWidth: 600 }}>
              Engram exposes a complete memory interface through the Model Context Protocol. Every tool is available via stdio transport.
            </p>
          </ScrollReveal>

          {CATEGORIES.map((cat, gi) => {
            const tools = MCP_TOOLS.filter((t) => t.cat === cat.key);
            return (
              <ScrollReveal key={cat.key} delay={gi * 60}>
                <div style={{ marginBottom: 32 }}>
                  {/* Category header */}
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                    <span style={{ width: 8, height: 8, borderRadius: "50%", background: cat.color }} />
                    <span style={{ ...mono, fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: cat.color }}>
                      {cat.label}
                    </span>
                  </div>

                  {/* Tool cards */}
                  <div style={{ display: "grid", gap: 6 }}>
                    {tools.map((tool) => (
                      <div
                        key={tool.name}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 16,
                          padding: "14px 20px",
                          borderRadius: 10,
                          background: "rgba(255,255,255,0.02)",
                          border: "1px solid var(--border)",
                          transition: "border-color 0.2s, background 0.2s",
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
                          e.currentTarget.style.background = "rgba(255,255,255,0.03)";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.borderColor = "var(--border)";
                          e.currentTarget.style.background = "rgba(255,255,255,0.02)";
                        }}
                      >
                        <code style={{ ...mono, fontSize: 13, fontWeight: 500, color: cat.color, flexShrink: 0, minWidth: 160 }}>
                          {tool.name}
                        </code>
                        <span style={{ ...body, fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                          {tool.desc}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </ScrollReveal>
            );
          })}
        </div>
      </section>

      {/* ── ARCHITECTURE ─────────────────────────────────────── */}
      <section style={{ paddingTop: 64, paddingBottom: 96 }}>
        <div style={{ maxWidth: 800, margin: "0 auto", padding: "0 24px" }}>
          <ScrollReveal>
            <Label>ARCHITECTURE</Label>
            <Heading>Dual mode, CQRS, offline consolidation</Heading>
          </ScrollReveal>

          {/* Mode cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16, marginBottom: 48 }}>
            <ScrollReveal delay={60}>
              <div style={{ padding: 24, borderRadius: 14, background: "var(--surface)", border: "1px solid var(--border)", height: "100%" }}>
                <h3 style={{ ...body, fontSize: 16, fontWeight: 600, marginBottom: 12 }}>Dual Mode</h3>
                <p style={{ ...body, fontSize: 13, lineHeight: 1.7, color: "var(--text-secondary)", marginBottom: 8 }}>
                  <strong style={{ color: "var(--text-primary)" }}>Lite:</strong> SQLite only. Single file, zero infrastructure. Great for local agents and development.
                </p>
                <p style={{ ...body, fontSize: 13, lineHeight: 1.7, color: "var(--text-secondary)" }}>
                  <strong style={{ color: "var(--text-primary)" }}>Full:</strong> FalkorDB + Redis. Graph database, vector search, real-time dashboard. Production-ready.
                </p>
              </div>
            </ScrollReveal>

            <ScrollReveal delay={120}>
              <div style={{ padding: 24, borderRadius: 14, background: "var(--surface)", border: "1px solid var(--border)", height: "100%" }}>
                <h3 style={{ ...body, fontSize: 16, fontWeight: 600, marginBottom: 12 }}>CQRS Pattern</h3>
                <p style={{ ...body, fontSize: 13, lineHeight: 1.7, color: "var(--text-secondary)" }}>
                  <code style={{ ...mono, fontSize: 12, color: "#67e8f9" }}>observe</code> is the fast write path — stores raw text with no LLM call.{" "}
                  <code style={{ ...mono, fontSize: 12, color: "#67e8f9" }}>remember</code> is the slow path — runs extraction, entity resolution, and embedding.
                  If uncertain, observe it.
                </p>
              </div>
            </ScrollReveal>
          </div>

          {/* Consolidation pipeline */}
          <ScrollReveal delay={160}>
            <h3 style={{ ...body, fontSize: 17, fontWeight: 600, marginBottom: 8 }}>Consolidation Pipeline</h3>
            <p style={{ ...body, fontSize: 13, lineHeight: 1.7, color: "var(--text-secondary)", marginBottom: 20 }}>
              Fifteen phases on a three-tier schedule. Manual triggers bypass tiering and run all phases.
            </p>

            <div style={{ borderRadius: 14, border: "1px solid var(--border)", overflow: "hidden" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ background: "var(--surface)" }}>
                    {["Phase", "Tier", "Interval", "Description"].map((h, i) => (
                      <th
                        key={h}
                        style={{
                          ...mono,
                          fontSize: 10,
                          fontWeight: 500,
                          letterSpacing: "0.1em",
                          textTransform: "uppercase",
                          textAlign: "left",
                          padding: "12px 16px",
                          color: "var(--text-muted)",
                          borderBottom: "1px solid var(--border)",
                        }}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {PHASES.map((p, i) => (
                    <tr key={p.name} style={{ transition: "background 0.15s" }} onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.02)")} onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>
                      <td style={{ ...mono, fontSize: 13, fontWeight: 500, color: "#67e8f9", padding: "10px 16px", borderBottom: i < PHASES.length - 1 ? "1px solid var(--border)" : "none" }}>
                        {p.name}
                      </td>
                      <td style={{ padding: "10px 16px", borderBottom: i < PHASES.length - 1 ? "1px solid var(--border)" : "none" }}>
                        <TierBadge tier={p.tier} />
                      </td>
                      <td style={{ ...mono, fontSize: 12, color: "var(--text-muted)", padding: "10px 16px", borderBottom: i < PHASES.length - 1 ? "1px solid var(--border)" : "none" }}>
                        {p.interval}
                      </td>
                      <td style={{ ...body, fontSize: 13, color: "var(--text-secondary)", padding: "10px 16px", borderBottom: i < PHASES.length - 1 ? "1px solid var(--border)" : "none" }}>
                        {p.desc}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* ── KEY CONCEPTS ──────────────────────────────────────── */}
      <section style={{ paddingTop: 64, paddingBottom: 96 }}>
        <div style={{ maxWidth: 800, margin: "0 auto", padding: "0 24px" }}>
          <ScrollReveal>
            <Label>KEY CONCEPTS</Label>
            <Heading>The building blocks of memory</Heading>
          </ScrollReveal>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
            {CONCEPT_CARDS.map((card, i) => (
              <ScrollReveal key={card.title} delay={i * 60}>
                <div
                  style={{
                    padding: 24,
                    borderRadius: 14,
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    height: "100%",
                    transition: "border-color 0.2s",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)")}
                  onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
                >
                  {/* Accent stripe */}
                  <div style={{ width: 32, height: 2, borderRadius: 1, background: "linear-gradient(90deg, #67e8f9, #0e7490)", marginBottom: 16 }} />
                  <h3 style={{ ...body, fontSize: 15, fontWeight: 600, marginBottom: 8 }}>{card.title}</h3>
                  <p style={{ ...body, fontSize: 13, lineHeight: 1.7, color: "var(--text-secondary)", margin: 0 }}>{card.desc}</p>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* ── CONFIGURATION ─────────────────────────────────────── */}
      <section style={{ paddingTop: 64, paddingBottom: 96 }}>
        <div style={{ maxWidth: 800, margin: "0 auto", padding: "0 24px" }}>
          <ScrollReveal>
            <Label>CONFIGURATION</Label>
            <Heading>Tune memory behavior</Heading>
            <p style={{ ...body, fontSize: 14, lineHeight: 1.7, color: "var(--text-secondary)", marginBottom: 32, maxWidth: 580 }}>
              All settings live in <code style={{ ...mono, fontSize: 12, color: "#67e8f9" }}>server/engram/config.py</code> and can be overridden with environment variables.
            </p>
          </ScrollReveal>

          <div style={{ display: "grid", gap: 10 }}>
            {CONFIG_FIELDS.map((cfg, i) => (
              <ScrollReveal key={cfg.field} delay={i * 50}>
                <div
                  style={{
                    padding: "18px 24px",
                    borderRadius: 12,
                    background: "rgba(255,255,255,0.02)",
                    border: "1px solid var(--border)",
                  }}
                >
                  <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: 12, marginBottom: 6 }}>
                    <code style={{ ...mono, fontSize: 13, fontWeight: 500, color: "#67e8f9" }}>{cfg.field}</code>
                    <span style={{ ...mono, fontSize: 11, color: "var(--text-muted)" }}>{cfg.values}</span>
                  </div>
                  <p style={{ ...body, fontSize: 13, lineHeight: 1.6, color: "var(--text-secondary)", margin: 0 }}>{cfg.desc}</p>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* ── RESOURCES / CTA ───────────────────────────────────── */}
      <section style={{ paddingTop: 64, paddingBottom: 120, position: "relative" }}>
        <div
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            background: "radial-gradient(ellipse 60% 50% at 50% 70%, rgba(103,232,249,0.04) 0%, transparent 70%)",
          }}
        />

        <div style={{ maxWidth: 800, margin: "0 auto", padding: "0 24px", position: "relative" }}>
          <ScrollReveal>
            <Label>RESOURCES</Label>
            <Heading>Go deeper</Heading>
          </ScrollReveal>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 14, marginBottom: 48 }}>
            {[
              { href: "https://github.com/Moshik21/engram", label: "GitHub Repository", sub: "Source code, issues, contributions", icon: "M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22", ext: true },
              { href: "http://localhost:8100/docs", label: "API Reference", sub: "Interactive OpenAPI docs", icon: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z", ext: true },
              { href: "http://localhost:3000", label: "Dashboard", sub: "Real-time graph explorer", icon: "M3 3h18v18H3zM3 9h18M9 21V9", ext: true },
              { href: "/science", label: "Science & Architecture", sub: "Cognitive science foundations", icon: "M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2zM22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z", ext: false },
            ].map((item, i) => {
              const inner = (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 16,
                    padding: "18px 20px",
                    borderRadius: 14,
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    transition: "border-color 0.2s",
                    height: "100%",
                    cursor: "pointer",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.borderColor = "rgba(103,232,249,0.25)")}
                  onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
                >
                  <div style={{ width: 40, height: 40, borderRadius: 10, background: "rgba(103,232,249,0.08)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#67e8f9" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d={item.icon} />
                    </svg>
                  </div>
                  <div>
                    <span style={{ ...body, fontSize: 14, fontWeight: 500, display: "block" }}>{item.label}</span>
                    <span style={{ ...body, fontSize: 12, color: "var(--text-muted)" }}>{item.sub}</span>
                  </div>
                </div>
              );

              return (
                <ScrollReveal key={item.label} delay={i * 60}>
                  {item.ext ? (
                    <a href={item.href} target="_blank" rel="noopener noreferrer" style={{ textDecoration: "none", color: "inherit", display: "block", height: "100%" }}>
                      {inner}
                    </a>
                  ) : (
                    <Link to={item.href} style={{ textDecoration: "none", color: "inherit", display: "block", height: "100%" }}>
                      {inner}
                    </Link>
                  )}
                </ScrollReveal>
              );
            })}
          </div>

          {/* Final CTA */}
          <ScrollReveal delay={300}>
            <div
              style={{
                textAlign: "center",
                padding: "48px 32px",
                borderRadius: 16,
                background: "var(--surface)",
                border: "1px solid rgba(103,232,249,0.15)",
              }}
            >
              <p style={{ ...serif, fontSize: "clamp(1.25rem, 2.5vw, 1.75rem)", color: "#67e8f9", marginBottom: 24 }}>
                Ready to give your agent a brain?
              </p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 12, justifyContent: "center" }}>
                <a
                  href="https://github.com/Moshik21/engram"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-primary"
                >
                  Clone the repo
                </a>
                <Link to="/roadmap" className="btn btn-secondary">
                  View Roadmap
                </Link>
              </div>
            </div>
          </ScrollReveal>
        </div>
      </section>
    </main>
  );
}
