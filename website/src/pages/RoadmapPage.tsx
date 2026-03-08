import { Link } from "react-router-dom";
import { ScrollReveal } from "../components/ScrollReveal";

/* ───────────────────────────── constants ───────────────────────────── */

const PHASES = [
  {
    label: "Phase 1",
    title: "Foundation",
    status: "current" as const,
    items: [
      "Private local brain per person",
      "Episodic capture + cue-based recall",
      "ACT-R activation-aware retrieval",
      "12-phase offline consolidation",
      "MCP server integration",
      "SQLite lite mode + FalkorDB full mode",
      "Memory maturation (episodic \u2192 transitional \u2192 semantic)",
      "Schema formation (recurring structural patterns)",
      "Dream associations (cross-domain creative connections)",
      "Reconsolidation (update memories when recalled)",
      "Progressive projection (cue-first, extract on demand)",
    ],
  },
  {
    label: "Phase 2",
    title: "Federation",
    status: "next" as const,
    items: [
      "Telemetry capsules (privacy-bounded memory policy stats)",
      "Constitutional memory packs (signed, inspectable, rollbackable)",
      "Archetype priors (engineering, research, personal assistant)",
      "Challenge packs for counterfactual local evaluation",
      "Coordinated learning about memory policy, NOT about private memory content",
    ],
  },
] as const;

const WRONG_MODEL_STEPS = [
  "Centralize data",
  "Train global model",
  "Push down",
] as const;

const BETTER_MODEL_STEPS = [
  "Sovereign local brains",
  "Privacy-bounded telemetry",
  "Signed policy packs",
  "Local shadow evaluation",
] as const;

const KEY_CONCEPTS = [
  {
    title: "Telemetry Capsules",
    description:
      "Sufficient statistics, not memory content. Capsules carry aggregate patterns \u2014 how often memories consolidate, what decay curves look like, which retrieval strategies succeed \u2014 without exposing any individual memory.",
  },
  {
    title: "Constitutional Memory Packs",
    description:
      "Inspectable, rollbackable policy declarations. Each pack is a signed bundle of consolidation rules, retrieval weights, and scheduling parameters that a brain can adopt, audit, or reject.",
  },
  {
    title: "Archetype Priors",
    description:
      "Different defaults for different brain types. An engineering brain prioritizes code patterns and architecture decisions. A research brain weights citation chains and evolving arguments. A personal brain foregrounds relationships and goals.",
  },
  {
    title: "Challenge Packs",
    description:
      "Local evaluation batteries for candidate policies. Before adopting a new memory pack, the brain runs counterfactual tests against its own history \u2014 measuring whether the proposed policy would have improved recall quality, reduced noise, or caught missed connections.",
  },
] as const;

/* ────────────────────────────── colors ─────────────────────────────── */

const C = {
  void: "#030408",
  textPrimary: "#e4e4ed",
  textSecondary: "#7a7a94",
  textMuted: "#44445c",
  accent: "#67e8f9",
  warm: "#f97316",
  purple: "#a78bfa",
  green: "#34d399",
  border: "rgba(255,255,255,0.04)",
} as const;

/* ────────────────────────────── helpers ────────────────────────────── */

const serif = {
  fontFamily: '"Instrument Serif", Georgia, serif',
  fontStyle: "italic" as const,
};
const body = { fontFamily: '"Outfit", sans-serif' };
const mono = { fontFamily: '"JetBrains Mono", monospace' };

function TimelineDot({ status }: { status: "current" | "next" | "future" }) {
  if (status === "current") {
    return (
      <div
        style={{ position: "relative", display: "flex", alignItems: "center", justifyContent: "center", width: 14, height: 14 }}
      >
        <span
          style={{
            position: "absolute",
            width: 24,
            height: 24,
            borderRadius: "50%",
            backgroundColor: C.accent,
            opacity: 0.25,
            animation: "ping 1s cubic-bezier(0, 0, 0.2, 1) infinite",
          }}
        />
        <span
          style={{
            position: "relative",
            width: 14,
            height: 14,
            borderRadius: "50%",
            backgroundColor: C.accent,
            boxShadow: `0 0 0 2px rgba(103,232,249,0.3)`,
          }}
        />
      </div>
    );
  }
  if (status === "next") {
    return (
      <span
        style={{
          width: 14,
          height: 14,
          borderRadius: "50%",
          border: `2px solid ${C.accent}`,
          backgroundColor: "transparent",
          display: "block",
        }}
      />
    );
  }
  return (
    <span
      style={{
        width: 14,
        height: 14,
        borderRadius: "50%",
        border: `2px solid ${C.textMuted}`,
        backgroundColor: "transparent",
        display: "block",
      }}
    />
  );
}

function StatusBadge({ status }: { status: "current" | "next" | "future" }) {
  const map = {
    current: {
      label: "In Progress",
      bg: "rgba(103,232,249,0.15)",
      color: C.accent,
      borderColor: "rgba(103,232,249,0.25)",
    },
    next: {
      label: "Up Next",
      bg: "rgba(249,115,22,0.10)",
      color: C.warm,
      borderColor: "rgba(249,115,22,0.20)",
    },
    future: {
      label: "Planned",
      bg: "rgba(68,68,92,0.10)",
      color: C.textSecondary,
      borderColor: "rgba(68,68,92,0.20)",
    },
  } as const;
  const { label, bg, color, borderColor } = map[status];
  return (
    <span
      style={{
        ...mono,
        display: "inline-block",
        fontSize: 10,
        letterSpacing: "0.15em",
        textTransform: "uppercase",
        padding: "2px 10px",
        borderRadius: 9999,
        border: `1px solid ${borderColor}`,
        backgroundColor: bg,
        color,
      }}
    >
      {label}
    </span>
  );
}

function ArrowRight() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ flexShrink: 0 }}
    >
      <line x1="4" y1="10" x2="16" y2="10" />
      <polyline points="11 5 16 10 11 15" />
    </svg>
  );
}

function XIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      stroke={C.warm}
      strokeWidth="1.5"
      strokeLinecap="round"
      style={{ flexShrink: 0, marginTop: 2 }}
    >
      <line x1="4" y1="4" x2="12" y2="12" />
      <line x1="12" y1="4" x2="4" y2="12" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      stroke={C.accent}
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ flexShrink: 0, marginTop: 2 }}
    >
      <polyline points="3 8.5 6.5 12 13 4" />
    </svg>
  );
}

/* ── shared card style ─────────────────────────────────────────────── */

const cardBase: React.CSSProperties = {
  padding: 24,
  borderRadius: 14,
  backgroundColor: "rgba(255,255,255,0.02)",
  border: `1px solid ${C.border}`,
};

/* ═══════════════════════════════════════════════════════════════════════
   ROADMAP PAGE
   ═══════════════════════════════════════════════════════════════════════ */

export function RoadmapPage() {
  return (
    <main
      style={{ position: "relative", overflowX: "hidden", backgroundColor: C.void, color: C.textPrimary }}
    >
      {/* ── HERO ────────────────────────────────────────────────────── */}
      <section
        style={{ position: "relative", padding: "160px 24px 96px" }}
      >
        {/* Subtle accent glow */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            background:
              "radial-gradient(ellipse 50% 40% at 50% 20%, rgba(103,232,249,0.05) 0%, transparent 70%)",
          }}
        />

        <div
          style={{ position: "relative", maxWidth: 720, marginInline: "auto", textAlign: "center" }}
        >
          <ScrollReveal>
            <span
              style={{
                ...mono,
                fontSize: 12,
                letterSpacing: "0.2em",
                textTransform: "uppercase",
                marginBottom: 24,
                display: "inline-block",
                color: C.textMuted,
              }}
            >
              Roadmap
            </span>
          </ScrollReveal>

          <ScrollReveal delay={100}>
            <h1
              style={{
                ...serif,
                fontSize: "clamp(2.5rem,5.5vw,4rem)",
                lineHeight: 1.12,
                marginBottom: 24,
              }}
            >
              Where Engram Goes Next
            </h1>
          </ScrollReveal>

          <ScrollReveal delay={200}>
            <p
              style={{
                ...body,
                fontSize: 20,
                lineHeight: 1.65,
                color: C.textSecondary,
                maxWidth: 540,
                margin: "0 auto",
              }}
            >
              From local memory to federated policy intelligence.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* ── VISUAL ROADMAP TIMELINE ─────────────────────────────────── */}
      <section style={{ position: "relative", padding: "80px 24px" }}>
        <div style={{ maxWidth: 820, marginInline: "auto" }}>
          <ScrollReveal>
            <h2
              style={{
                ...serif,
                textAlign: "center",
                fontSize: "clamp(1.75rem,3.5vw,2.5rem)",
                lineHeight: 1.2,
                marginBottom: 64,
              }}
            >
              Two phases, one trajectory.
            </h2>
          </ScrollReveal>

          {/* Timeline */}
          <div style={{ position: "relative" }}>
            {/* Vertical line */}
            <div
              style={{
                position: "absolute",
                left: 18,
                top: 0,
                bottom: 0,
                width: 2,
                background: `linear-gradient(to bottom, ${C.accent}, ${C.purple} 60%, transparent)`,
              }}
            />

            <div style={{ display: "flex", flexDirection: "column", gap: 64 }}>
              {PHASES.map((phase, phaseIdx) => (
                <ScrollReveal key={phase.label} delay={phaseIdx * 150}>
                  <div style={{ position: "relative", display: "flex", gap: 32 }}>
                    {/* Dot */}
                    <div
                      style={{
                        position: "relative",
                        display: "flex",
                        alignItems: "flex-start",
                        flexShrink: 0,
                        zIndex: 10,
                        paddingTop: 4,
                        width: 14,
                        marginLeft: 12,
                      }}
                    >
                      <TimelineDot status={phase.status} />
                    </div>

                    {/* Card */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div
                        style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 12, marginBottom: 12 }}
                      >
                        <span
                          style={{
                            ...mono,
                            fontSize: 12,
                            color: C.textMuted,
                            letterSpacing: "0.15em",
                            textTransform: "uppercase",
                          }}
                        >
                          {phase.label}
                        </span>
                        <StatusBadge status={phase.status} />
                      </div>

                      <h3
                        style={{
                          ...serif,
                          fontSize: "clamp(1.5rem,3vw,1.875rem)",
                          marginBottom: 16,
                          color: C.textPrimary,
                        }}
                      >
                        {phase.title}
                      </h3>

                      <div
                        style={{
                          ...cardBase,
                          borderColor:
                            phase.status === "current"
                              ? "rgba(103,232,249,0.18)"
                              : C.border,
                        }}
                      >
                        <ul
                          style={{ display: "flex", flexDirection: "column", gap: 12, listStyle: "none", margin: 0, padding: 0 }}
                        >
                          {phase.items.map((item) => (
                            <li
                              key={item}
                              style={{
                                ...body,
                                display: "flex",
                                alignItems: "flex-start",
                                gap: 12,
                                fontSize: 14,
                                lineHeight: 1.65,
                                color: C.textSecondary,
                              }}
                            >
                              <span
                                style={{
                                  flexShrink: 0,
                                  marginTop: 6,
                                  width: 6,
                                  height: 6,
                                  borderRadius: "50%",
                                  backgroundColor:
                                    phase.status === "current"
                                      ? C.accent
                                      : phase.status === "next"
                                        ? C.warm
                                        : C.textMuted,
                                }}
                              />
                              {item}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                </ScrollReveal>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── THE WRONG MODEL vs THE BETTER MODEL ────────────────────── */}
      <section style={{ position: "relative", padding: "112px 24px" }}>
        <div
          style={{ maxWidth: 720, marginInline: "auto", marginBottom: 56 }}
        >
          <ScrollReveal>
            <h2
              style={{
                ...serif,
                textAlign: "center",
                fontSize: "clamp(1.75rem,3.5vw,2.5rem)",
                lineHeight: 1.2,
                marginBottom: 24,
              }}
            >
              Two models for learning from many brains.
            </h2>
          </ScrollReveal>
          <ScrollReveal delay={100}>
            <p
              style={{
                ...body,
                textAlign: "center",
                color: C.textSecondary,
                fontSize: 18,
                lineHeight: 1.65,
              }}
            >
              The conventional path centralizes private data. The better path
              keeps data sovereign and shares only policy.
            </p>
          </ScrollReveal>
        </div>

        <div
          style={{ maxWidth: 1000, marginInline: "auto", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 24 }}
        >
          {/* Wrong Model */}
          <ScrollReveal delay={100}>
            <div
              style={{
                ...cardBase,
                height: "100%",
                borderColor: "rgba(249,115,22,0.20)",
                display: "flex",
                flexDirection: "column",
              }}
            >
              <div
                style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 24 }}
              >
                <XIcon />
                <h3
                  style={{
                    ...body,
                    fontSize: 18,
                    fontWeight: 500,
                    color: C.warm,
                    margin: 0,
                  }}
                >
                  The Wrong Model
                </h3>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {WRONG_MODEL_STEPS.map((step, i) => (
                  <div
                    key={step}
                    style={{ display: "flex", alignItems: "center", gap: 12 }}
                  >
                    <span
                      style={{
                        ...mono,
                        flexShrink: 0,
                        fontSize: 10,
                        color: C.textMuted,
                        width: 20,
                        textAlign: "right",
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {i + 1}.
                    </span>
                    <span
                      style={{
                        ...body,
                        fontSize: 14,
                        color: C.textSecondary,
                      }}
                    >
                      {step}
                    </span>
                    {i < WRONG_MODEL_STEPS.length - 1 && (
                      <span style={{ color: C.textMuted, marginLeft: "auto" }}>
                        <ArrowRight />
                      </span>
                    )}
                  </div>
                ))}
              </div>

              <p
                style={{
                  ...body,
                  marginTop: 24,
                  paddingTop: 24,
                  borderTop: "1px solid rgba(255,255,255,0.05)",
                  fontSize: 12,
                  color: C.textMuted,
                  lineHeight: 1.65,
                }}
              >
                Data leaves the device. Privacy is an afterthought. Users
                cannot inspect or reject what the model learned from their
                memories.
              </p>
            </div>
          </ScrollReveal>

          {/* Better Model */}
          <ScrollReveal delay={200}>
            <div
              style={{
                ...cardBase,
                height: "100%",
                borderColor: "rgba(103,232,249,0.20)",
                display: "flex",
                flexDirection: "column",
              }}
            >
              <div
                style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 24 }}
              >
                <CheckIcon />
                <h3
                  style={{
                    ...body,
                    fontSize: 18,
                    fontWeight: 500,
                    color: C.accent,
                    margin: 0,
                  }}
                >
                  The Better Model
                </h3>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {BETTER_MODEL_STEPS.map((step, i) => (
                  <div
                    key={step}
                    style={{ display: "flex", alignItems: "center", gap: 12 }}
                  >
                    <span
                      style={{
                        ...mono,
                        flexShrink: 0,
                        fontSize: 10,
                        color: C.textMuted,
                        width: 20,
                        textAlign: "right",
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {i + 1}.
                    </span>
                    <span
                      style={{
                        ...body,
                        fontSize: 14,
                        color: C.textSecondary,
                      }}
                    >
                      {step}
                    </span>
                    {i < BETTER_MODEL_STEPS.length - 1 && (
                      <span style={{ color: C.textMuted, marginLeft: "auto" }}>
                        <ArrowRight />
                      </span>
                    )}
                  </div>
                ))}
              </div>

              <p
                style={{
                  ...body,
                  marginTop: 24,
                  paddingTop: 24,
                  borderTop: "1px solid rgba(255,255,255,0.05)",
                  fontSize: 12,
                  color: C.textMuted,
                  lineHeight: 1.65,
                }}
              >
                Data never leaves the device. Policies are signed, inspectable,
                and rollbackable. Every brain evaluates candidate policies
                locally before adoption.
              </p>
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* ── KEY CONCEPTS ────────────────────────────────────────────── */}
      <section style={{ position: "relative", padding: "112px 24px" }}>
        <div
          style={{ maxWidth: 720, marginInline: "auto", marginBottom: 64 }}
        >
          <ScrollReveal>
            <h2
              style={{
                ...serif,
                textAlign: "center",
                fontSize: "clamp(1.75rem,3.5vw,2.5rem)",
                lineHeight: 1.2,
              }}
            >
              Key concepts behind federated policy intelligence.
            </h2>
          </ScrollReveal>
        </div>

        <div
          style={{ maxWidth: 1000, marginInline: "auto", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 24 }}
        >
          {KEY_CONCEPTS.map((concept, i) => (
            <ScrollReveal key={concept.title} delay={i * 100}>
              <div
                style={{
                  ...cardBase,
                  display: "flex",
                  flexDirection: "column",
                  height: "100%",
                }}
              >
                {/* Accent stripe at top */}
                <div
                  style={{
                    width: 32,
                    height: 3,
                    borderRadius: 2,
                    backgroundColor: C.accent,
                    marginBottom: 16,
                  }}
                />
                <h3
                  style={{
                    ...body,
                    fontSize: 18,
                    fontWeight: 500,
                    marginBottom: 12,
                    color: C.textPrimary,
                  }}
                >
                  {concept.title}
                </h3>
                <p
                  style={{
                    ...body,
                    fontSize: 14,
                    lineHeight: 1.65,
                    color: C.textSecondary,
                    flex: 1,
                    margin: 0,
                  }}
                >
                  {concept.description}
                </p>
              </div>
            </ScrollReveal>
          ))}
        </div>
      </section>

      {/* ── PULL QUOTE ──────────────────────────────────────────────── */}
      <section style={{ position: "relative", padding: "80px 24px 112px" }}>
        <div style={{ maxWidth: 720, marginInline: "auto" }}>
          <ScrollReveal>
            <div
              style={{
                textAlign: "center",
                padding: "32px 48px",
                borderRadius: 14,
                backgroundColor: "rgba(255,255,255,0.02)",
                border: `1px solid rgba(103,232,249,0.12)`,
                boxShadow:
                  "0 0 60px rgba(103,232,249,0.06), 0 0 120px rgba(103,232,249,0.03)",
              }}
            >
              <p
                style={{
                  ...serif,
                  fontSize: "clamp(1.25rem,3vw,1.875rem)",
                  lineHeight: 1.4,
                  color: C.accent,
                  margin: 0,
                }}
              >
                &ldquo;Learn from many private brains without reading their
                thoughts.&rdquo;
              </p>
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* ── CTA ─────────────────────────────────────────────────────── */}
      <section style={{ position: "relative", padding: "112px 24px 128px" }}>
        {/* Accent glow */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            background:
              "radial-gradient(ellipse 60% 50% at 50% 60%, rgba(103,232,249,0.06) 0%, transparent 70%)",
          }}
        />

        <div
          style={{ position: "relative", maxWidth: 720, marginInline: "auto", textAlign: "center" }}
        >
          <ScrollReveal>
            <h2
              style={{
                ...serif,
                fontSize: "clamp(2rem,4.5vw,3.25rem)",
                lineHeight: 1.15,
                marginBottom: 24,
              }}
            >
              Build on the foundation today.
            </h2>
          </ScrollReveal>

          <ScrollReveal delay={100}>
            <p
              style={{
                ...body,
                color: C.textSecondary,
                fontSize: 18,
                lineHeight: 1.65,
                marginBottom: 40,
                maxWidth: 540,
                margin: "0 auto 40px",
              }}
            >
              Phase 1 is live and open source. Explore the cognitive science
              behind the architecture or start building with the documentation.
            </p>
          </ScrollReveal>

          <ScrollReveal delay={200}>
            <div
              style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 16 }}
            >
              <Link
                to="/science"
                style={{
                  ...body,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  padding: "12px 28px",
                  borderRadius: 10,
                  fontSize: 14,
                  fontWeight: 500,
                  backgroundColor: C.accent,
                  color: C.void,
                  textDecoration: "none",
                  transition: "opacity 0.2s",
                }}
              >
                Read The Science
              </Link>
              <Link
                to="/docs"
                style={{
                  ...body,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  padding: "12px 28px",
                  borderRadius: 10,
                  fontSize: 14,
                  fontWeight: 500,
                  backgroundColor: "transparent",
                  color: C.textPrimary,
                  textDecoration: "none",
                  border: `1px solid rgba(255,255,255,0.10)`,
                  transition: "border-color 0.2s",
                }}
              >
                Get Started
              </Link>
            </div>
          </ScrollReveal>
        </div>
      </section>
    </main>
  );
}
