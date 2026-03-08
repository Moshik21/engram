import { Link } from "react-router-dom";
import { ScrollReveal } from "../components/ScrollReveal";

/* ---------------------------------------------------------------------------
   Font & color constants
   --------------------------------------------------------------------------- */

const serif = { fontFamily: '"Instrument Serif", Georgia, serif', fontStyle: "italic" as const };
const mono = { fontFamily: '"JetBrains Mono", monospace' };
const body = { fontFamily: '"Outfit", sans-serif' };

const colors = {
  void: "#030408",
  textPrimary: "#e4e4ed",
  textSecondary: "#7a7a94",
  textMuted: "#44445c",
  accent: "#67e8f9",
  accentDim: "rgba(103, 232, 249, 0.25)",
  accentMuted: "rgba(103, 232, 249, 0.06)",
  warm: "#f97316",
  surface: "rgba(8,10,18,0.82)",
  border: "rgba(255,255,255,0.04)",
  borderHover: "rgba(255,255,255,0.08)",
};

/* ---------------------------------------------------------------------------
   Section number label
   --------------------------------------------------------------------------- */

function SectionLabel({ number, title }: { number: string; title: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
      <span
        style={{
          ...mono,
          fontSize: "0.6875rem",
          fontWeight: 500,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: colors.textMuted,
        }}
      >
        {number}
      </span>
      <span
        style={{
          height: 1,
          width: 32,
          flexShrink: 0,
          background: colors.borderHover,
        }}
      />
      <span
        style={{
          ...mono,
          fontSize: "0.6875rem",
          fontWeight: 500,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: colors.textMuted,
        }}
      >
        {title}
      </span>
    </div>
  );
}

/* ---------------------------------------------------------------------------
   Pull quote with accent left border
   --------------------------------------------------------------------------- */

function PullQuote({ children }: { children: React.ReactNode }) {
  return (
    <blockquote
      style={{
        marginTop: 40,
        marginBottom: 40,
        paddingLeft: 24,
        paddingTop: 4,
        paddingBottom: 4,
        borderLeft: `2px solid ${colors.accent}`,
        ...serif,
        fontSize: "clamp(1.125rem, 2vw, 1.375rem)",
        lineHeight: 1.5,
        color: colors.textPrimary,
      }}
    >
      {children}
    </blockquote>
  );
}

/* ---------------------------------------------------------------------------
   Body paragraph
   --------------------------------------------------------------------------- */

function P({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <p
      style={{
        ...body,
        marginBottom: 24,
        lineHeight: 1.75,
        color: colors.textSecondary,
        fontSize: "clamp(1rem, 1.15vw, 1.0625rem)",
      }}
    >
      {children}
    </p>
  );
}

/* ---------------------------------------------------------------------------
   Section heading (serif italic)
   --------------------------------------------------------------------------- */

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2
      style={{
        ...serif,
        marginBottom: 24,
        fontWeight: 400,
        fontSize: "clamp(1.5rem, 3.5vw, 2.25rem)",
        lineHeight: 1.2,
        letterSpacing: "-0.01em",
        color: colors.textPrimary,
      }}
    >
      {children}
    </h2>
  );
}

/* ---------------------------------------------------------------------------
   Comparison card
   --------------------------------------------------------------------------- */

function ComparisonCard({
  title,
  accent,
  items,
}: {
  title: string;
  accent: boolean;
  items: string[];
}) {
  return (
    <div
      style={{
        flex: 1,
        minWidth: 260,
        borderRadius: 14,
        padding: 24,
        background: accent ? colors.accentMuted : colors.surface,
        border: accent
          ? "1px solid rgba(103, 232, 249, 0.15)"
          : `1px solid ${colors.border}`,
      }}
    >
      <h3
        style={{
          ...mono,
          marginBottom: 16,
          fontSize: "0.75rem",
          fontWeight: 500,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: accent ? colors.accent : colors.textMuted,
        }}
      >
        {title}
      </h3>
      <ul style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {items.map((item) => (
          <li
            key={item}
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 8,
              ...body,
              fontSize: "0.9375rem",
              lineHeight: 1.625,
              color: colors.textSecondary,
            }}
          >
            <span
              style={{
                marginTop: 7,
                display: "block",
                height: 6,
                width: 6,
                flexShrink: 0,
                borderRadius: "50%",
                background: accent ? colors.accent : colors.textMuted,
              }}
            />
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ---------------------------------------------------------------------------
   Pipeline step
   --------------------------------------------------------------------------- */

function PipelineStep({
  label,
  description,
  last = false,
}: {
  label: string;
  description: string;
  last?: boolean;
}) {
  return (
    <div style={{ display: "flex", gap: 16 }}>
      {/* Connector */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
        <div
          style={{
            height: 10,
            width: 10,
            borderRadius: "50%",
            flexShrink: 0,
            marginTop: 6,
            background: colors.accent,
            boxShadow: `0 0 8px ${colors.accentDim}`,
          }}
        />
        {!last && (
          <div
            style={{
              width: 1,
              flex: 1,
              marginTop: 4,
              marginBottom: 4,
              background: `linear-gradient(to bottom, ${colors.accentDim}, ${colors.border})`,
            }}
          />
        )}
      </div>
      {/* Content */}
      <div style={{ paddingBottom: 24 }}>
        <span
          style={{
            ...mono,
            fontSize: "0.8125rem",
            fontWeight: 500,
            color: colors.accent,
          }}
        >
          {label}
        </span>
        <p
          style={{
            ...body,
            marginTop: 4,
            fontSize: "0.9375rem",
            lineHeight: 1.625,
            color: colors.textSecondary,
          }}
        >
          {description}
        </p>
      </div>
    </div>
  );
}

/* ===========================================================================
   VisionPage
   =========================================================================== */

export function VisionPage() {
  return (
    <main
      style={{ position: "relative", minHeight: "100vh", background: colors.void }}
    >
      {/* ------------------------------------------------------------------ */}
      {/* Hero                                                                */}
      {/* ------------------------------------------------------------------ */}
      <section
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "70vh",
          paddingTop: "clamp(6rem, 14vw, 10rem)",
          paddingBottom: "clamp(4rem, 8vw, 6rem)",
          paddingLeft: 24,
          paddingRight: 24,
        }}
      >
        <div style={{ maxWidth: 680, marginLeft: "auto", marginRight: "auto", textAlign: "center" }}>
          <ScrollReveal>
            <span
              style={{
                display: "inline-block",
                ...mono,
                fontSize: "0.6875rem",
                fontWeight: 500,
                letterSpacing: "0.14em",
                textTransform: "uppercase",
                marginBottom: 24,
                color: colors.accent,
              }}
            >
              Vision
            </span>
          </ScrollReveal>

          <ScrollReveal delay={100}>
            <h1
              style={{
                ...serif,
                fontWeight: 400,
                fontSize: "clamp(2.75rem, 7vw, 4.5rem)",
                lineHeight: 1.05,
                letterSpacing: "-0.025em",
                color: colors.textPrimary,
              }}
            >
              One Brain Per Person
            </h1>
          </ScrollReveal>

          <ScrollReveal delay={200}>
            <p
              style={{
                ...body,
                marginTop: 24,
                fontSize: "clamp(1.0625rem, 1.5vw, 1.25rem)",
                lineHeight: 1.6,
                color: colors.textSecondary,
              }}
            >
              The product philosophy behind Engram.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Essay body — constrained width                                      */}
      {/* ------------------------------------------------------------------ */}
      <div
        style={{
          maxWidth: 680,
          marginLeft: "auto",
          marginRight: "auto",
          paddingLeft: 24,
          paddingRight: 24,
          paddingBottom: 128,
        }}
      >
        {/* -------------------------------------------------------------- */}
        {/* 01 — Why This Matters                                          */}
        {/* -------------------------------------------------------------- */}
        <ScrollReveal style={{ marginBottom: 96 }}>
          <SectionLabel number="01" title="Why This Matters" />
          <SectionHeading>
            Your life is not cleanly partitioned
          </SectionHeading>

          <P>
            Work bleeds into calendar. Old research resurfaces in new
            conversations. A health change rewrites your travel plans. A
            half-remembered article shapes how you approach a design problem
            six months later. The connections that matter most are precisely
            the ones that cross boundaries.
          </P>

          <P>
            Most memory systems ignore this. They silo information by
            project, by workspace, by app. Each bucket is tidy on its own,
            but the topology of your actual thinking is lost. The system
            becomes easier to implement but less useful to live with.
          </P>

          <PullQuote>
            Projects are topology, not tenancy.
          </PullQuote>

          <P>
            In Engram, a project is not a walled database. It is a region of
            a single, continuous knowledge graph — a local neighborhood of
            entities, weighted by topic and recency. When you work inside a
            project, retrieval is biased toward its neighborhood. But the
            edges that connect one neighborhood to another are never severed.
            A fact about your health can still reach a planning conversation
            if the activation signal is strong enough.
          </P>

          <P>
            This is not a convenience feature. It is the core architectural
            bet. One brain per person. Not one brain per project.
          </P>
        </ScrollReveal>

        {/* -------------------------------------------------------------- */}
        {/* 02 — Memory Is Not a Bigger Context Window                     */}
        {/* -------------------------------------------------------------- */}
        <ScrollReveal style={{ marginBottom: 96 }}>
          <SectionLabel number="02" title="The Distinction" />
          <SectionHeading>
            Memory is not a bigger context window
          </SectionHeading>

          <P>
            It is tempting to treat memory as a scaling problem: just make
            the context window larger and stuff more text into it. This
            approach fails for reasons that go beyond cost.
          </P>

          <div
            style={{ display: "flex", flexWrap: "wrap", gap: 16, marginTop: 40, marginBottom: 40 }}
          >
            <ComparisonCard
              title="Context Windows"
              accent={false}
              items={[
                "Expensive to fill and maintain",
                "Flat — no structure or hierarchy",
                "Temporary — gone after the session",
                "Weak at separating signal from noise",
                "Treats all tokens as equally important",
              ]}
            />
            <ComparisonCard
              title="Good Memory"
              accent
              items={[
                "Selective about what to retain",
                "Layered — episodic, transitional, semantic",
                "Durable — persists across months and years",
                "Updateable — corrections overwrite, not append",
                "Shaped by use — recall strengthens traces",
              ]}
            />
          </div>

          <P>
            A context window is a buffer. Memory is a discipline. The
            difference is not in how much you can hold at once, but in how
            intelligently you forget, compress, and resurface.
          </P>
        </ScrollReveal>

        {/* -------------------------------------------------------------- */}
        {/* 03 — How Engram Works in Plain Terms                           */}
        {/* -------------------------------------------------------------- */}
        <ScrollReveal style={{ marginBottom: 96 }}>
          <SectionLabel number="03" title="How It Works" />
          <SectionHeading>Engram in plain terms</SectionHeading>

          <P>
            The system is built around a cycle that mirrors how biological
            memory works — not as metaphor, but as engineering constraint.
            Each stage solves a specific failure mode of naive storage.
          </P>

          <div style={{ marginTop: 40, marginBottom: 40 }}>
            <PipelineStep
              label="Episode Storage"
              description="Conversations are captured as temporal episodes. Fast, cheap, no analysis yet. The raw material."
            />
            <PipelineStep
              label="Cue Creation"
              description="Entities, relationships, and facts are extracted from episodes. The graph grows. Names are resolved, duplicates are merged."
            />
            <PipelineStep
              label="Recall"
              description="When you ask a question, spreading activation flows through the graph. Relevant memories surface based on structure, recency, and frequency — not just keyword match."
            />
            <PipelineStep
              label="Usage Feedback"
              description="Every recall strengthens the traces that were useful. Memory adapts to how you actually think, not to a static index."
            />
            <PipelineStep
              label="Projection"
              description="Entities mature from episodic to semantic. Summaries tighten. The graph becomes denser and more precise over time."
            />
            <PipelineStep
              label="Consolidation"
              description="Offline cycles prune noise, merge duplicates, infer missing links, and discover structural patterns. The system dreams."
              last
            />
          </div>

          <PullQuote>
            The point is not just to remember more. The point is to remember
            better.
          </PullQuote>
        </ScrollReveal>

        {/* -------------------------------------------------------------- */}
        {/* 04 — The Product Bet                                           */}
        {/* -------------------------------------------------------------- */}
        <ScrollReveal style={{ marginBottom: 96 }}>
          <SectionLabel number="04" title="The Product Bet" />
          <SectionHeading>Continuity over cleverness</SectionHeading>

          <P>
            The future of AI assistants will be decided less by who can
            generate a good reply in one shot, and more by who can build
            continuity over weeks, months, and years.
          </P>

          <P>
            A single brilliant response has diminishing returns. What
            compounds is the ability to remember that you tried something
            last March and it did not work, that your team renamed a project
            in August, that you prefer concise answers on technical topics
            and longer explanations on strategic ones. That kind of
            continuity requires a memory system with specific properties:
          </P>

          <ul style={{ marginTop: 32, marginBottom: 32, display: "flex", flexDirection: "column", gap: 16 }}>
            {[
              [
                "Privacy-first.",
                "Memory storage is local. Entity extraction currently uses cloud APIs but centralized memory is a liability, not a feature.",
              ],
              [
                "Adaptive.",
                "The system should get better at remembering what matters to you specifically, without manual curation.",
              ],
              [
                "Cross-domain.",
                "Work, personal, health, interests — one graph, with soft boundaries, not hard walls.",
              ],
              [
                "Evidence-based structure.",
                "Entities and relationships should be grounded in observed conversations, not hallucinated from patterns.",
              ],
              [
                "Noise-resistant.",
                "Remembering everything is worse than remembering nothing. The system must be selective or it becomes a junk drawer.",
              ],
            ].map(([strong, rest]) => (
              <li
                key={strong}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 12,
                  ...body,
                  fontSize: "0.9375rem",
                  lineHeight: 1.625,
                  color: colors.textSecondary,
                }}
              >
                <span
                  style={{
                    marginTop: 7,
                    display: "block",
                    height: 6,
                    width: 6,
                    flexShrink: 0,
                    borderRadius: "50%",
                    background: colors.accent,
                  }}
                />
                <span>
                  <strong style={{ color: colors.textPrimary }}>
                    {strong}
                  </strong>{" "}
                  {rest}
                </span>
              </li>
            ))}
          </ul>

          <P>
            We believe the end state is not one giant model that knows
            everything about everyone. It is many private brains — one per
            person — each shaped by individual experience.
          </P>

          <PullQuote>
            Many private brains should be able to improve the global memory
            discipline without centralizing private memory itself.
          </PullQuote>

          <P>
            The techniques for selective retention, activation-based
            retrieval, and consolidation scheduling are general. They can be
            shared, refined, and improved across instances without any single
            instance exposing its contents. Federated learning for memory,
            not federated storage.
          </P>

          <P>
            That is the long bet. Not a product with a moat around your
            data, but a product that earns trust by never needing to see it
            in the first place.
          </P>
        </ScrollReveal>

        {/* -------------------------------------------------------------- */}
        {/* CTA                                                             */}
        {/* -------------------------------------------------------------- */}
        <ScrollReveal>
          <hr
            style={{
              border: "none",
              height: 1,
              marginBottom: 64,
              background: `linear-gradient(90deg, transparent, ${colors.borderHover} 30%, ${colors.accentDim} 50%, ${colors.borderHover} 70%, transparent)`,
            }}
          />

          <div style={{ textAlign: "center" }}>
            <p
              style={{
                ...serif,
                marginBottom: 32,
                fontSize: "clamp(1.25rem, 2.5vw, 1.625rem)",
                lineHeight: 1.3,
                color: colors.textPrimary,
              }}
            >
              Read more about the architecture, or start building.
            </p>

            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "center", gap: 16 }}>
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
                  minWidth: 160,
                  backgroundColor: "transparent",
                  color: colors.textPrimary,
                  border: `1px solid ${colors.borderHover}`,
                  textDecoration: "none",
                }}
              >
                The Science
              </Link>
              <Link
                to="/roadmap"
                style={{
                  ...body,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  padding: "12px 28px",
                  borderRadius: 10,
                  fontSize: 14,
                  fontWeight: 500,
                  minWidth: 160,
                  backgroundColor: "transparent",
                  color: colors.textPrimary,
                  border: `1px solid ${colors.borderHover}`,
                  textDecoration: "none",
                }}
              >
                Roadmap
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
                  minWidth: 160,
                  backgroundColor: colors.accent,
                  color: colors.void,
                  border: "1px solid transparent",
                  textDecoration: "none",
                }}
              >
                Get Started
              </Link>
            </div>
          </div>
        </ScrollReveal>
      </div>
    </main>
  );
}
