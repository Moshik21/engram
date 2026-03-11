import { Link } from "react-router-dom";
import { ScrollReveal } from "../components/ScrollReveal";

/* -------------------------------------------------------------------------- */
/*  Style objects                                                              */
/* -------------------------------------------------------------------------- */

const serif = { fontFamily: '"Instrument Serif", Georgia, serif', fontStyle: "italic" as const };
const mono = { fontFamily: '"JetBrains Mono", monospace' };
const body = { fontFamily: '"Outfit", sans-serif' };

/* -------------------------------------------------------------------------- */
/*  Helpers                                                                    */
/* -------------------------------------------------------------------------- */

function SectionNumber({ n }: { n: number }) {
  return (
    <span
      style={{
        display: "block",
        marginBottom: 16,
        ...mono,
        fontSize: 11,
        fontWeight: 500,
        letterSpacing: "0.2em",
        textTransform: "uppercase" as const,
        color: "var(--text-muted)",
      }}
    >
      {String(n).padStart(2, "0")}
    </span>
  );
}

function Pullquote({ children }: { children: React.ReactNode }) {
  return (
    <blockquote
      style={{
        ...body,
        borderLeft: "2px solid #67e8f9",
        paddingLeft: 24,
        marginTop: 40,
        marginBottom: 40,
        fontSize: 18,
        lineHeight: 1.7,
        color: "rgba(228,228,237,0.9)",
        fontStyle: "italic",
      }}
    >
      {children}
    </blockquote>
  );
}

function Divider() {
  return (
    <div
      style={{
        width: 64,
        height: 1,
        margin: "0 auto",
        background: "linear-gradient(to right, transparent, rgba(103,232,249,0.3), transparent)",
      }}
    />
  );
}

function BodyText({
  children,
  mb = 24,
}: {
  children: React.ReactNode;
  mb?: number;
}) {
  return (
    <p style={{ ...body, color: "var(--text-secondary)", lineHeight: 1.75, marginBottom: mb }}>
      {children}
    </p>
  );
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2
      style={{
        ...serif,
        fontSize: "clamp(1.75rem, 3.5vw, 2.5rem)",
        lineHeight: 1.2,
        marginBottom: 32,
      }}
    >
      {children}
    </h2>
  );
}

function Strong({ children }: { children: React.ReactNode }) {
  return (
    <strong style={{ color: "var(--text-primary)", fontWeight: 500 }}>
      {children}
    </strong>
  );
}

/* -------------------------------------------------------------------------- */
/*  Consolidation phase cards                                                  */
/* -------------------------------------------------------------------------- */

const consolidationPhases = [
  {
    name: "Triage",
    desc: "Score incoming episodes by novelty and signal density. Promote the valuable fraction; skip the noise.",
  },
  {
    name: "Merge",
    desc: "Detect duplicate entities across sessions using fuzzy name analysis, embeddings, and structural signals.",
  },
  {
    name: "Infer",
    desc: "Discover implicit relationships between entities that co-occur but were never explicitly linked.",
  },
  {
    name: "Replay",
    desc: "Re-extract recent episodes to catch entities and relationships missed on the first pass.",
  },
  {
    name: "Mature",
    desc: "Graduate well-attested entities from episodic to transitional to semantic memory tiers.",
  },
  {
    name: "Semanticize",
    desc: "Promote episodes whose linked entities have matured, shifting them into long-term storage.",
  },
  {
    name: "Prune",
    desc: "Soft-delete entities that are old, rarely accessed, and low-activation. Forgetting as maintenance.",
  },
];

/* -------------------------------------------------------------------------- */
/*  Page                                                                       */
/* -------------------------------------------------------------------------- */

export default function SciencePage() {
  return (
    <main style={{ minHeight: "100vh", background: "var(--void)", color: "var(--text-primary)", overflowX: "hidden" }}>

      {/* ------------------------------------------------------------------ */}
      {/*  Hero                                                               */}
      {/* ------------------------------------------------------------------ */}
      <header
        style={{ position: "relative", padding: "128px 24px 80px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center" }}
      >
        <ScrollReveal>
          <span
            style={{ ...mono, display: "inline-block", marginBottom: 24, fontSize: 12, letterSpacing: "0.3em", textTransform: "uppercase" as const, color: "#67e8f9" }}
          >
            The Science
          </span>
          <h1
            style={{ ...serif, fontSize: "clamp(2.5rem, 5.5vw, 4rem)", lineHeight: 1.1, maxWidth: 768, marginInline: "auto", marginBottom: 24 }}
          >
            The Science Behind Engram
          </h1>
          <p
            style={{ ...body, maxWidth: 540, marginInline: "auto", fontSize: 18, color: "var(--text-secondary)", lineHeight: 1.7 }}
          >
            Inspired by memory science, not a claim of biological equivalence.
          </p>
          <div
            style={{
              marginTop: 48,
              height: 1,
              width: 96,
              marginLeft: "auto",
              marginRight: "auto",
              background: "linear-gradient(to right, transparent, rgba(103,232,249,0.4), transparent)",
            }}
          />
        </ScrollReveal>
      </header>

      {/* ------------------------------------------------------------------ */}
      {/*  Editorial body                                                     */}
      {/* ------------------------------------------------------------------ */}
      <div style={{ maxWidth: 680, margin: "0 auto", padding: "0 24px" }}>

        {/* -------------------------------------------------------------- */}
        {/*  01 — What Is an Engram?                                        */}
        {/* -------------------------------------------------------------- */}
        <section style={{ paddingTop: 80, paddingBottom: 80 }}>
          <ScrollReveal>
            <SectionNumber n={1} />
            <SectionHeading>What Is an Engram?</SectionHeading>
            <BodyText>
              The word <em>engram</em> comes from neuroscience. It refers to the
              physical or biochemical trace that a memory leaves in the brain
              &mdash; not a file in a folder, but a pattern distributed across
              biological tissue. The concept was introduced by Richard Semon in
              1904 and formalized through decades of research into how the brain
              encodes, stores, and retrieves experience.
            </BodyText>
            <BodyText>
              An engram is not static. It can be strengthened through repeated
              retrieval, weakened through disuse, modified when reactivated, and
              sometimes lost entirely. Memory, in biological systems, is an
              active process &mdash; not a passive archive.
            </BodyText>
            <Pullquote>
              Memory is not just stored data. It is a trace that can be
              strengthened, reactivated, updated, and sometimes lost.
            </Pullquote>
            <BodyText mb={0}>
              We named this project Engram because it reflects the design
              philosophy: memory as a living, dynamic structure rather than a
              static database. The principles that govern how biological memory
              works turn out to be remarkably useful for designing AI memory
              systems that behave well over time.
            </BodyText>
          </ScrollReveal>
        </section>

        <Divider />

        {/* -------------------------------------------------------------- */}
        {/*  02 — Episodic Capture Before Durable Knowledge                 */}
        {/* -------------------------------------------------------------- */}
        <section style={{ paddingTop: 80, paddingBottom: 80 }}>
          <ScrollReveal>
            <SectionNumber n={2} />
            <SectionHeading>Episodic Capture Before Durable Knowledge</SectionHeading>
            <BodyText>
              One of the most important discoveries in memory science is that the
              brain uses two complementary systems. The hippocampus captures
              episodes quickly and with high fidelity &mdash; a fast system. The
              neocortex integrates knowledge slowly, finding patterns across many
              episodes over time &mdash; a slow system. This is known as{" "}
              <Strong>Complementary Learning Systems</Strong>{" "}
              theory, proposed by McClelland, McNaughton, and O&apos;Reilly in 1995.
            </BodyText>
            <BodyText mb={32}>
              Engram mirrors this architecture. New information arrives as raw
              episodes &mdash; cheap to capture, requiring no immediate analysis.
              Offline consolidation then extracts entities, detects patterns, and
              builds durable knowledge over time. The fast path and the slow path
              serve different purposes, and both are necessary.
            </BodyText>

            {/* Dual-system diagram */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16, marginTop: 40, marginBottom: 40 }}>
              <div
                style={{
                  borderRadius: 12,
                  border: "1px solid rgba(103,232,249,0.2)",
                  background: "rgba(103,232,249,0.03)",
                  padding: 24,
                }}
              >
                <div
                  style={{ ...mono, fontSize: 12, color: "#67e8f9", letterSpacing: "0.05em", textTransform: "uppercase" as const, marginBottom: 12 }}
                >
                  Fast System
                </div>
                <p style={{ ...body, fontSize: 14, color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: 16 }}>
                  Capture raw episodes immediately. No extraction, no LLM call.
                  Preserve fidelity.
                </p>
                <div style={{ ...mono, display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "rgba(122,122,148,0.6)" }}>
                  <span style={{ height: 6, width: 6, borderRadius: "50%", background: "rgba(103,232,249,0.6)", display: "inline-block" }} />
                  observe &rarr; store
                </div>
              </div>
              <div
                style={{
                  borderRadius: 12,
                  border: "1px solid rgba(255,255,255,0.06)",
                  background: "rgba(255,255,255,0.02)",
                  padding: 24,
                }}
              >
                <div
                  style={{ ...mono, fontSize: 12, color: "var(--text-secondary)", letterSpacing: "0.05em", textTransform: "uppercase" as const, marginBottom: 12 }}
                >
                  Slow System
                </div>
                <p style={{ ...body, fontSize: 14, color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: 16 }}>
                  Offline consolidation extracts entities, merges duplicates,
                  infers relationships.
                </p>
                <div style={{ ...mono, display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "rgba(122,122,148,0.6)" }}>
                  <span style={{ height: 6, width: 6, borderRadius: "50%", background: "rgba(255,255,255,0.3)", display: "inline-block" }} />
                  triage &rarr; project &rarr; consolidate
                </div>
              </div>
            </div>

            <BodyText mb={0}>
              Most AI memory systems skip the fast system entirely &mdash;
              everything is immediately chunked, embedded, and stored. This is
              expensive, slow, and forces premature decisions about what matters.
              The two-system approach lets capture be cheap and analysis be
              deliberate.
            </BodyText>
          </ScrollReveal>
        </section>

        <Divider />

        {/* -------------------------------------------------------------- */}
        {/*  03 — Cue-Dependent Retrieval                                   */}
        {/* -------------------------------------------------------------- */}
        <section style={{ paddingTop: 80, paddingBottom: 80 }}>
          <ScrollReveal>
            <SectionNumber n={3} />
            <SectionHeading>Cue-Dependent Retrieval</SectionHeading>
            <BodyText>
              Human memory is not accessed by querying a database. It is
              triggered by cues &mdash; a word, a context, a feeling. Endel
              Tulving&apos;s{" "}
              <Strong>encoding specificity principle</Strong>{" "}
              (1973) established that the effectiveness of a retrieval cue
              depends on how well it overlaps with the original encoding context.
              Partial overlap is enough. Exact match is not required.
            </BodyText>
            <BodyText>
              Most RAG systems treat retrieval as document search: embed the
              query, find the nearest vectors, return the top-k chunks. This
              works for explicit lookup but fails for the kind of associative,
              context-sensitive recall that makes memory useful in conversation.
            </BodyText>
            <Pullquote>
              This is closer to reminding than to search.
            </Pullquote>
            <BodyText mb={0}>
              Engram extracts cues from incoming content and uses them for
              layered retrieval &mdash; combining text similarity, graph
              structure, spreading activation, and recency signals. A partial
              mention of a person&apos;s name, a reference to a project, or a
              thematic overlap can all serve as retrieval cues, just as they do
              in biological memory.
            </BodyText>
          </ScrollReveal>
        </section>

        <Divider />

        {/* -------------------------------------------------------------- */}
        {/*  04 — Activation                                                */}
        {/* -------------------------------------------------------------- */}
        <section style={{ paddingTop: 80, paddingBottom: 80 }}>
          <ScrollReveal>
            <SectionNumber n={4} />
            <SectionHeading>Activation: Recency, Frequency, Relevance</SectionHeading>
            <BodyText>
              John Anderson&apos;s{" "}
              <Strong>ACT-R cognitive architecture</Strong>{" "}
              models memory retrieval as a competitive process. Each memory trace
              has an activation level that reflects how recently and frequently
              it has been accessed, plus how well it matches the current context.
              The most activated trace wins.
            </BodyText>
            <BodyText mb={32}>
              Engram implements activation as a core retrieval signal. Activation
              is computed lazily from each entity&apos;s access history &mdash;
              never stored as a decaying float. Recent accesses contribute more.
              Repeated accesses build strength. Contextual relevance adds a
              boost. The result is a retrieval system that naturally prioritizes
              what is currently relevant without manual tuning.
            </BodyText>

            {/* Activation decay visual */}
            <div
              style={{
                marginTop: 40,
                marginBottom: 40,
                borderRadius: 12,
                border: "1px solid rgba(255,255,255,0.06)",
                background: "rgba(255,255,255,0.02)",
                padding: 24,
              }}
            >
              <div
                style={{ ...mono, fontSize: 12, color: "var(--text-secondary)", letterSpacing: "0.05em", textTransform: "uppercase" as const, marginBottom: 16 }}
              >
                Activation Over Time
              </div>
              <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 96 }}>
                {[
                  100, 95, 88, 80, 72, 64, 56, 49, 43, 38, 33, 29, 26, 23, 21,
                  19, 17, 16, 15, 14, 13, 12, 12, 11, 11, 10, 10, 10, 9, 9, 9,
                  9, 8, 8, 8, 8, 8, 8, 7, 7,
                ].map((h, i) => (
                  <div
                    key={i}
                    style={{
                      flex: 1,
                      minWidth: 3,
                      borderRadius: 2,
                      height: `${h}%`,
                      backgroundColor: `rgba(103,232,249,${Math.max(h / 100, 0.15)})`,
                    }}
                  />
                ))}
              </div>
              <div
                style={{ ...mono, display: "flex", justifyContent: "space-between", marginTop: 12, fontSize: 10, color: "rgba(122,122,148,0.5)" }}
              >
                <span>now</span>
                <span>time &rarr;</span>
              </div>
              <p style={{ ...body, marginTop: 16, fontSize: 12, color: "rgba(122,122,148,0.7)", lineHeight: 1.7 }}>
                Power-law decay: recent accesses dominate, but frequently
                accessed entities maintain a warm baseline. Semantic-tier
                entities decay more slowly (exponent 0.3 vs 0.5).
              </p>
            </div>

            <BodyText mb={0}>
              This is not a novelty. ACT-R has been validated across thousands of
              studies over four decades. What is novel is applying it as the
              primary retrieval signal in a knowledge graph, where activation
              flows through entity relationships via spreading activation
              &mdash; another well-established cognitive mechanism.
            </BodyText>
          </ScrollReveal>
        </section>

        <Divider />

        {/* -------------------------------------------------------------- */}
        {/*  05 — Consolidation                                             */}
        {/* -------------------------------------------------------------- */}
        <section style={{ paddingTop: 80, paddingBottom: 80 }}>
          <ScrollReveal>
            <SectionNumber n={5} />
            <SectionHeading>Consolidation</SectionHeading>
            <BodyText>
              Biological memory does not become permanent at the moment of
              encoding. It undergoes{" "}
              <Strong>consolidation</Strong>{" "}
              &mdash; a process that transforms fragile, recently formed traces
              into stable, long-term representations. Much of this happens during
              sleep, when the hippocampus replays recent experiences and the
              neocortex gradually integrates them into existing knowledge
              structures.
            </BodyText>
            <BodyText mb={32}>
              Engram runs offline consolidation cycles that mirror this process.
              Episodes that were cheaply captured during active use are later
              analyzed, cross-referenced, and integrated into the knowledge
              graph. The system identifies duplicates, discovers implicit
              relationships, and promotes well-attested knowledge to more durable
              storage tiers.
            </BodyText>

            {/* Phase cards */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 40, marginBottom: 40 }}>
              {consolidationPhases.map((phase, i) => (
                <div
                  key={phase.name}
                  style={{
                    display: "flex",
                    gap: 16,
                    borderRadius: 10,
                    border: "1px solid rgba(255,255,255,0.04)",
                    background: "rgba(255,255,255,0.015)",
                    padding: "16px 20px",
                    transition: "border-color 0.2s, background 0.2s",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "rgba(103,232,249,0.2)";
                    e.currentTarget.style.background = "rgba(103,232,249,0.02)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "rgba(255,255,255,0.04)";
                    e.currentTarget.style.background = "rgba(255,255,255,0.015)";
                  }}
                >
                  <span
                    style={{ ...mono, flexShrink: 0, fontSize: 12, color: "rgba(103,232,249,0.5)", paddingTop: 2 }}
                  >
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <div>
                    <div style={{ ...body, fontWeight: 500, fontSize: 14, color: "var(--text-primary)", marginBottom: 4 }}>
                      {phase.name}
                    </div>
                    <p style={{ ...body, fontSize: 14, color: "rgba(122,122,148,0.8)", lineHeight: 1.7 }}>
                      {phase.desc}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            <BodyText mb={0}>
              Phases are scheduled across three temperature tiers &mdash; hot,
              warm, and cold &mdash; so lightweight maintenance runs frequently
              while expensive operations run only when needed. The system adapts
              to workload pressure, not fixed timers.
            </BodyText>
          </ScrollReveal>
        </section>

        <Divider />

        {/* -------------------------------------------------------------- */}
        {/*  06 — Reconsolidation                                           */}
        {/* -------------------------------------------------------------- */}
        <section style={{ paddingTop: 80, paddingBottom: 80 }}>
          <ScrollReveal>
            <SectionNumber n={6} />
            <SectionHeading>Reconsolidation</SectionHeading>
            <BodyText>
              One of the most striking findings in modern memory research is that
              retrieving a memory does not merely read it &mdash; it temporarily
              destabilizes it. The retrieved trace enters a{" "}
              <Strong>labile state</Strong>{" "}
              where it can be modified by new information before being
              restabilized. This process, called reconsolidation, was
              demonstrated by Nader, Schafe, and LeDoux in 2000 and has since
              been replicated across many memory systems and species.
            </BodyText>
            <Pullquote>
              A good memory system should not treat retrieval as frozen lookup.
            </Pullquote>
            <BodyText>
              Engram implements reconsolidation directly. When an entity is
              retrieved, it enters a short labile window. If new, relevant
              information arrives during that window, the entity&apos;s summary
              can be updated in place &mdash; integrating the correction or
              addition without creating a duplicate.
            </BodyText>
            <BodyText mb={0}>
              This means the system naturally handles corrections, updates, and
              elaborations. If a user says &ldquo;actually, I moved to Austin
              last month&rdquo; shortly after a retrieval about their location,
              the existing entity is updated rather than a contradictory new one
              being created. The window is bounded and does not extend on
              re-retrieval, preventing runaway instability.
            </BodyText>
          </ScrollReveal>
        </section>

        <Divider />

        {/* -------------------------------------------------------------- */}
        {/*  07 — Forgetting Is a Feature                                   */}
        {/* -------------------------------------------------------------- */}
        <section style={{ paddingTop: 80, paddingBottom: 80 }}>
          <ScrollReveal>
            <SectionNumber n={7} />
            <SectionHeading>Forgetting Is a Feature</SectionHeading>
            <BodyText>
              There is a persistent assumption in software engineering that data
              retention is always good. More data, more context, better results.
              Memory science tells a different story.
            </BodyText>
            <BodyText>
              Robert Bjork&apos;s{" "}
              <Strong>New Theory of Disuse</Strong>{" "}
              (1992) distinguishes between storage strength and retrieval
              strength. Information may still be stored but become inaccessible
              because its retrieval strength has decayed. This is not a bug. It
              is an optimization &mdash; the system surfaces what is likely to be
              needed and lets the rest fade.
            </BodyText>
            <Pullquote>
              Indiscriminate retention is not the same as good memory. It creates
              noise, stale facts, contradictions, and retrieval interference.
            </Pullquote>
            <BodyText mb={0}>
              Engram treats forgetting as part of memory health. The pruning
              phase removes entities that are old, rarely accessed, and
              low-activation. Activation naturally decays with time. Entities
              that are never reinforced through retrieval or new mentions
              eventually fall below the threshold and are cleaned up. The result
              is a knowledge graph that stays current and relevant rather than
              accumulating an ever-growing pile of stale information.
            </BodyText>
          </ScrollReveal>
        </section>

        <Divider />

        {/* -------------------------------------------------------------- */}
        {/*  08 — Memory Should Improve Through Use                         */}
        {/* -------------------------------------------------------------- */}
        <section style={{ paddingTop: 80, paddingBottom: 80 }}>
          <ScrollReveal>
            <SectionNumber n={8} />
            <SectionHeading>Memory Should Improve Through Use</SectionHeading>
            <BodyText>
              In biological systems, retrieval is not a passive read. Each
              retrieval event strengthens the trace &mdash; a phenomenon known as
              the{" "}
              <Strong>testing effect</Strong>
              . Memories that are repeatedly retrieved become more durable and
              more accessible. Memories that are never retrieved fade.
            </BodyText>
            <BodyText>
              For a memory system to improve through use, it needs to distinguish
              between different kinds of retrieval outcomes. Surfacing a memory
              is not the same as the user actually relying on it. Engram
              separates these stages:
            </BodyText>
            <div style={{ marginTop: 32, marginBottom: 32, display: "flex", flexDirection: "column", gap: 10 }}>
              {[
                {
                  label: "Surfaced",
                  desc: "Retrieved and presented as context",
                },
                {
                  label: "Selected",
                  desc: "Chosen from candidates as relevant",
                },
                {
                  label: "Used",
                  desc: "Incorporated into a response or decision",
                },
                {
                  label: "Dismissed",
                  desc: "Surfaced but not selected",
                },
                {
                  label: "Corrected",
                  desc: "The user explicitly updated or contradicted the memory",
                },
              ].map((item) => (
                <div
                  key={item.label}
                  style={{ display: "flex", alignItems: "baseline", gap: 12, fontSize: 14 }}
                >
                  <span
                    style={{ ...mono, flexShrink: 0, textAlign: "right", width: 80, color: "rgba(103,232,249,0.8)" }}
                  >
                    {item.label}
                  </span>
                  <span style={{ color: "rgba(122,122,148,0.6)" }}>&mdash;</span>
                  <span style={{ ...body, color: "var(--text-secondary)" }}>{item.desc}</span>
                </div>
              ))}
            </div>
            <BodyText mb={0}>
              These distinctions feed back into activation and consolidation.
              Memories that are consistently used grow stronger. Memories that
              are repeatedly dismissed lose retrieval priority. Corrections
              trigger reconsolidation. The system learns from its own retrieval
              patterns, getting better at surfacing the right information over
              time.
            </BodyText>
          </ScrollReveal>
        </section>

        <Divider />

        {/* -------------------------------------------------------------- */}
        {/*  09 — What Engram Is Not Claiming                               */}
        {/* -------------------------------------------------------------- */}
        <section style={{ paddingTop: 80, paddingBottom: 80 }}>
          <ScrollReveal>
            <SectionNumber n={9} />
            <SectionHeading>What Engram Is Not Claiming</SectionHeading>
            <BodyText>
              Engram is not a brain simulator. It does not model neurons, does
              not replicate synaptic plasticity, and does not claim to implement
              the actual mechanisms of biological memory. The brain is
              extraordinarily complex, and any software system that claims to
              replicate it is overstating its case.
            </BodyText>
            <BodyText>
              What we do claim is that memory science offers a{" "}
              <Strong>better set of design principles</Strong>{" "}
              than the alternatives most AI systems use today. The dominant
              approach &mdash; chunk everything, embed it, store it in a vector
              database, retrieve top-k &mdash; was designed for document search.
              It works for information retrieval. It does not work well for the
              kind of persistent, evolving, context-sensitive memory that AI
              agents need.
            </BodyText>
            <Pullquote>
              Memory science offers a better set of design principles than flat
              storage plus prompt stuffing.
            </Pullquote>
            <BodyText>
              The principles we draw from &mdash; complementary learning systems,
              cue-dependent retrieval, activation-based competition, offline
              consolidation, reconsolidation, adaptive forgetting &mdash; are
              well-established findings with decades of empirical support. We use
              them as design inspiration, not as implementation blueprints.
            </BodyText>
            <BodyText mb={0}>
              The result is a system that behaves more like memory and less like
              a search engine. That is the claim, and we think it is worth making
              carefully.
            </BodyText>
          </ScrollReveal>
        </section>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/*  Footer CTA                                                         */}
      {/* ------------------------------------------------------------------ */}
      <section style={{ padding: "96px 24px" }}>
        <ScrollReveal>
          <div style={{ maxWidth: 640, marginInline: "auto", textAlign: "center" }}>
            <h2 style={{ ...serif, fontSize: "clamp(1.75rem, 3.5vw, 2.5rem)", lineHeight: 1.2, marginBottom: 16 }}>
              Ready to give your AI real memory?
            </h2>
            <p
              style={{ ...body, color: "var(--text-secondary)", marginBottom: 40, maxWidth: 448, marginInline: "auto", lineHeight: 1.7 }}
            >
              Engram is open source and designed to run locally. One brain per
              person. No cloud dependency.
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "center", gap: 16 }}>
              <Link
                to="/docs"
                style={{
                  ...body,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  borderRadius: 10,
                  background: "#67e8f9",
                  padding: "12px 28px",
                  fontSize: 14,
                  fontWeight: 500,
                  color: "#030408",
                  transition: "opacity 0.2s",
                  textDecoration: "none",
                }}
                onMouseEnter={(e) => { e.currentTarget.style.opacity = "0.9"; }}
                onMouseLeave={(e) => { e.currentTarget.style.opacity = "1"; }}
              >
                Read the Docs
              </Link>
              <a
                href="https://github.com/engram-labs/engram"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  ...body,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  borderRadius: 10,
                  border: "1px solid rgba(255,255,255,0.1)",
                  padding: "12px 28px",
                  fontSize: 14,
                  fontWeight: 500,
                  color: "var(--text-primary)",
                  transition: "border-color 0.2s, background 0.2s",
                  textDecoration: "none",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "rgba(255,255,255,0.2)";
                  e.currentTarget.style.background = "rgba(255,255,255,0.03)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)";
                  e.currentTarget.style.background = "transparent";
                }}
              >
                View on GitHub
              </a>
            </div>
          </div>
        </ScrollReveal>
      </section>
    </main>
  );
}
