import { useMemo, useState, type CSSProperties } from "react";

import { useShowcaseExport } from "../lib/showcaseData";

const serif: CSSProperties = { fontFamily: '"Instrument Serif", serif', fontStyle: "italic" };
const body: CSSProperties = { fontFamily: '"Outfit", sans-serif' };
const mono: CSSProperties = { fontFamily: '"JetBrains Mono", monospace' };

type ShowcaseTheaterProps = {
  dataPath?: string;
};

export function ShowcaseTheater({ dataPath }: ShowcaseTheaterProps) {
  const { data, error } = useShowcaseExport(dataPath);
  const beats = data?.beats ?? [];
  const [activeIndex, setActiveIndex] = useState(0);

  const activeBeat = beats[activeIndex];
  const stepLabels = useMemo(
    () => ["Episode", "Cue", "Recall", "Answer"],
    [],
  );

  if (error) {
    return (
      <div style={{ ...body, color: "var(--text-muted)", fontSize: 14 }}>
        Showcase theater unavailable: {error}
      </div>
    );
  }

  if (!activeBeat) {
    return (
      <div style={{ ...body, color: "var(--text-muted)", fontSize: 14 }}>
        Loading showcase beats...
      </div>
    );
  }

  const steps = activeBeat.steps?.length
    ? activeBeat.steps
    : stepLabels.map((label) => ({ label, detail: "" }));

  return (
    <div
      style={{
        border: "1px solid rgba(103,232,249,0.14)",
        borderRadius: 22,
        padding: 24,
        background: "rgba(255,255,255,0.02)",
      }}
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 10,
          marginBottom: 20,
        }}
      >
        {beats.map((beat, index) => (
          <button
            key={beat.id}
            type="button"
            onClick={() => setActiveIndex(index)}
            style={{
              ...mono,
              fontSize: 12,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              borderRadius: 999,
              border: index === activeIndex ? "1px solid rgba(103,232,249,0.45)" : "1px solid rgba(255,255,255,0.08)",
              background: index === activeIndex ? "rgba(103,232,249,0.12)" : "transparent",
              color: index === activeIndex ? "var(--accent)" : "var(--text-muted)",
              padding: "8px 14px",
              cursor: "pointer",
            }}
          >
            {beat.title}
          </button>
        ))}
      </div>

      <div style={{ display: "grid", gap: 18 }}>
        <div>
          <div style={{ ...mono, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 8 }}>
            User
          </div>
          <p style={{ ...body, margin: 0, fontSize: 18, color: "var(--text-primary)", lineHeight: 1.5 }}>
            {activeBeat.user_message}
          </p>
        </div>

        <div style={{ display: "grid", gap: 12 }}>
          {steps.map((step, index) => (
            <div
              key={`${activeBeat.id}-${step.label}`}
              style={{
                display: "grid",
                gridTemplateColumns: "120px 1fr",
                gap: 16,
                alignItems: "start",
                padding: "14px 0",
                borderTop: index === 0 ? "none" : "1px solid rgba(255,255,255,0.06)",
              }}
            >
              <div style={{ ...mono, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--accent)" }}>
                {String(index + 1).padStart(2, "0")} {step.label}
              </div>
              <div>
                <p style={{ ...body, margin: 0, color: "var(--text-secondary)", lineHeight: 1.7, fontSize: 14 }}>
                  {step.detail || activeBeat.narrative}
                </p>
                {step.label === "Recall" && activeBeat.highlights.length > 0 ? (
                  <ul style={{ ...body, margin: "10px 0 0", paddingLeft: 18, color: "var(--text-primary)", fontSize: 14, lineHeight: 1.6 }}>
                    {activeBeat.highlights.map((highlight) => (
                      <li key={highlight}>{highlight}</li>
                    ))}
                  </ul>
                ) : null}
              </div>
            </div>
          ))}
        </div>

        <div
          style={{
            borderRadius: 16,
            border: "1px solid rgba(255,255,255,0.06)",
            padding: 16,
            background: "rgba(255,255,255,0.015)",
          }}
        >
          <div style={{ ...mono, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 8 }}>
            Suggested reply
          </div>
          <p style={{ ...serif, margin: 0, fontSize: 22, color: "var(--text-primary)", lineHeight: 1.45 }}>
            {activeBeat.answer_hint}
          </p>
          <p style={{ ...mono, margin: "12px 0 0", fontSize: 12, color: "var(--text-muted)" }}>
            Action: {activeBeat.action}({activeBeat.query}) · matched: {activeBeat.matched_tokens.join(", ") || "n/a"}
          </p>
        </div>
      </div>
    </div>
  );
}