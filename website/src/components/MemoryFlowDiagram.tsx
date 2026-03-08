import { useEffect, useState } from "react";

interface FlowStep {
  id: string;
  label: string;
  description: string;
}

const steps: FlowStep[] = [
  { id: "observe", label: "observe", description: "Store episodes cheaply" },
  { id: "cue", label: "cue", description: "Create latent memory traces" },
  { id: "recall", label: "recall", description: "Surface relevant context" },
  {
    id: "feedback",
    label: "feedback",
    description: "Learn from actual use",
  },
  {
    id: "projection",
    label: "projection",
    description: "Extract rich structure",
  },
  {
    id: "consolidation",
    label: "consolidation",
    description: "Merge, mature, prune",
  },
];

function ArrowConnector({ active }: { active: boolean }) {
  return (
    <div
      className="hidden md:flex items-center justify-center flex-shrink-0"
      style={{ width: 48 }}
    >
      <svg
        width="48"
        height="16"
        viewBox="0 0 48 16"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        <line
          x1="0"
          y1="8"
          x2="40"
          y2="8"
          stroke={active ? "var(--accent)" : "var(--border-hover)"}
          strokeWidth="1.5"
          strokeDasharray="4 3"
          style={{
            transition: "stroke 600ms ease",
          }}
        >
          {active && (
            <animate
              attributeName="stroke-dashoffset"
              from="14"
              to="0"
              dur="0.8s"
              repeatCount="indefinite"
            />
          )}
        </line>
        <polyline
          points="36,4 42,8 36,12"
          stroke={active ? "var(--accent)" : "var(--border-hover)"}
          strokeWidth="1.5"
          fill="none"
          strokeLinejoin="round"
          style={{
            transition: "stroke 600ms ease",
          }}
        />
      </svg>
    </div>
  );
}

function MobileArrowConnector({ active }: { active: boolean }) {
  return (
    <div className="flex md:hidden items-center justify-center" style={{ height: 32 }}>
      <svg
        width="16"
        height="32"
        viewBox="0 0 16 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        <line
          x1="8"
          y1="0"
          x2="8"
          y2="24"
          stroke={active ? "var(--accent)" : "var(--border-hover)"}
          strokeWidth="1.5"
          strokeDasharray="4 3"
          style={{
            transition: "stroke 600ms ease",
          }}
        >
          {active && (
            <animate
              attributeName="stroke-dashoffset"
              from="14"
              to="0"
              dur="0.8s"
              repeatCount="indefinite"
            />
          )}
        </line>
        <polyline
          points="4,20 8,26 12,20"
          stroke={active ? "var(--accent)" : "var(--border-hover)"}
          strokeWidth="1.5"
          fill="none"
          strokeLinejoin="round"
          style={{
            transition: "stroke 600ms ease",
          }}
        />
      </svg>
    </div>
  );
}

export function MemoryFlowDiagram() {
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % steps.length);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-full" role="img" aria-label="Memory flow diagram showing the Engram pipeline: observe, cue, recall, feedback, projection, consolidation">
      {/* Desktop: horizontal layout */}
      <div className="hidden md:flex items-stretch justify-center">
        {steps.map((step, i) => {
          const isActive = i === activeIndex;
          return (
            <div key={step.id} className="flex items-center">
              {i > 0 && (
                <ArrowConnector
                  active={
                    i === activeIndex ||
                    (i - 1 === activeIndex && activeIndex < steps.length - 1)
                  }
                />
              )}
              <div
                className="relative flex flex-col items-center text-center"
                style={{
                  width: 152,
                  padding: "1.25rem 0.75rem",
                  background: isActive
                    ? "rgba(103, 232, 249, 0.06)"
                    : "var(--surface)",
                  border: `1px solid ${isActive ? "rgba(103, 232, 249, 0.25)" : "var(--border)"}`,
                  borderRadius: "var(--radius-lg)",
                  backdropFilter: "blur(24px) saturate(1.2)",
                  WebkitBackdropFilter: "blur(24px) saturate(1.2)",
                  boxShadow: isActive
                    ? "0 0 24px rgba(103, 232, 249, 0.15), 0 0 48px rgba(103, 232, 249, 0.05)"
                    : "none",
                  transition:
                    "border-color 600ms ease, background 600ms ease, box-shadow 600ms ease",
                }}
              >
                {/* Pulse ring behind active node */}
                {isActive && (
                  <span
                    className="absolute inset-0 rounded-2xl pointer-events-none"
                    style={{
                      border: "1px solid rgba(103, 232, 249, 0.15)",
                      animation: "glow-breathe 2s ease-in-out infinite",
                      borderRadius: "var(--radius-lg)",
                    }}
                    aria-hidden="true"
                  />
                )}

                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.8125rem",
                    fontWeight: 600,
                    letterSpacing: "-0.01em",
                    color: isActive ? "var(--accent)" : "var(--text-primary)",
                    transition: "color 600ms ease",
                    marginBottom: 6,
                  }}
                >
                  {step.label}
                </span>
                <span
                  style={{
                    fontSize: "0.75rem",
                    lineHeight: 1.4,
                    color: isActive
                      ? "var(--text-secondary)"
                      : "var(--text-muted)",
                    transition: "color 600ms ease",
                  }}
                >
                  {step.description}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Mobile: vertical layout */}
      <div className="flex md:hidden flex-col items-center">
        {steps.map((step, i) => {
          const isActive = i === activeIndex;
          return (
            <div key={step.id} className="flex flex-col items-center w-full" style={{ maxWidth: 280 }}>
              {i > 0 && (
                <MobileArrowConnector
                  active={
                    i === activeIndex ||
                    (i - 1 === activeIndex && activeIndex < steps.length - 1)
                  }
                />
              )}
              <div
                className="relative flex flex-col items-center text-center w-full"
                style={{
                  padding: "1rem 1.25rem",
                  background: isActive
                    ? "rgba(103, 232, 249, 0.06)"
                    : "var(--surface)",
                  border: `1px solid ${isActive ? "rgba(103, 232, 249, 0.25)" : "var(--border)"}`,
                  borderRadius: "var(--radius-lg)",
                  backdropFilter: "blur(24px) saturate(1.2)",
                  WebkitBackdropFilter: "blur(24px) saturate(1.2)",
                  boxShadow: isActive
                    ? "0 0 24px rgba(103, 232, 249, 0.15), 0 0 48px rgba(103, 232, 249, 0.05)"
                    : "none",
                  transition:
                    "border-color 600ms ease, background 600ms ease, box-shadow 600ms ease",
                }}
              >
                {isActive && (
                  <span
                    className="absolute inset-0 rounded-2xl pointer-events-none"
                    style={{
                      border: "1px solid rgba(103, 232, 249, 0.15)",
                      animation: "glow-breathe 2s ease-in-out infinite",
                      borderRadius: "var(--radius-lg)",
                    }}
                    aria-hidden="true"
                  />
                )}

                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.8125rem",
                    fontWeight: 600,
                    letterSpacing: "-0.01em",
                    color: isActive ? "var(--accent)" : "var(--text-primary)",
                    transition: "color 600ms ease",
                    marginBottom: 4,
                  }}
                >
                  {step.label}
                </span>
                <span
                  style={{
                    fontSize: "0.75rem",
                    lineHeight: 1.4,
                    color: isActive
                      ? "var(--text-secondary)"
                      : "var(--text-muted)",
                    transition: "color 600ms ease",
                  }}
                >
                  {step.description}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
