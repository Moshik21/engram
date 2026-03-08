import { type CSSProperties } from "react";

interface FeatureCardProps {
  icon?: string;
  title: string;
  description: string;
  accent?: string;
}

export function FeatureCard({
  icon,
  title,
  description,
  accent = "var(--accent)",
}: FeatureCardProps) {
  const cardStyle: CSSProperties = {
    position: "relative",
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius-lg)",
    backdropFilter: "blur(24px) saturate(1.2)",
    WebkitBackdropFilter: "blur(24px) saturate(1.2)",
    padding: "1.5rem",
    overflow: "hidden",
    transition: [
      "border-color var(--duration-normal) ease",
      "box-shadow var(--duration-normal) ease",
      "transform var(--duration-normal) var(--ease-out-expo)",
    ].join(", "),
    cursor: "default",
  };

  return (
    <div
      className="feature-card"
      style={cardStyle}
      onMouseEnter={(e) => {
        const el = e.currentTarget;
        el.style.borderColor = accent === "var(--accent)"
          ? "rgba(103, 232, 249, 0.25)"
          : accent;
        el.style.boxShadow = `0 2px 8px rgba(0, 0, 0, 0.2), 0 0 24px ${accent === "var(--accent)" ? "rgba(103, 232, 249, 0.1)" : `${accent}22`}`;
        el.style.transform = "translateY(-2px) scale(1.005)";
      }}
      onMouseLeave={(e) => {
        const el = e.currentTarget;
        el.style.borderColor = "var(--border)";
        el.style.boxShadow = "none";
        el.style.transform = "translateY(0) scale(1)";
      }}
    >
      {/* Top accent stripe */}
      <div
        aria-hidden="true"
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 2,
          background: `linear-gradient(90deg, ${accent}, transparent)`,
          borderRadius: "var(--radius-lg) var(--radius-lg) 0 0",
        }}
      />

      {/* Icon */}
      {icon && (
        <div
          style={{
            fontSize: "1.5rem",
            lineHeight: 1,
            marginBottom: "0.75rem",
          }}
          aria-hidden="true"
        >
          {icon}
        </div>
      )}

      {/* Title */}
      <h3
        style={{
          fontFamily: "var(--font-body)",
          fontSize: "1.0625rem",
          fontWeight: 600,
          color: "var(--text-primary)",
          lineHeight: 1.3,
          marginBottom: "0.5rem",
          letterSpacing: "-0.01em",
        }}
      >
        {title}
      </h3>

      {/* Description */}
      <p
        style={{
          fontSize: "0.9375rem",
          lineHeight: 1.6,
          color: "var(--text-secondary)",
          margin: 0,
        }}
      >
        {description}
      </p>
    </div>
  );
}
