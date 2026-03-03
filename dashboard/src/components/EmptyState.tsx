export function EmptyState() {
  return (
    <div
      className="animate-fade-in"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        width: "100%",
        gap: 24,
      }}
    >
      {/* Animated rings */}
      <div style={{ position: "relative", width: 120, height: 120 }}>
        <div
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: "50%",
            border: "1px solid var(--border)",
            animation: "glow-ring 4s ease-in-out infinite",
          }}
        />
        <div
          style={{
            position: "absolute",
            inset: 20,
            borderRadius: "50%",
            border: "1px solid var(--border-hover)",
            animation: "glow-ring 4s ease-in-out infinite 0.5s",
          }}
        />
        <div
          style={{
            position: "absolute",
            inset: 40,
            borderRadius: "50%",
            border: "1px solid var(--border-active)",
            animation: "glow-ring 4s ease-in-out infinite 1s",
          }}
        />
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: "var(--accent)",
            boxShadow: "0 0 20px var(--accent-glow-strong)",
          }}
        />
      </div>

      <div style={{ textAlign: "center" }}>
        <h2
          className="display"
          style={{
            fontSize: 28,
            color: "#fff",
            margin: "0 0 8px",
            lineHeight: 1.2,
          }}
        >
          No memories yet
        </h2>
        <p
          style={{
            maxWidth: 320,
            fontSize: 13,
            lineHeight: 1.7,
            color: "var(--text-secondary)",
            margin: 0,
          }}
        >
          Use the MCP tools to store memories. Entities and their
          relationships will appear here as an interactive graph.
        </p>
      </div>

      <div className="pill" style={{ marginTop: 4 }}>
        awaiting input
      </div>
    </div>
  );
}
