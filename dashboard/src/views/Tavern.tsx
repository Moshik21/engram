import { MemoryFeed } from "../components/MemoryFeed";

export function Tavern() {
  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Atmospheric header */}
      <div style={{
        padding: "20px 20px 16px",
        borderBottom: "1px solid var(--border)",
        background: "linear-gradient(180deg, rgba(212, 168, 75, 0.03) 0%, transparent 100%)",
      }}>
        <h2 className="display" style={{ fontSize: 22, color: "#D4A84B", marginBottom: 4 }}>
          The Tavern
        </h2>
        <p style={{ color: "var(--text-muted)", fontSize: 12, fontStyle: "italic", lineHeight: 1.5 }}>
          Travelers share tales of knowledge gained and memories forged.
          <br />
          Each episode is a story waiting to be heard.
        </p>
      </div>
      <div style={{ flex: 1, overflow: "auto", padding: 20 }}>
        <MemoryFeed />
      </div>
    </div>
  );
}
