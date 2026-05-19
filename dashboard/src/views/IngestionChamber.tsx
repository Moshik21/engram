import { MemoryFeed } from "../components/MemoryFeed";

export function IngestionChamber() {
  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Atmospheric header */}
      <div style={{
        padding: "20px 20px 16px",
        borderBottom: "1px solid var(--border)",
        background: "linear-gradient(180deg, rgba(103, 232, 249, 0.03) 0%, transparent 100%)",
      }}>
        <h2 className="display" style={{ fontSize: 22, color: "#67e8f9", marginBottom: 4 }}>
          Ingestion Chamber
        </h2>
        <p style={{ color: "var(--text-muted)", fontSize: 12, fontStyle: "italic", lineHeight: 1.5 }}>
          Raw cognitive stimulus arriving from the sensorium.
          <br />
          Each signal is a potential thread in the synaptic tapestry.
        </p>
      </div>
      <div style={{ flex: 1, overflow: "auto", padding: 20 }}>
        <MemoryFeed />
      </div>
    </div>
  );
}
