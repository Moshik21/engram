interface Fact {
  subject: string;
  predicate: string;
  object: string;
  confidence?: number | null;
}

export function FactTable({ facts }: { facts: Fact[] }) {
  if (!facts || facts.length === 0) return null;

  return (
    <div
      style={{
        borderRadius: "var(--radius-sm)",
        border: "1px solid var(--border)",
        background: "var(--surface)",
        overflow: "hidden",
        fontSize: 12,
      }}
    >
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr
            style={{
              borderBottom: "1px solid var(--border)",
              background: "rgba(255,255,255,0.02)",
            }}
          >
            <th style={thStyle}>Subject</th>
            <th style={thStyle}>Predicate</th>
            <th style={thStyle}>Object</th>
            <th style={{ ...thStyle, width: 60 }}>Conf.</th>
          </tr>
        </thead>
        <tbody>
          {facts.map((fact, i) => (
            <tr
              key={i}
              style={{
                borderBottom: i < facts.length - 1 ? "1px solid var(--border)" : "none",
              }}
            >
              <td style={tdStyle}>{fact.subject}</td>
              <td style={{ ...tdStyle, color: "var(--accent)", fontFamily: "var(--font-mono)", fontSize: 10 }}>
                {fact.predicate}
              </td>
              <td style={tdStyle}>{fact.object}</td>
              <td style={{ ...tdStyle, textAlign: "center", fontFamily: "var(--font-mono)", fontSize: 10 }}>
                {fact.confidence != null ? `${(fact.confidence * 100).toFixed(0)}%` : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const thStyle: React.CSSProperties = {
  padding: "6px 10px",
  textAlign: "left",
  fontWeight: 500,
  fontSize: 10,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  color: "var(--text-muted)",
  fontFamily: "var(--font-mono)",
};

const tdStyle: React.CSSProperties = {
  padding: "6px 10px",
  color: "var(--text-secondary)",
  lineHeight: 1.4,
};
