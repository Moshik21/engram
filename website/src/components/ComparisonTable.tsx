interface Row {
  approach: string;
  whatItDoes: string;
  whereItBreaks: string;
  highlight?: boolean;
}

const rows: Row[] = [
  {
    approach: "Chat history",
    whatItDoes: "Keeps recent text around",
    whereItBreaks: "Flat, noisy, session-bound",
  },
  {
    approach: "Bigger context windows",
    whatItDoes: "Fits more tokens",
    whereItBreaks: "Expensive, unstructured, non-durable",
  },
  {
    approach: "Basic RAG",
    whatItDoes: "Retrieves related text",
    whereItBreaks: "Often helpful, rarely memory-aware",
  },
  {
    approach: "Engram",
    whatItDoes: "Builds a private, self-organizing memory loop",
    whereItBreaks:
      "More ambitious, but much closer to real continuity",
    highlight: true,
  },
];

export function ComparisonTable() {
  return (
    <div
      className="w-full overflow-x-auto"
      style={{
        borderRadius: "var(--radius-lg)",
        border: "1px solid var(--border)",
        background: "var(--surface)",
        backdropFilter: "blur(24px) saturate(1.2)",
        WebkitBackdropFilter: "blur(24px) saturate(1.2)",
      }}
    >
      <table
        className="comparison-table"
        style={{ minWidth: 640 }}
        role="table"
      >
        <thead>
          <tr>
            <th scope="col">Approach</th>
            <th scope="col">What It Does</th>
            <th scope="col">Where It Breaks</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.approach}
              style={
                row.highlight
                  ? {
                      background: "rgba(103, 232, 249, 0.04)",
                      borderLeft: "2px solid var(--accent)",
                      position: "relative",
                    }
                  : undefined
              }
            >
              <td
                style={
                  row.highlight
                    ? {
                        color: "var(--accent)",
                        fontWeight: 600,
                        fontFamily: "var(--font-mono)",
                        fontSize: "0.875rem",
                        letterSpacing: "-0.01em",
                      }
                    : undefined
                }
              >
                {row.highlight ? row.approach : row.approach}
              </td>
              <td
                style={
                  row.highlight
                    ? { color: "var(--text-primary)" }
                    : undefined
                }
              >
                {row.whatItDoes}
              </td>
              <td
                style={
                  row.highlight
                    ? {
                        color: "var(--text-primary)",
                        fontStyle: "italic",
                      }
                    : undefined
                }
              >
                {row.whereItBreaks}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Accent glow on highlighted row — rendered via box-shadow on wrapper */}
      <style>{`
        .comparison-table tbody tr:last-child {
          box-shadow:
            inset 0 0 32px rgba(103, 232, 249, 0.03),
            0 0 24px rgba(103, 232, 249, 0.05);
        }
        .comparison-table tbody tr:last-child td {
          border-bottom: none;
        }
      `}</style>
    </div>
  );
}
