import { useEngramStore } from "../../store";
import { RecallResultCard } from "./RecallResultCard";

export function KnowledgeRecallResults() {
  const results = useEngramStore((s) => s.knowledgeResults);
  const isRecalling = useEngramStore((s) => s.isRecalling);
  const query = useEngramStore((s) => s.knowledgeQuery);

  if (isRecalling) {
    return (
      <div style={{ padding: 20 }}>
        {[1, 2, 3].map((i) => (
          <div key={i} className="skeleton" style={{ height: 80, marginBottom: 8 }} />
        ))}
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div style={{ textAlign: "center", padding: "60px 20px" }}>
        <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
          No results for "{query}"
        </p>
      </div>
    );
  }

  return (
    <div className="stagger">
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <span className="label">{results.length} results</span>
      </div>
      {results.map((result, i) => (
        <RecallResultCard key={i} result={result} />
      ))}
    </div>
  );
}
