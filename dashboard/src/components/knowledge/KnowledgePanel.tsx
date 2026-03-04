import { useEffect } from "react";
import { useEngramStore } from "../../store";
import { KnowledgeSearchBar } from "./KnowledgeSearchBar";
import { KnowledgeContent } from "./KnowledgeContent";
import { KnowledgeInputBar } from "./KnowledgeInputBar";

export function KnowledgePanel() {
  const loadEntityGroups = useEngramStore((s) => s.loadEntityGroups);

  useEffect(() => {
    loadEntityGroups();
  }, [loadEntityGroups]);

  return (
    <div
      className="animate-fade-in"
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflow: "hidden",
      }}
    >
      {/* Search bar section */}
      <div style={{ padding: "0 20px 12px", flexShrink: 0 }}>
        <KnowledgeSearchBar />
      </div>

      {/* Main content area */}
      <KnowledgeContent />

      {/* Input bar */}
      <KnowledgeInputBar />
    </div>
  );
}
