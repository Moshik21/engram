import { memo } from "react";
import type { DynamicToolUIPart } from "ai";
import { EntityCards } from "./EntityCards";
import { RelationshipMiniGraph } from "./RelationshipMiniGraph";
import { FactTable } from "./FactTable";
import { ActivationChart } from "./ActivationChart";
import { MemoryTimeline } from "./MemoryTimeline";

function ToolSkeleton({ name }: { name: string }) {
  return (
    <div
      style={{
        padding: "14px 16px",
        borderRadius: "var(--radius-sm)",
        border: "1px solid var(--border)",
        background: "var(--surface)",
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: "var(--accent)",
          boxShadow: "0 0 8px var(--accent-glow-strong)",
          animation: "pulse-soft 1.5s ease-in-out infinite",
          flexShrink: 0,
        }}
      />
      <span
        style={{
          fontSize: 11,
          color: "var(--text-muted)",
          fontFamily: "var(--font-mono)",
        }}
      >
        Loading {name}...
      </span>
    </div>
  );
}

export const ToolRenderer = memo(function ToolRenderer({ part }: { part: DynamicToolUIPart }) {
  const { toolName, state } = part;
  const args = "input" in part && part.input ? (part.input as Record<string, unknown>) : {};

  if (state !== "output-available") {
    return <ToolSkeleton name={toolName} />;
  }

  switch (toolName) {
    case "show_entities":
      return <EntityCards entities={args.entities as never} />;
    case "show_relationship_graph":
      return <RelationshipMiniGraph data={args as never} />;
    case "show_facts":
      return <FactTable facts={args.facts as never} />;
    case "show_activation_chart":
      return <ActivationChart entities={args.entities as never} />;
    case "show_timeline":
      return <MemoryTimeline episodes={args.episodes as never} />;
    default:
      return null;
  }
}, (prev, next) => prev.part.toolCallId === next.part.toolCallId && prev.part.state === next.part.state);
