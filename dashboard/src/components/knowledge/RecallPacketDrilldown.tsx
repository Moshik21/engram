import { useEngramStore } from "../../store";
import type { RecallPacket } from "../../store/types";

function packetType(packet: RecallPacket) {
  return packet.packetType ?? packet.packet_type ?? "memory";
}

function whyNow(packet: RecallPacket) {
  return packet.trust?.whyNow ?? packet.trust?.why_now ?? packet.whyNow ?? packet.why_now ?? "";
}

function trustSource(packet: RecallPacket) {
  return packet.trust?.source ?? "unknown";
}

function freshness(packet: RecallPacket) {
  return packet.trust?.freshness ?? "unknown";
}

function confidence(packet: RecallPacket) {
  const value = packet.trust?.confidence ?? packet.confidence;
  return typeof value === "number" && Number.isFinite(value)
    ? `${Math.round(value * 100)}%`
    : "n/a";
}

function count(value: number | null | undefined, fallback: unknown[] | undefined) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return fallback?.length ?? 0;
}

function trustCount(packet: RecallPacket, camel: "confirmedCount" | "correctedCount" | "dismissedCount", snake: "confirmed_count" | "corrected_count" | "dismissed_count") {
  const value = packet.trust?.[camel] ?? packet.trust?.[snake];
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function trustTimestamp(packet: RecallPacket, camel: "lastConfirmedAt" | "lastCorrectedAt" | "lastDismissedAt", snake: "last_confirmed_at" | "last_corrected_at" | "last_dismissed_at") {
  return packet.trust?.[camel] ?? packet.trust?.[snake] ?? null;
}

export function RecallPacketDrilldown() {
  const packets = useEngramStore((s) => s.knowledgePackets);
  const status = useEngramStore((s) => s.knowledgeRecallStatus);
  const lifecycle = useEngramStore((s) => s.knowledgeRecallLifecycle);
  const budget = useEngramStore((s) => s.knowledgeRecallBudget);
  const hasRuntimeDetails = Boolean(status || lifecycle || budget);
  if (!packets.length && !hasRuntimeDetails) return null;

  const runtimeTone = budget?.degraded || lifecycle?.degraded || status === "degraded"
    ? "#fb7185"
    : budget?.budgetMiss || lifecycle?.timeout
      ? "#facc15"
      : "#34d399";

  return (
    <section
      aria-label="Recall packet trust"
      style={{
        borderBottom: "1px solid var(--border)",
        padding: "10px 14px",
        display: "grid",
        gap: 8,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
        }}
      >
        <span style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase" }}>
          Recall Packets
        </span>
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          {packets.length.toLocaleString()} surfaced
        </span>
      </div>

      {hasRuntimeDetails && (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: 8,
            fontSize: 11,
            color: "var(--text-secondary)",
          }}
        >
          <span style={{ color: runtimeTone, fontWeight: 700 }}>
            {status ?? "ok"}
          </span>
          {budget && (
            <>
              <span>{budget.surface}/{budget.profile}</span>
              <span>{Math.round(budget.durationMs).toLocaleString()}ms</span>
              <span>budget {budget.maxWallMs.toLocaleString()}ms</span>
            </>
          )}
          {(budget?.timeout || lifecycle?.timeout) && <span>timeout</span>}
          {(budget?.skipReason || lifecycle?.skipReason) && (
            <span>{budget?.skipReason ?? lifecycle?.skipReason}</span>
          )}
        </div>
      )}

      <div style={{ display: "grid", gap: 8 }}>
        {packets.slice(0, 3).map((packet, index) => {
          const provenanceCount = count(
            packet.trust?.provenanceCount ?? packet.trust?.provenance_count,
            packet.provenance,
          );
          const evidenceCount = count(
            packet.trust?.evidenceCount ?? packet.trust?.evidence_count,
            packet.evidenceLines ?? packet.evidence_lines,
          );
          const confirmedCount = trustCount(packet, "confirmedCount", "confirmed_count");
          const correctedCount = trustCount(packet, "correctedCount", "corrected_count");
          const dismissedCount = trustCount(packet, "dismissedCount", "dismissed_count");
          const lastConfirmedAt = trustTimestamp(packet, "lastConfirmedAt", "last_confirmed_at");
          const lastCorrectedAt = trustTimestamp(packet, "lastCorrectedAt", "last_corrected_at");
          const lastDismissedAt = trustTimestamp(packet, "lastDismissedAt", "last_dismissed_at");
          return (
            <article
              key={`${packet.title ?? packetType(packet)}-${index}`}
              style={{
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)",
                padding: 10,
                display: "grid",
                gap: 6,
                background: "rgba(255,255,255,0.025)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <strong style={{ fontSize: 13, color: "var(--text-primary)" }}>
                  {packet.title ?? packetType(packet)}
                </strong>
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  {packetType(packet)} | {trustSource(packet)} | {freshness(packet)}
                </span>
              </div>
              {packet.summary && (
                <p style={{ margin: 0, color: "var(--text-secondary)", fontSize: 12 }}>
                  {packet.summary}
                </p>
              )}
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, fontSize: 11 }}>
                <span>confidence {confidence(packet)}</span>
                <span>evidence {evidenceCount}</span>
                <span>provenance {provenanceCount}</span>
                <span>
                  belief{" "}
                  {packet.trust?.beliefStatus ?? packet.trust?.belief_status ?? "unknown"}
                </span>
                {(confirmedCount > 0 || correctedCount > 0 || dismissedCount > 0) && (
                  <span>
                    feedback confirmed {confirmedCount} | corrected {correctedCount} |
                    dismissed {dismissedCount}
                  </span>
                )}
              </div>
              {(lastConfirmedAt || lastCorrectedAt || lastDismissedAt) && (
                <p style={{ margin: 0, color: "var(--text-muted)", fontSize: 11 }}>
                  last feedback{" "}
                  {lastConfirmedAt ? `confirmed ${lastConfirmedAt}` : ""}
                  {lastCorrectedAt ? ` corrected ${lastCorrectedAt}` : ""}
                  {lastDismissedAt ? ` dismissed ${lastDismissedAt}` : ""}
                </p>
              )}
              {whyNow(packet) && (
                <p style={{ margin: 0, color: "var(--text-muted)", fontSize: 11 }}>
                  {whyNow(packet)}
                </p>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
