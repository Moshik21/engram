import { useRef, useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from "recharts";
import { activationColor } from "../../../lib/colors";

interface ActivationEntity {
  name: string;
  entityType: string;
  activation: number;
}

export function ActivationChart({ entities }: { entities: ActivationEntity[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(400);

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w && w > 0) setWidth(Math.round(w));
    });
    ro.observe(containerRef.current);
    // Initial measurement
    const w = containerRef.current.getBoundingClientRect().width;
    if (w > 0) setWidth(Math.round(w));
    return () => ro.disconnect();
  }, []);

  if (!entities || entities.length < 2) return null;

  const data = entities.map((e) => ({
    name: e.name.length > 14 ? e.name.slice(0, 12) + "..." : e.name,
    fullName: e.name,
    activation: parseFloat((e.activation * 100).toFixed(1)),
    color: activationColor(e.activation),
  }));

  const chartHeight = data.length * 28;

  return (
    <div
      ref={containerRef}
      style={{
        borderRadius: "var(--radius-sm)",
        border: "1px solid var(--border)",
        background: "var(--surface)",
        padding: "12px 8px 4px",
        height: Math.max(120, chartHeight + 30),
      }}
    >
      <div
        style={{
          fontSize: 10,
          fontFamily: "var(--font-mono)",
          color: "var(--text-muted)",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          marginBottom: 8,
          paddingLeft: 4,
        }}
      >
        Activation Levels
      </div>
      <BarChart
        data={data}
        layout="vertical"
        width={width - 16}
        height={chartHeight}
        margin={{ left: 0, right: 12, top: 0, bottom: 0 }}
      >
        <XAxis type="number" domain={[0, 100]} hide />
        <YAxis
          type="category"
          dataKey="name"
          width={80}
          tick={{ fill: "#94a3b8", fontSize: 10, fontFamily: "var(--font-mono)" }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          formatter={(value) => [`${value}%`, "Activation"]}
          contentStyle={{
            background: "var(--surface-solid)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-xs)",
            fontSize: 11,
            color: "var(--text-primary)",
          }}
        />
        <Bar dataKey="activation" radius={[0, 3, 3, 0]} barSize={14}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </div>
  );
}
