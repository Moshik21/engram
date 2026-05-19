import { useEffect } from "react";
import { useEngramStore } from "../store";
import { NEURAL_SPECIALIZATIONS, DOMAIN_TO_REGION } from "../constants/nerveCenter";
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer } from "recharts";

export function CerebralProfile() {
  const stats = useEngramStore((s) => s.stats);
  const playerStats = useEngramStore((s) => s.cerebralStats);
  const computePlayerStats = useEngramStore((s) => s.computeCerebralStats);
  const loadStats = useEngramStore((s) => s.loadStats);

  useEffect(() => {
    void loadStats();
  }, [loadStats]);

  useEffect(() => {
    if (stats) computePlayerStats(stats);
  }, [stats, computePlayerStats]);

  const cls = NEURAL_SPECIALIZATIONS[playerStats.specialization] || NEURAL_SPECIALIZATIONS["Polymath"];
  const radarData = Object.entries(playerStats.domainScores).map(([domain, value]) => ({
    domain: DOMAIN_TO_REGION[domain] ?? domain,
    value: Math.min(value, 100),
  }));
  const plasticityPct = playerStats.plasticityToNext > 0 ? (playerStats.plasticity / playerStats.plasticityToNext) * 100 : 0;

  // Coherence: density of relationships
  const coherence = stats
    ? stats.totalEntities > 0
      ? Math.min(100, Math.round((stats.totalRelationships / Math.max(stats.totalEntities, 1)) * 33))
      : 0
    : -1;

  const regionsMapped = Object.values(playerStats.domainScores).filter((v) => v > 0).length;

  return (
    <div style={{ padding: 20, height: "100%", overflow: "auto" }}>
      <h2 className="display" style={{ fontSize: 22, marginBottom: 20, color: "#67e8f9" }}>
        Cerebral Profile
      </h2>

      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr 220px", gap: 16, maxWidth: 1000 }}>
        {/* Overseer Card */}
        <div className="card" style={{ padding: 20, textAlign: "center", position: "relative", overflow: "hidden" }}>
          {/* Ambient class glow */}
          <div style={{
            position: "absolute", top: -20, left: "50%", transform: "translateX(-50%)",
            width: 120, height: 120, borderRadius: "50%",
            background: `radial-gradient(circle, ${cls.color}15 0%, transparent 70%)`,
            pointerEvents: "none",
          }} />

          <div style={{
            fontSize: 56, marginBottom: 6, lineHeight: 1,
            filter: `drop-shadow(0 0 12px ${cls.color}40)`,
          }}>
            {cls.icon}
          </div>
          <div style={{
            fontSize: 14, fontWeight: 600, color: cls.color, letterSpacing: "0.05em",
            textShadow: `0 0 12px ${cls.color}30`,
            fontFamily: "var(--font-mono)",
            textTransform: "uppercase",
          }}>
            {playerStats.specialization}
          </div>
          <p style={{ color: "var(--text-muted)", fontSize: 11, marginTop: 2, fontStyle: "italic" }}>
            {cls.description}
          </p>

          {/* Cortical Level + Plasticity */}
          <div style={{ marginTop: 16, padding: "0 8px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span className="label">Level {playerStats.level}</span>
              <span className="mono" style={{ fontSize: 10, color: "#67e8f9" }}>
                {playerStats.plasticity}/{playerStats.plasticityToNext} Plasticity
              </span>
            </div>
            <div style={{
              height: 8, background: "rgba(255,255,255,0.04)", borderRadius: 4,
              border: "1px solid rgba(103, 232, 249, 0.15)", overflow: "hidden",
            }}>
              <div style={{
                width: `${plasticityPct}%`, height: "100%", borderRadius: 3,
                background: "linear-gradient(90deg, #0891b2, #67e8f9, #22d3ee)",
                boxShadow: plasticityPct > 80 ? "0 0 8px rgba(103, 232, 249, 0.5)" : "none",
                transition: "width 0.6s ease",
              }} />
            </div>
          </div>

          {/* Coherence */}
          <div style={{ marginTop: 14, padding: "0 8px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span className="label">Coherence</span>
              <span className="mono" style={{ fontSize: 10, color: "var(--info)" }}>
                {coherence >= 0 ? `${coherence}%` : "..."}
              </span>
            </div>
            <div style={{
              height: 4, background: "rgba(255,255,255,0.04)", borderRadius: 2,
              border: "1px solid rgba(129, 140, 248, 0.1)",
              overflow: "hidden",
            }}>
              <div style={{
                width: `${coherence}%`, height: "100%", borderRadius: 2,
                background: "var(--info)",
                transition: "width 1s ease",
              }} />
            </div>
          </div>
        </div>

        {/* Neural Radar */}
        <div className="card" style={{ padding: 12, display: "flex", flexDirection: "column" }}>
          <div className="label" style={{ marginBottom: 8, padding: "4px 8px" }}>Synaptic Distribution</div>
          <div style={{ flex: 1, minHeight: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                <PolarGrid stroke="var(--border)" strokeDasharray="3 3" />
                <PolarAngleAxis
                  dataKey="domain"
                  tick={{ fill: "var(--text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                />
                <Radar
                  name="Cerebral Density"
                  dataKey="value"
                  stroke={cls.color}
                  fill={cls.color}
                  fillOpacity={0.2}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Vital Signs */}
        <div className="stagger" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div className="card" style={{ padding: "12px 16px" }}>
            <div className="label" style={{ color: "var(--danger)" }}>Homeostasis</div>
            <div className="display" style={{ fontSize: 24, marginTop: 4 }}>
              {playerStats.homeostasis}%
            </div>
            <div className="mono" style={{ fontSize: 9, color: "var(--text-ghost)", marginTop: 4 }}>
              Graph connectivity health
            </div>
          </div>

          <div className="card" style={{ padding: "12px 16px" }}>
            <div className="label" style={{ color: "var(--success)" }}>Plasticity</div>
            <div className="display" style={{ fontSize: 24, marginTop: 4 }}>
              {playerStats.morale}%
            </div>
            <div className="mono" style={{ fontSize: 9, color: "var(--text-ghost)", marginTop: 4 }}>
              Feedback-driven tuning morale
            </div>
          </div>

          <div className="card" style={{ padding: "12px 16px" }}>
            <div className="label" style={{ color: "#fbbf24" }}>Synaptic Credits</div>
            <div className="display" style={{ fontSize: 24, marginTop: 4 }}>
              {playerStats.synapticCredits}
            </div>
            <div className="mono" style={{ fontSize: 9, color: "var(--text-ghost)", marginTop: 4 }}>
              Successful episodic recalls
            </div>
          </div>

          <div className="card" style={{ padding: "12px 16px", flex: 1 }}>
            <div className="label">Field Analysis</div>
            <div style={{ marginTop: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>Nodes Registered</span>
                <span className="mono" style={{ fontSize: 11 }}>{stats?.totalEntities || 0}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>Links Formed</span>
                <span className="mono" style={{ fontSize: 11 }}>{stats?.totalRelationships || 0}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>Regions Mapped</span>
                <span className="mono" style={{ fontSize: 11 }}>{regionsMapped}/7</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
