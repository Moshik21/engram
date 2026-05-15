import { useEffect } from "react";
import { useEngramStore } from "../store";
import { PLAYER_CLASSES, DOMAIN_TO_CONTINENT } from "../constants/rpg";
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer } from "recharts";

export function CharacterSheet() {
  const stats = useEngramStore((s) => s.stats);
  const playerStats = useEngramStore((s) => s.playerStats);
  const computePlayerStats = useEngramStore((s) => s.computePlayerStats);
  const loadStats = useEngramStore((s) => s.loadStats);

  useEffect(() => {
    void loadStats();
  }, [loadStats]);

  useEffect(() => {
    if (stats) computePlayerStats(stats);
  }, [stats, computePlayerStats]);

  const cls = PLAYER_CLASSES[playerStats.playerClass];
  const radarData = Object.entries(playerStats.domainScores).map(([domain, value]) => ({
    domain: DOMAIN_TO_CONTINENT[domain] ?? domain,
    value: Math.min(value, 100),
  }));
  const xpPct = playerStats.xpToNext > 0 ? (playerStats.xp / playerStats.xpToNext) * 100 : 0;
  const understanding = stats
    ? stats.totalEntities > 0
      ? Math.min(100, Math.round((stats.totalRelationships / Math.max(stats.totalEntities, 1)) * 33))
      : 0
    : -1;

  const domainsDiscovered = Object.values(playerStats.domainScores).filter((v) => v > 0).length;

  return (
    <div style={{ padding: 20, height: "100%", overflow: "auto" }}>
      <h2 className="display" style={{ fontSize: 22, marginBottom: 20, color: "#D4A84B" }}>
        Character Sheet
      </h2>

      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr 220px", gap: 16, maxWidth: 1000 }}>
        {/* Player Card */}
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
            animation: "glow-ring 4s ease-in-out infinite",
          }}>
            {cls.icon}
          </div>
          <div style={{
            fontSize: 14, fontWeight: 600, color: cls.color, letterSpacing: "0.05em",
            textShadow: `0 0 12px ${cls.color}30`,
            fontFamily: "var(--font-body)",
          }}>
            {playerStats.playerClass}
          </div>
          <p style={{ color: "var(--text-muted)", fontSize: 11, marginTop: 2, fontStyle: "italic" }}>
            {cls.description}
          </p>

          {/* Level + XP */}
          <div style={{ marginTop: 16, padding: "0 8px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span className="label">Level {playerStats.level}</span>
              <span className="mono" style={{ fontSize: 10, color: "#D4A84B" }}>
                {playerStats.xp}/{playerStats.xpToNext}
              </span>
            </div>
            <div style={{
              height: 8, background: "rgba(255,255,255,0.04)", borderRadius: 4,
              border: "1px solid rgba(212, 168, 75, 0.15)", overflow: "hidden",
            }}>
              <div style={{
                width: `${xpPct}%`, height: "100%", borderRadius: 3,
                background: "linear-gradient(90deg, #8B6914, #D4A84B, #E8C876)",
                boxShadow: xpPct > 80 ? "0 0 8px rgba(212, 168, 75, 0.5)" : "none",
                transition: "width 0.6s ease",
              }} />
            </div>
          </div>

          {/* Understanding */}
          <div style={{ marginTop: 14, padding: "0 8px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span className="label">Understanding</span>
              <span className="mono" style={{ fontSize: 10, color: "var(--info)" }}>
                {understanding >= 0 ? `${understanding}%` : "..."}
              </span>
            </div>
            <div style={{
              height: 4, background: "rgba(255,255,255,0.04)", borderRadius: 2,
              border: "1px solid rgba(129, 140, 248, 0.1)",
            }}>
              <div style={{
                width: `${Math.max(understanding, 0)}%`, height: "100%", borderRadius: 2,
                background: "var(--info)", opacity: 0.7,
                transition: "width 0.6s ease",
              }} />
            </div>
          </div>

          {/* Quick Stats Grid */}
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 16,
            padding: "12px 0 0", borderTop: "1px solid var(--border)",
          }}>
            {[
              { label: "Cycles", value: stats?.totalEpisodes ?? 0 },
              { label: "Skills", value: stats?.totalEntities ?? 0 },
              { label: "Gold", value: playerStats.gold },
              { label: "Domains", value: domainsDiscovered },
            ].map((s) => (
              <div key={s.label} style={{ textAlign: "center" }}>
                <div className="mono" style={{ fontSize: 16, color: "var(--text-primary)", fontWeight: 600 }}>
                  {s.value}
                </div>
                <div className="label" style={{ fontSize: 8 }}>{s.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Radar Chart */}
        <div className="card" style={{ padding: 16, display: "flex", flexDirection: "column" }}>
          <div className="label" style={{ marginBottom: 8, color: "#D4A84B" }}>Domain Mastery</div>
          {radarData.length > 0 ? (
            <div style={{ flex: 1, minHeight: 260 }}>
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                  <PolarGrid stroke="rgba(255,255,255,0.06)" />
                  <PolarAngleAxis
                    dataKey="domain"
                    tick={{ fill: "var(--text-secondary)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                  />
                  <Radar
                    dataKey="value"
                    stroke="#D4A84B"
                    fill="#D4A84B"
                    fillOpacity={0.15}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <span className="label">No domain data yet</span>
            </div>
          )}
        </div>

        {/* Vitals Panel */}
        <div className="card" style={{ padding: 16 }}>
          <div className="label" style={{ marginBottom: 14, color: "#D4A84B" }}>Vitals</div>
          {[
            { label: "HP", sublabel: "Graph Health", value: playerStats.hp, max: 100, color: "#ff6b6b", glow: "rgba(255, 107, 107, 0.3)" },
            { label: "MP", sublabel: "Preference", value: playerStats.mp, max: 100, color: "#4ecdc4", glow: "rgba(78, 205, 196, 0.3)" },
            { label: "XP", sublabel: "Progress", value: playerStats.xp, max: playerStats.xpToNext, color: "#00ff88", glow: "rgba(0, 255, 136, 0.3)" },
          ].map((stat) => {
            const pct = stat.max > 0 ? (stat.value / stat.max) * 100 : 0;
            return (
              <div key={stat.label} style={{ marginBottom: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                  <div>
                    <span style={{ color: stat.color, fontSize: 13, fontWeight: 600, fontFamily: "var(--font-mono)" }}>
                      {stat.label}
                    </span>
                    <span style={{ color: "var(--text-muted)", fontSize: 10, marginLeft: 6 }}>
                      {stat.sublabel}
                    </span>
                  </div>
                  <span className="mono" style={{ fontSize: 12, color: stat.color }}>
                    {stat.value}/{stat.max}
                  </span>
                </div>
                <div style={{
                  height: 10, background: "rgba(255,255,255,0.04)", borderRadius: 5,
                  border: `1px solid ${stat.color}20`, overflow: "hidden",
                }}>
                  <div style={{
                    width: `${pct}%`, height: "100%", borderRadius: 4,
                    background: stat.color,
                    boxShadow: `0 0 8px ${stat.glow}, inset 0 1px 0 rgba(255,255,255,0.2)`,
                    transition: "width 0.6s ease",
                  }} />
                </div>
              </div>
            );
          })}

          {/* Gold display */}
          <div style={{
            marginTop: 8, padding: "10px 0 0", borderTop: "1px solid var(--border)",
            display: "flex", alignItems: "center", gap: 8,
          }}>
            <span style={{ fontSize: 20 }}>{"\u2726"}</span>
            <div>
              <div className="mono" style={{ fontSize: 18, color: "#D4A84B", fontWeight: 600 }}>
                {playerStats.gold}
              </div>
              <div className="label" style={{ fontSize: 8 }}>Gold (Episodes)</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
