import React, { useEffect } from 'react';
import { useEngramStore } from '../../store';
import { Glyph } from './Glyph';

function formatTimeAgo(timestampMs: number): string {
  const diff = Math.max(0, Date.now() - timestampMs);
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

export const ImmunitySweep: React.FC = () => {
  const {
    notifications,
    loadNotifications,
    dismissNotifications
  } = useEngramStore();

  useEffect(() => {
    loadNotifications();
  }, [loadNotifications]);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'dream_association': return <Glyph label="D" className="text-purple-400" size={20} />;
      case 'schema_discovery': return <Glyph label="S" className="text-blue-400" size={20} />;
      case 'entity_merge': return <Glyph label="M" className="text-amber-400" size={20} />;
      case 'entity_maturation': return <Glyph label="G" className="text-emerald-400" size={20} />;
      case 'immunity_sweep': return <Glyph label="I" className="text-blue-400" size={20} />;
      default: return <Glyph label="N" className="text-zinc-400" size={20} />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'dream_association': return 'border-purple-500/20 bg-purple-500/5';
      case 'schema_discovery': return 'border-blue-500/20 bg-blue-500/5';
      case 'entity_merge': return 'border-amber-500/20 bg-amber-500/5';
      case 'entity_maturation': return 'border-emerald-500/20 bg-emerald-500/5';
      case 'immunity_sweep': return 'border-blue-500/20 bg-blue-500/5';
      default: return 'border-zinc-800 bg-zinc-900/50';
    }
  };

  if (notifications.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-zinc-500 bg-black/20 rounded-2xl border border-zinc-800/50 p-12">
        <div className="relative mb-6">
          <div className="absolute inset-0 bg-blue-500/5 blur-3xl rounded-full" />
          <Glyph label="I" size={80} className="relative opacity-20 text-blue-400" />
        </div>
        <h3 className="text-2xl font-bold text-zinc-300">Immune System Nominal</h3>
        <p className="mt-2 text-zinc-500 max-w-xs text-center">
          No active maintenance events detected. Your neural field is clean and synthesized.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-black/40 backdrop-blur-md rounded-2xl border border-zinc-800/50 overflow-hidden shadow-2xl">
      <header className="p-6 border-b border-zinc-800/50 flex justify-between items-center bg-zinc-900/20">
        <div>
          <h2 className="text-xl font-bold text-zinc-100 flex items-center gap-3">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <Glyph label="I" className="text-blue-400" size={20} />
            </div>
            Immunity Sweep
          </h2>
          <p className="text-zinc-500 text-xs mt-1">Review autonomous maintenance and synthesis events.</p>
        </div>
        <button
          onClick={() => dismissNotifications(notifications.map(n => n.id))}
          className="px-4 py-2 bg-zinc-800/50 hover:bg-zinc-700 text-zinc-300 text-xs font-bold rounded-lg border border-zinc-700/50 transition-all flex items-center gap-2 group"
        >
          <Glyph label="OK" size={14} className="group-hover:text-blue-400 transition-colors" />
          Acknowledge All
        </button>
      </header>

      <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
        {notifications.map((notification) => (
          <div
            key={notification.id}
            className={`
              relative group p-6 rounded-2xl border transition-all duration-500 transform hover:-translate-y-1
              ${getTypeColor(notification.notification_type)}
              hover:shadow-[0_20px_40px_rgba(0,0,0,0.5)] hover:bg-opacity-10
            `}
          >
            <div className="flex gap-5">
              <div className="flex-shrink-0 mt-1 p-3 bg-black/40 rounded-xl border border-zinc-800/50 group-hover:scale-110 transition-transform duration-500">
                {getTypeIcon(notification.notification_type)}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex justify-between items-start">
                  <h4 className="text-sm font-bold text-zinc-100 group-hover:text-white transition-colors truncate pr-12 tracking-tight">
                    {notification.title}
                  </h4>
                  <button
                    onClick={() => dismissNotifications([notification.id])}
                    className="absolute top-6 right-6 p-1.5 text-zinc-600 hover:text-white transition-colors hover:bg-white/5 rounded-md"
                  >
                    <Glyph label="X" size={14} />
                  </button>
                </div>
                <div className="mt-3 text-[13px] text-zinc-400 leading-relaxed whitespace-pre-wrap font-medium">
                  {notification.body}
                </div>
                <div className="mt-5 flex items-center gap-6 text-[10px] text-zinc-500 font-mono uppercase tracking-widest">
                  <div className="flex items-center gap-2">
                    <Glyph label="T" size={12} className="text-zinc-600" />
                    {formatTimeAgo(notification.created_at * 1000)}
                  </div>
                  {notification.source_cycle_id && (
                    <div className="flex items-center gap-2">
                      <div className="w-1 h-1 rounded-full bg-zinc-700" />
                      <span className="text-zinc-600">Cycle</span>
                      <span className="text-zinc-400">{notification.source_cycle_id.slice(0, 8)}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Background Glow on Hover */}
            <div className={`
              absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-1000 pointer-events-none rounded-2xl
              bg-gradient-to-br from-white/5 to-transparent
            `} />
          </div>
        ))}
      </div>
    </div>
  );
};
