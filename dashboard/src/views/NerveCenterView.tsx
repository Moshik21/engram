import { Glyph } from '../components/nerve/Glyph';
import { useEngramStore } from '../store';

export function NerveCenterView() {
  const adjudicationCount = useEngramStore(s => s.adjudicationRequests?.length || 0);
  const notificationCount = useEngramStore(s => s.notifications?.length || 0);
  const playerStats = useEngramStore(s => s.cerebralStats);

  return (
    <div className="h-full p-8 overflow-y-auto custom-scrollbar bg-[#030408]">
      <header className="mb-10">
        <div className="flex items-center gap-4 mb-2">
          <div className="w-12 h-12 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
            <Glyph label="N" className="text-amber-500" size={24} />
          </div>
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-white">Labs — Nerve Center</h1>
            <p className="text-zinc-500 text-sm mt-1">
              Operator and experimental panels. Use the Labs sidebar for Adjudication, Immunity Sweep, and Cerebral Profile.
            </p>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
        <section className="p-5 rounded-2xl bg-zinc-900/40 border border-zinc-800/50">
          <div className="text-[9px] font-mono text-zinc-600 uppercase mb-2 tracking-widest">Homeostasis</div>
          <div className="text-2xl font-bold text-white">{playerStats.homeostasis}%</div>
        </section>
        <section className="p-5 rounded-2xl bg-zinc-900/40 border border-zinc-800/50">
          <div className="text-[9px] font-mono text-zinc-600 uppercase mb-2 tracking-widest">Plasticity</div>
          <div className="text-2xl font-bold text-white">{playerStats.morale}%</div>
        </section>
        <section className="p-5 rounded-2xl bg-zinc-900/40 border border-zinc-800/50">
          <div className="text-[9px] font-mono text-zinc-600 uppercase mb-2 tracking-widest">
            Adjudication Queue
          </div>
          <div className="text-2xl font-bold text-amber-400">{adjudicationCount}</div>
          <p className="text-[10px] text-zinc-600 mt-2 leading-snug">
            Hygiene backlog — not the product KPI. Cold Decision recall is success.
          </p>
        </section>
        <section className="p-5 rounded-2xl bg-zinc-900/40 border border-zinc-800/50">
          <div className="text-[9px] font-mono text-zinc-600 uppercase mb-2 tracking-widest">Immunity Signals</div>
          <div className="text-2xl font-bold text-blue-400">{notificationCount}</div>
        </section>
      </div>
    </div>
  );
}