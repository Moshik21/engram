import { AdjudicationPanel } from '../components/nerve/AdjudicationPanel';
import { Glyph } from '../components/nerve/Glyph';
import { ImmunitySweep } from '../components/nerve/ImmunitySweep';
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
            <h1 className="text-3xl font-bold tracking-tight text-white">Nerve Center</h1>
            <p className="text-zinc-500 text-sm mt-1">Centralized neural command and maintenance.</p>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        {/* Adjudication Section */}
        <div className="xl:col-span-2 space-y-6">
           <div className="flex items-center justify-between px-2">
             <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-zinc-500 flex items-center gap-2">
               <Glyph label="A" size={14} className="text-amber-500" />
               Neural Adjudication
               {adjudicationCount > 0 && (
                 <span className="bg-amber-500 text-black px-2 py-0.5 rounded-full text-[10px] ml-2 tracking-normal">
                   {adjudicationCount}
                 </span>
               )}
             </h3>
           </div>
           <div className="h-[600px]">
             <AdjudicationPanel />
           </div>
        </div>

        {/* Sidebar Sections */}
        <div className="space-y-8">
          {/* Quick Stats */}
          <section className="space-y-4">
            <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-zinc-500 px-2 flex items-center gap-2">
              <Glyph label="H" size={14} className="text-blue-400" />
              Cerebral Health
            </h3>
            <div className="grid grid-cols-2 gap-4">
               <div className="p-5 rounded-2xl bg-zinc-900/40 border border-zinc-800/50 hover:border-zinc-700 transition-all group">
                 <div className="text-[9px] font-mono text-zinc-600 uppercase mb-2 tracking-widest">Homeostasis</div>
                 <div className="text-2xl font-bold text-white group-hover:text-emerald-400 transition-colors">{playerStats.homeostasis}%</div>
                 <div className="mt-3 h-1 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]"
                      style={{ width: `${playerStats.homeostasis}%` }}
                    />
                 </div>
               </div>
               <div className="p-5 rounded-2xl bg-zinc-900/40 border border-zinc-800/50 hover:border-zinc-700 transition-all group">
                 <div className="text-[9px] font-mono text-zinc-600 uppercase mb-2 tracking-widest">Plasticity</div>
                 <div className="text-2xl font-bold text-white group-hover:text-blue-400 transition-colors">{playerStats.morale}%</div>
                 <div className="mt-3 h-1 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.4)]"
                      style={{ width: `${playerStats.morale}%` }}
                    />
                 </div>
               </div>
            </div>
          </section>

          {/* Immunity Sweep */}
          <section className="space-y-4 flex-1 min-h-0">
            <div className="flex items-center justify-between px-2">
              <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-zinc-500 flex items-center gap-2">
                <Glyph label="I" size={14} className="text-blue-400" />
                Immunity Logs
                {notificationCount > 0 && (
                  <span className="bg-blue-500 text-black px-2 py-0.5 rounded-full text-[10px] ml-2 tracking-normal">
                    {notificationCount}
                  </span>
                )}
              </h3>
            </div>
            <div className="h-[456px]">
              <ImmunitySweep />
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
