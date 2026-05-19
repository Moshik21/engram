import { BrainMapPanel } from "../components/BrainMapPanel";
import { NeuralFieldControls } from "../components/graph/NeuralFieldControls";
import { useEngramStore } from "../store";

export function NeuralField() {
  const isConsolidating = useEngramStore(s => s.isRunning);

  return (
    <div className="relative h-full w-full bg-black overflow-hidden">
      <BrainMapPanel />
      <NeuralFieldControls />

      {/* Bio-Digital Status Indicators */}
      <div className="absolute bottom-6 left-6 z-10 flex flex-col gap-2 pointer-events-none">
        <div className="flex items-center gap-2 px-3 py-1.5 bg-zinc-900/50 backdrop-blur border border-zinc-800/50 rounded-full">
          <div className={`h-1.5 w-1.5 rounded-full ${isConsolidating ? 'bg-purple-500 animate-ping' : 'bg-zinc-700'}`} />
          <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">
            {isConsolidating ? 'Dream Phase Active' : 'Neural Stability: 100%'}
          </span>
        </div>
      </div>

      {/* Grid Pattern Overlay */}
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.4)_100%)]" />
      <div className="absolute inset-0 pointer-events-none opacity-[0.03]"
           style={{ backgroundImage: 'linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px)', backgroundSize: '100px 100px' }} />
    </div>
  );
}
