import { useEngramStore } from '../../store';
import { Glyph } from '../nerve/Glyph';
import type { NeuralFieldLayer } from '../../store/types';

export function NeuralFieldControls() {
  const selectedLayer = useEngramStore(s => s.selectedNeuralLayer);
  const setSelectedLayer = useEngramStore(s => s.setSelectedNeuralLayer);
  const level = useEngramStore(s => s.cerebralStats.level);

  const layers: { id: NeuralFieldLayer; label: string; glyph: string; minLevel: number }[] = [
    { id: 'activity', label: 'Activity', glyph: 'A', minLevel: 1 },
    { id: 'clusters', label: 'Semantic Clusters', glyph: 'C', minLevel: 5 },
    { id: 'heatmap', label: 'Activation Heatmap', glyph: 'H', minLevel: 10 },
    { id: 'entropy', label: 'Entropy Field', glyph: 'E', minLevel: 20 },
  ];

  return (
    <div className="absolute top-6 left-6 z-10 flex flex-col gap-3">
      <div className="flex items-center gap-3 px-4 py-3 bg-zinc-900/80 backdrop-blur-xl border border-zinc-800/50 rounded-2xl shadow-2xl">
        <Glyph label="L" className="text-zinc-500" size={18} />
        <div className="h-4 w-px bg-zinc-800" />
        <div className="flex gap-1">
          {layers.map(layer => {
            const isLocked = level < layer.minLevel;
            const isActive = selectedLayer === layer.id;

            return (
              <button
                key={layer.id}
                disabled={isLocked}
                onClick={() => setSelectedLayer(layer.id)}
                className={`
                  relative px-3 py-1.5 rounded-xl flex items-center gap-2 transition-all group
                  ${isActive ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' : 'text-zinc-500 hover:text-zinc-300'}
                  ${isLocked ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'}
                `}
              >
                <Glyph label={layer.glyph} size={14} className={isActive ? 'animate-pulse' : ''} />
                <span className="text-[11px] font-bold uppercase tracking-wider">{layer.label}</span>

                {isLocked && (
                  <div className="absolute -top-1 -right-1 bg-zinc-800 text-[8px] px-1 rounded border border-zinc-700">
                    LVL {layer.minLevel}
                  </div>
                )}

                {!isLocked && !isActive && (
                  <div className="absolute inset-0 bg-white/5 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity" />
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
