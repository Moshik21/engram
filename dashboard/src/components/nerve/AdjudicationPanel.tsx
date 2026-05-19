import React, { useEffect, useState } from 'react';
import { useEngramStore } from '../../store';
import { Glyph } from './Glyph';

export const AdjudicationPanel: React.FC = () => {
  const {
    adjudicationRequests,
    loadAdjudications,
    resolveAdjudication,
    isAdjudicating
  } = useEngramStore();

  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedEvidenceIds, setSelectedEvidenceIds] = useState<string[]>([]);

  useEffect(() => {
    loadAdjudications();
  }, [loadAdjudications]);

  if (adjudicationRequests.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-zinc-500">
        <div className="relative mb-6">
          <div className="absolute inset-0 bg-amber-500/10 blur-3xl rounded-full" />
          <Glyph label="S" size={80} className="relative opacity-20 text-amber-500" />
        </div>
        <h3 className="text-2xl font-bold text-zinc-300">Neural Harmony Maintained</h3>
        <p className="mt-2 text-zinc-500 max-w-xs text-center">
          No ambiguous signals currently require manual adjudication. The system is operating at peak homeostasis.
        </p>
      </div>
    );
  }

  const current = adjudicationRequests[currentIndex];

  const handleResolve = () => {
    if (!current) return;

    const rejectIds = current.candidate_evidence
      .map(e => e.evidence_id)
      .filter(id => !selectedEvidenceIds.includes(id));

    const entities = current.candidate_evidence
      .filter(e => e.fact_class === 'entity' && selectedEvidenceIds.includes(e.evidence_id))
      .map(e => ({
        name: e.payload.name,
        entity_type: e.payload.entity_type,
        attributes: e.payload.attributes
      }));

    const relationships = current.candidate_evidence
      .filter(e => e.fact_class === 'relationship' && selectedEvidenceIds.includes(e.evidence_id))
      .map(e => ({
        source: e.payload.source,
        target: e.payload.target,
        predicate: e.payload.predicate
      }));

    resolveAdjudication({
      request_id: current.request_id,
      entities,
      relationships,
      reject_evidence_ids: rejectIds,
      rationale: "Adjudicated via Nerve Center Adjudication",
      model_tier: "default"
    });

    // Clear selection for next one
    setSelectedEvidenceIds([]);
    if (currentIndex >= adjudicationRequests.length - 1 && currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  return (
    <div className="flex flex-col h-full bg-black/40 backdrop-blur-md rounded-2xl border border-zinc-800/50 overflow-hidden shadow-2xl">
      <header className="flex justify-between items-center p-6 border-b border-zinc-800/50 bg-zinc-900/20">
        <div>
          <h2 className="text-xl font-bold text-zinc-100 flex items-center gap-3">
            <div className="p-2 bg-amber-500/10 rounded-lg">
              <Glyph label="A" size={20} className="text-amber-500" />
            </div>
            Neural Adjudication
          </h2>
          <p className="text-zinc-500 text-xs mt-1">Resolve competing interpretations of neural signals.</p>
        </div>
        <div className="flex items-center gap-4">
           <div className="px-3 py-1 bg-zinc-800/50 rounded-full border border-zinc-700/50">
             <span className="text-[10px] font-mono text-zinc-400 uppercase tracking-widest">
              Conflict {currentIndex + 1} / {adjudicationRequests.length}
             </span>
           </div>
           <button
             onClick={() => loadAdjudications()}
             className="p-2 text-zinc-500 hover:text-amber-500 transition-colors hover:bg-amber-500/5 rounded-lg"
           >
             <Glyph label="R" size={18} className={isAdjudicating ? 'animate-spin' : ''} />
           </button>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-0 overflow-hidden">
        {/* Left Side: Context */}
        <div className="flex flex-col gap-6 p-8 overflow-y-auto border-r border-zinc-800/50 bg-zinc-950/20">
          <section className="space-y-4">
            <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
              <Glyph label="!" size={12} className="text-amber-500" />
              Ambiguous Signal
            </h4>
            <div className="relative group">
              <div className="absolute -inset-1 bg-gradient-to-r from-amber-500/20 to-transparent blur opacity-0 group-hover:opacity-100 transition duration-1000" />
              <div className="relative text-lg text-zinc-200 leading-relaxed font-medium bg-zinc-900/80 p-6 rounded-xl border border-zinc-800 shadow-inner">
                <span className="text-amber-500/40 text-4xl font-serif absolute -top-2 -left-1">"</span>
                {current.selected_text}
                <span className="text-amber-500/40 text-4xl font-serif absolute -bottom-6 -right-1">"</span>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              {current.ambiguity_tags.map(tag => (
                <span key={tag} className="px-2 py-1 bg-amber-500/5 text-amber-500/70 border border-amber-500/10 rounded text-[9px] font-mono uppercase tracking-wider">
                  {tag}
                </span>
              ))}
            </div>
          </section>

          <section className="p-5 bg-blue-500/5 border border-blue-500/10 rounded-xl space-y-2">
            <h4 className="text-[10px] font-bold text-blue-400 uppercase tracking-widest">Cerebral Directive</h4>
            <p className="text-xs text-zinc-400 leading-relaxed italic">
              "{current.instructions}"
            </p>
          </section>

          <div className="flex gap-3 mt-auto">
            <button
              disabled={currentIndex === 0}
              onClick={() => { setCurrentIndex(i => i - 1); setSelectedEvidenceIds([]); }}
              className="flex-1 py-2 bg-zinc-900/50 border border-zinc-800 rounded-lg text-xs font-medium text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all disabled:opacity-20"
            >
              Previous Signal
            </button>
            <button
              disabled={currentIndex === adjudicationRequests.length - 1}
              onClick={() => { setCurrentIndex(i => i + 1); setSelectedEvidenceIds([]); }}
              className="flex-1 py-2 bg-zinc-900/50 border border-zinc-800 rounded-lg text-xs font-medium text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all disabled:opacity-20"
            >
              Next Signal
            </button>
          </div>
        </div>

        {/* Right Side: Contenders */}
        <div className="flex flex-col p-8 overflow-y-auto bg-zinc-900/10">
          <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-6">Contending Interpretations</h4>

          <div className="space-y-4">
            {current.candidate_evidence.map((evidence) => {
              const isSelected = selectedEvidenceIds.includes(evidence.evidence_id);
              return (
                <div
                  key={evidence.evidence_id}
                  onClick={() => setSelectedEvidenceIds(prev =>
                    prev.includes(evidence.evidence_id)
                      ? prev.filter(id => id !== evidence.evidence_id)
                      : [...prev, evidence.evidence_id]
                  )}
                  className={`
                    relative cursor-pointer group p-5 rounded-2xl border transition-all duration-300 transform active:scale-[0.98]
                    ${isSelected
                      ? 'bg-amber-500/5 border-amber-500/40 shadow-[0_0_25px_-12px_rgba(245,158,11,0.4)]'
                      : 'bg-zinc-900/40 border-zinc-800/50 hover:border-zinc-700 hover:bg-zinc-900/60'}
                  `}
                >
                  <div className="flex justify-between items-start mb-4">
                    <span className="text-[9px] font-mono text-zinc-500 uppercase tracking-widest bg-black/40 px-2 py-0.5 rounded border border-zinc-800/50">
                      {evidence.fact_class}
                    </span>
                    <div className={`
                      w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all duration-300
                      ${isSelected
                        ? 'bg-amber-500 border-amber-500 text-black rotate-0 scale-100'
                        : 'border-zinc-700 text-transparent scale-90 -rotate-12 group-hover:border-zinc-500'}
                    `}>
                      <Glyph label="OK" size={14} />
                    </div>
                  </div>

                  <div className="space-y-3">
                    {evidence.fact_class === 'entity' && (
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-zinc-800 to-zinc-900 border border-zinc-700/50 flex items-center justify-center text-zinc-500 group-hover:text-amber-500 transition-colors shadow-lg">
                          <Glyph label="E" size={20} />
                        </div>
                        <div>
                          <div className="text-sm font-bold text-zinc-100 group-hover:text-white transition-colors">{evidence.payload.name}</div>
                          <div className="text-[10px] text-zinc-500 font-mono mt-0.5">{evidence.payload.entity_type}</div>
                        </div>
                      </div>
                    )}
                    {evidence.fact_class === 'relationship' && (
                      <div className="flex flex-col gap-2 p-3 bg-black/30 rounded-xl border border-zinc-800/30">
                        <div className="text-[11px] text-zinc-400 flex items-center justify-between">
                          <span className="text-zinc-600 text-[9px] uppercase tracking-tighter">Subject</span>
                          <span className="font-bold text-zinc-200">{evidence.payload.source}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="h-[1px] flex-1 bg-gradient-to-r from-transparent via-amber-500/30 to-transparent" />
                          <div className="text-[9px] text-amber-500/80 font-mono px-2 py-0.5 bg-amber-500/5 border border-amber-500/10 rounded">
                             {evidence.payload.predicate}
                          </div>
                          <div className="h-[1px] flex-1 bg-gradient-to-r from-transparent via-amber-500/30 to-transparent" />
                        </div>
                        <div className="text-[11px] text-zinc-400 flex items-center justify-between">
                          <span className="text-zinc-600 text-[9px] uppercase tracking-tighter">Object</span>
                          <span className="font-bold text-zinc-200">{evidence.payload.target}</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-auto pt-8 flex gap-4">
            <button
              onClick={() => { setSelectedEvidenceIds([]); handleResolve(); }}
              className="px-6 py-4 bg-zinc-900 border border-zinc-800 text-zinc-500 rounded-xl font-bold hover:bg-red-950/20 hover:text-red-500 hover:border-red-900/30 transition-all flex items-center justify-center gap-2"
            >
              <Glyph label="X" size={18} />
            </button>
            <button
              disabled={selectedEvidenceIds.length === 0 || isAdjudicating}
              onClick={handleResolve}
              className={`
                flex-1 py-4 rounded-xl font-bold transition-all flex items-center justify-center gap-3 overflow-hidden relative group
                ${selectedEvidenceIds.length > 0
                  ? 'bg-amber-500 text-black hover:bg-amber-400 shadow-[0_0_40px_-10px_rgba(245,158,11,0.5)] active:translate-y-0.5'
                  : 'bg-zinc-800 text-zinc-600 cursor-not-allowed'}
              `}
            >
              {isAdjudicating ? (
                <Glyph label="R" size={20} className="animate-spin" />
              ) : (
                <>
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]" />
                  <Glyph label="A" size={20} className="group-hover:rotate-12 transition-transform" />
                  <span className="tracking-tight">Authorize Synthesis</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
