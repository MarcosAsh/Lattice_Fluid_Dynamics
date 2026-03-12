'use client';

export interface SimulationResult {
  videoUrl: string;
  cdValue: number | null;
  clValue: number | null;
  model: string;
  windSpeed: number;
  timestamp: number;
}

const MODEL_LABELS: Record<string, string> = {
  car: 'Car',
  ahmed25: 'Ahmed 25',
  ahmed35: 'Ahmed 35',
  custom: 'Custom',
};

interface ResultsPanelProps {
  current: SimulationResult | null;
  history: SimulationResult[];
  onSelect: (result: SimulationResult) => void;
}

export default function ResultsPanel({ current, history, onSelect }: ResultsPanelProps) {
  if (history.length === 0) return null;

  return (
    <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
      <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-3">
        Results
      </h2>

      {/* Current result summary */}
      {current && (
        <div className="grid grid-cols-3 gap-2 text-xs mb-3">
          <div className="border border-ctp-surface1 rounded p-3">
            <div className="text-ctp-overlay1 mb-1">Drag Coefficient</div>
            <div className="text-2xl font-mono text-ctp-text">
              {current.cdValue !== null ? current.cdValue.toFixed(4) : '--'}
            </div>
            <div className="text-ctp-overlay0 mt-0.5">
              C<sub>d</sub>
            </div>
          </div>
          <div className="border border-ctp-surface1 rounded p-3">
            <div className="text-ctp-overlay1 mb-1">Lift Coefficient</div>
            <div className="text-2xl font-mono text-ctp-text">
              {current.clValue !== null ? current.clValue.toFixed(4) : '--'}
            </div>
            <div className="text-ctp-overlay0 mt-0.5">
              C<sub>l</sub>
            </div>
          </div>
          <div className="border border-ctp-surface1 rounded p-3">
            <div className="text-ctp-overlay1 mb-1">Configuration</div>
            <div className="text-sm text-ctp-text font-medium">
              {MODEL_LABELS[current.model] ?? current.model}
            </div>
            <div className="text-ctp-overlay0 mt-0.5">
              Wind: {current.windSpeed.toFixed(1)}
            </div>
          </div>
        </div>
      )}

      {/* Comparison table */}
      {history.length > 1 && (
        <>
          <hr className="border-ctp-surface1 mb-3" />
          <h3 className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-2">
            All Runs
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-ctp-surface1">
                  <th className="py-1.5 pr-2 text-ctp-subtext0 font-medium">Model</th>
                  <th className="py-1.5 pr-2 text-ctp-subtext0 font-medium">Wind</th>
                  <th className="py-1.5 pr-2 text-ctp-subtext0 font-medium">C<sub>d</sub></th>
                  <th className="py-1.5 pr-2 text-ctp-subtext0 font-medium">C<sub>l</sub></th>
                  <th className="py-1.5 text-ctp-subtext0 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {[...history].reverse().map((result) => {
                  const isCurrent = result === current;
                  return (
                    <tr
                      key={result.timestamp}
                      className={`border-b border-ctp-surface0 ${isCurrent ? 'bg-ctp-surface0/50' : ''}`}
                    >
                      <td className="py-1.5 pr-2 text-ctp-subtext1">
                        {MODEL_LABELS[result.model] ?? result.model}
                      </td>
                      <td className="py-1.5 pr-2 font-mono text-ctp-subtext1">
                        {result.windSpeed.toFixed(1)}
                      </td>
                      <td className="py-1.5 pr-2 font-mono text-ctp-text">
                        {result.cdValue !== null ? result.cdValue.toFixed(4) : '--'}
                      </td>
                      <td className="py-1.5 pr-2 font-mono text-ctp-text">
                        {result.clValue !== null ? result.clValue.toFixed(4) : '--'}
                      </td>
                      <td className="py-1.5">
                        {!isCurrent && (
                          <button
                            onClick={() => onSelect(result)}
                            className="text-ctp-blue hover:text-ctp-lavender transition-colors"
                          >
                            View
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
