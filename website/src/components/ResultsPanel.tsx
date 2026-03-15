'use client';

export interface SimulationResult {
  videoUrl: string;
  cdValue: number | null;
  clValue: number | null;
  cdSeries: number[];
  clSeries: number[];
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

function downloadCsv(result: SimulationResult) {
  const rows = ['step,cd,cl'];
  const len = Math.max(result.cdSeries.length, result.clSeries.length);
  for (let i = 0; i < len; i++) {
    const cd = i < result.cdSeries.length ? result.cdSeries[i].toFixed(6) : '';
    const cl = i < result.clSeries.length ? result.clSeries[i].toFixed(6) : '';
    rows.push(`${i},${cd},${cl}`);
  }
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `lattice_${result.model}_w${result.windSpeed.toFixed(1)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ResultsPanel({
  current,
  history,
  onSelect,
}: ResultsPanelProps) {
  if (history.length === 0) return null;

  return (
    <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
      <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-3">
        Results
      </h2>

      {/* Current result summary */}
      {current && (
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-2 text-xs mb-3">
          <div className="border border-ctp-surface1 rounded p-3 min-w-0">
            <div className="text-ctp-overlay1 mb-1">Drag Coefficient</div>
            <div className="text-lg font-mono text-ctp-text truncate">
              {current.cdValue !== null ? current.cdValue.toFixed(4) : '--'}
            </div>
            <div className="text-ctp-overlay0 mt-0.5">
              C<sub>d</sub>
            </div>
          </div>
          <div className="border border-ctp-surface1 rounded p-3 min-w-0">
            <div className="text-ctp-overlay1 mb-1">Lift Coefficient</div>
            <div className="text-lg font-mono text-ctp-text truncate">
              {current.clValue !== null ? current.clValue.toFixed(4) : '--'}
            </div>
            <div className="text-ctp-overlay0 mt-0.5">
              C<sub>l</sub>
            </div>
          </div>
          <div className="border border-ctp-surface1 rounded p-3 col-span-2 lg:col-span-1">
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

      {current && current.cdSeries.length > 0 && (
        <button
          onClick={() => downloadCsv(current)}
          className="mb-3 bg-ctp-surface0 hover:bg-ctp-surface1 text-ctp-text text-xs font-medium py-1.5 px-3 rounded border border-ctp-surface1 transition-colors"
        >
          Export CSV
        </button>
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
                  <th className="py-1.5 pr-2 text-ctp-subtext0 font-medium">
                    Model
                  </th>
                  <th className="py-1.5 pr-2 text-ctp-subtext0 font-medium">
                    Wind
                  </th>
                  <th className="py-1.5 pr-2 text-ctp-subtext0 font-medium">
                    C<sub>d</sub>
                  </th>
                  <th className="py-1.5 pr-2 text-ctp-subtext0 font-medium">
                    C<sub>l</sub>
                  </th>
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
                        {result.cdValue !== null
                          ? result.cdValue.toFixed(4)
                          : '--'}
                      </td>
                      <td className="py-1.5 pr-2 font-mono text-ctp-text">
                        {result.clValue !== null
                          ? result.clValue.toFixed(4)
                          : '--'}
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
