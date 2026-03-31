'use client';

import { useMemo } from 'react';
import { computeStrouhal, type SpectrumResult } from '../lib/spectral';
import { MODEL_LABELS } from '../lib/models';

export interface SimulationResult {
  videoUrl: string;
  cdValue: number | null;
  clValue: number | null;
  cdSeries: number[];
  clSeries: number[];
  charLength: number | null;
  sampleInterval: number | null;
  model: string;
  windSpeed: number;
  timestamp: number;
}


interface ResultsPanelProps {
  current: SimulationResult | null;
  history: SimulationResult[];
  onSelect: (result: SimulationResult) => void;
}

function downloadCsv(result: SimulationResult, strouhal: number | null) {
  const rows: string[] = [];
  if (strouhal !== null) {
    rows.push(`# Strouhal number: ${strouhal.toFixed(4)}`);
  }
  rows.push('step,cd,cl');
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

function CdChart({
  cdSeries,
  clSeries,
}: {
  cdSeries: number[];
  clSeries: number[];
}) {
  if (cdSeries.length < 2) return null;

  const W = 320;
  const H = 120;
  const pad = { top: 8, right: 8, bottom: 20, left: 36 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const allVals = [...cdSeries, ...clSeries].filter(
    (v) => Number.isFinite(v) && v >= 0
  );
  if (allVals.length === 0) return null;

  const yMax = Math.max(...allVals) * 1.1 || 1;
  const n = cdSeries.length;

  function toPath(series: number[]) {
    return series
      .map((v, i) => {
        const x = pad.left + (i / (n - 1)) * plotW;
        const y = pad.top + plotH - (Math.max(0, v) / yMax) * plotH;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }

  const yTicks = [0, yMax / 2, yMax];

  return (
    <div className="mb-3">
      <div className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-1">
        Convergence
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ maxWidth: W }}
      >
        {/* Grid lines */}
        {yTicks.map((v) => {
          const y = pad.top + plotH - (v / yMax) * plotH;
          return (
            <g key={v}>
              <line
                x1={pad.left}
                y1={y}
                x2={W - pad.right}
                y2={y}
                className="stroke-ctp-surface1"
                strokeWidth={0.5}
              />
              <text
                x={pad.left - 4}
                y={y + 3}
                textAnchor="end"
                className="fill-ctp-overlay0"
                fontSize={8}
              >
                {v < 10 ? v.toFixed(2) : v.toFixed(1)}
              </text>
            </g>
          );
        })}

        {/* Axes */}
        <line
          x1={pad.left}
          y1={pad.top}
          x2={pad.left}
          y2={H - pad.bottom}
          className="stroke-ctp-surface1"
          strokeWidth={1}
        />
        <line
          x1={pad.left}
          y1={H - pad.bottom}
          x2={W - pad.right}
          y2={H - pad.bottom}
          className="stroke-ctp-surface1"
          strokeWidth={1}
        />

        {/* Cd line */}
        <path
          d={toPath(cdSeries)}
          fill="none"
          className="stroke-ctp-blue"
          strokeWidth={1.5}
        />

        {/* Cl line */}
        {clSeries.length >= 2 && (
          <path
            d={toPath(clSeries)}
            fill="none"
            className="stroke-ctp-mauve"
            strokeWidth={1.5}
            strokeDasharray="3,2"
          />
        )}

        {/* X axis label */}
        <text
          x={pad.left + plotW / 2}
          y={H - 2}
          textAnchor="middle"
          className="fill-ctp-overlay0"
          fontSize={8}
        >
          Measurement step
        </text>

        {/* Legend */}
        <line
          x1={W - pad.right - 60}
          y1={pad.top + 4}
          x2={W - pad.right - 48}
          y2={pad.top + 4}
          className="stroke-ctp-blue"
          strokeWidth={1.5}
        />
        <text
          x={W - pad.right - 45}
          y={pad.top + 7}
          className="fill-ctp-overlay1"
          fontSize={8}
        >
          Cd
        </text>
        <line
          x1={W - pad.right - 30}
          y1={pad.top + 4}
          x2={W - pad.right - 18}
          y2={pad.top + 4}
          className="stroke-ctp-mauve"
          strokeWidth={1.5}
          strokeDasharray="3,2"
        />
        <text
          x={W - pad.right - 15}
          y={pad.top + 7}
          className="fill-ctp-overlay1"
          fontSize={8}
        >
          Cl
        </text>
      </svg>
    </div>
  );
}

function SpectrumChart({ spectrum }: { spectrum: SpectrumResult }) {
  const W = 320;
  const H = 120;
  const pad = { top: 8, right: 8, bottom: 20, left: 36 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const { frequencies, power, peakIndex } = spectrum;
  if (frequencies.length < 2) return null;

  const pMax = Math.max(...power) * 1.1 || 1;

  const path = power
    .map((p, i) => {
      const x = pad.left + (i / (power.length - 1)) * plotW;
      const y = pad.top + plotH - (p / pMax) * plotH;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  const peakX = pad.left + (peakIndex / (power.length - 1)) * plotW;
  const peakY = pad.top + plotH - (power[peakIndex] / pMax) * plotH;
  const peakFreqLabel = frequencies[peakIndex].toExponential(2);

  return (
    <div className="mb-3">
      <div className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-1">
        Cl Power Spectrum
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ maxWidth: W }}
      >
        {/* Axes */}
        <line
          x1={pad.left}
          y1={pad.top}
          x2={pad.left}
          y2={H - pad.bottom}
          className="stroke-ctp-surface1"
          strokeWidth={1}
        />
        <line
          x1={pad.left}
          y1={H - pad.bottom}
          x2={W - pad.right}
          y2={H - pad.bottom}
          className="stroke-ctp-surface1"
          strokeWidth={1}
        />

        {/* Spectrum line */}
        <path
          d={path}
          fill="none"
          className="stroke-ctp-peach"
          strokeWidth={1.5}
        />

        {/* Peak marker */}
        <circle
          cx={peakX}
          cy={peakY}
          r={3}
          className="fill-ctp-red"
        />
        <text
          x={peakX + 5}
          y={peakY - 4}
          className="fill-ctp-red"
          fontSize={7}
        >
          f={peakFreqLabel}
        </text>

        {/* X axis label */}
        <text
          x={pad.left + plotW / 2}
          y={H - 2}
          textAnchor="middle"
          className="fill-ctp-overlay0"
          fontSize={8}
        >
          Frequency (1/dt)
        </text>
      </svg>
    </div>
  );
}

function useStrouhal(result: SimulationResult | null): SpectrumResult | null {
  return useMemo(() => {
    if (!result || result.clSeries.length < 8 || !result.charLength) return null;
    return computeStrouhal(result.clSeries, result.charLength, result.sampleInterval ?? undefined);
  }, [result]);
}

export default function ResultsPanel({
  current,
  history,
  onSelect,
}: ResultsPanelProps) {
  const spectrum = useStrouhal(current);

  if (history.length === 0) return null;

  return (
    <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
      <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-3">
        Results
      </h2>

      {/* Current result summary */}
      {current && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 text-xs mb-3">
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
          <div className="border border-ctp-surface1 rounded p-3 min-w-0">
            <div className="text-ctp-overlay1 mb-1">Strouhal Number</div>
            <div className="text-lg font-mono text-ctp-text truncate">
              {spectrum ? spectrum.strouhal.toFixed(4) : '--'}
            </div>
            <div className="text-ctp-overlay0 mt-0.5">
              St = fL/U
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

      {current && current.cdSeries.length >= 2 && (
        <CdChart cdSeries={current.cdSeries} clSeries={current.clSeries} />
      )}

      {spectrum && <SpectrumChart spectrum={spectrum} />}

      {current && current.cdSeries.length > 0 && (
        <button
          onClick={() => downloadCsv(current, spectrum?.strouhal ?? null)}
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
                  <th className="py-1.5 pr-2 text-ctp-subtext0 font-medium">
                    St
                  </th>
                  <th className="py-1.5 text-ctp-subtext0 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {[...history].reverse().map((result) => {
                  const isCurrent = result === current;
                  const rowSt = result.charLength && result.clSeries.length >= 8
                    ? computeStrouhal(result.clSeries, result.charLength, result.sampleInterval ?? undefined)
                    : null;
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
                      <td className="py-1.5 pr-2 font-mono text-ctp-text">
                        {rowSt ? rowSt.strouhal.toFixed(4) : '--'}
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
