'use client';

import { useId, useMemo, useRef } from 'react';
import {
  compareMetrics,
  compareParams,
  comparisonCsv,
  runLabel,
  type MetricRow,
} from '../lib/compare';
import { downloadText } from '../lib/download';
import type { SimulationResult } from '../lib/types';

interface RunComparisonProps {
  a: SimulationResult;
  b: SimulationResult;
  onClose: () => void;
}

const SIDES = [
  { tag: 'A', text: 'text-ctp-blue' },
  { tag: 'B', text: 'text-ctp-peach' },
] as const;

function fmt(v: number | null, digits: number): string {
  return v === null ? '--' : v.toFixed(digits);
}

function fmtDiff(row: MetricRow): string {
  if (row.diff === null) return '--';
  const sign = row.diff > 0 ? '+' : '';
  const pct =
    row.pct === null
      ? ''
      : ` (${row.pct > 0 ? '+' : ''}${row.pct.toFixed(1)}%)`;
  return `${sign}${row.diff.toFixed(row.digits)}${pct}`;
}

function OverlayChart({ a, b }: { a: number[]; b: number[] }) {
  const W = 320;
  const H = 120;
  const pad = { top: 8, right: 8, bottom: 20, left: 36 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const vals = [...a, ...b].filter((v) => Number.isFinite(v) && v >= 0);
  const n = Math.max(a.length, b.length);
  if (vals.length === 0 || n < 2) return null;
  const yMax = Math.max(...vals) * 1.1 || 1;

  const toPath = (series: number[]) =>
    series
      .map((v, i) => {
        const x = pad.left + (i / (n - 1)) * plotW;
        const y = pad.top + plotH - (Math.max(0, v) / yMax) * plotH;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

  const last = (s: number[]) =>
    s.length > 0 ? (s[s.length - 1] ?? 0).toFixed(3) : 'none';
  const label = `Cd over ${n} measurement steps. Run A ends at ${last(a)}, run B ends at ${last(b)}.`;

  return (
    <div>
      <div className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-1">
        Cd convergence
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ maxWidth: W }}
        role="img"
        aria-label={label}
      >
        {[0, yMax / 2, yMax].map((v) => {
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
        {a.length >= 2 && (
          <path
            d={toPath(a)}
            fill="none"
            className="stroke-ctp-blue"
            strokeWidth={1.5}
          />
        )}
        {b.length >= 2 && (
          <path
            d={toPath(b)}
            fill="none"
            className="stroke-ctp-peach"
            strokeWidth={1.5}
            strokeDasharray="3,2"
          />
        )}
        <text
          x={pad.left + plotW / 2}
          y={H - 2}
          textAnchor="middle"
          className="fill-ctp-overlay0"
          fontSize={8}
        >
          Measurement step
        </text>
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
          A
        </text>
        <line
          x1={W - pad.right - 30}
          y1={pad.top + 4}
          x2={W - pad.right - 18}
          y2={pad.top + 4}
          className="stroke-ctp-peach"
          strokeWidth={1.5}
          strokeDasharray="3,2"
        />
        <text
          x={W - pad.right - 15}
          y={pad.top + 7}
          className="fill-ctp-overlay1"
          fontSize={8}
        >
          B
        </text>
      </svg>
    </div>
  );
}

export default function RunComparison({ a, b, onClose }: RunComparisonProps) {
  const headingId = useId();
  const videoA = useRef<HTMLVideoElement>(null);
  const videoB = useRef<HTMLVideoElement>(null);
  const metrics = useMemo(() => compareMetrics(a, b), [a, b]);
  const params = useMemo(() => compareParams(a, b), [a, b]);
  const runs = [a, b] as const;
  const refs = [videoA, videoB] as const;

  const playBoth = () => {
    for (const ref of refs) {
      const v = ref.current;
      if (!v) continue;
      v.currentTime = 0;
      v.play().catch(() => {});
    }
  };

  const exportCsv = () => {
    downloadText(
      comparisonCsv(a, b),
      `lattice_compare_${a.model}_w${a.windSpeed.toFixed(1)}_vs_${b.model}_w${b.windSpeed.toFixed(1)}.csv`,
      'text/csv',
    );
  };

  return (
    <section
      aria-labelledby={headingId}
      data-testid="run-comparison"
      className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle"
    >
      <div className="flex items-center justify-between mb-3">
        <h2
          id={headingId}
          className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider"
        >
          Compare Runs
        </h2>
        <button
          onClick={onClose}
          aria-label="Close comparison"
          className="text-xs text-ctp-blue hover:text-ctp-lavender transition-colors"
        >
          Close
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
        {runs.map((run, i) => (
          <figure key={SIDES[i].tag} className="min-w-0">
            <figcaption className="text-xs text-ctp-subtext1 mb-1 truncate">
              <span className={`font-mono font-semibold ${SIDES[i].text}`}>
                {SIDES[i].tag}
              </span>{' '}
              {runLabel(run)}
            </figcaption>
            <div className="aspect-video bg-ctp-crust rounded border border-ctp-surface1 overflow-hidden flex items-center justify-center">
              {run.videoUrl ? (
                <video
                  ref={refs[i]}
                  src={run.videoUrl}
                  controls
                  muted
                  loop
                  playsInline
                  preload="metadata"
                  aria-label={`Video of run ${SIDES[i].tag}`}
                  className="w-full h-full object-contain"
                />
              ) : (
                <p className="text-xs text-ctp-overlay0">
                  No video for this run
                </p>
              )}
            </div>
          </figure>
        ))}
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        <button
          onClick={playBoth}
          className="bg-ctp-surface0 hover:bg-ctp-surface1 text-ctp-text text-xs font-medium py-1.5 px-3 rounded border border-ctp-surface1 transition-colors"
        >
          Play both from start
        </button>
        <button
          onClick={exportCsv}
          className="bg-ctp-surface0 hover:bg-ctp-surface1 text-ctp-text text-xs font-medium py-1.5 px-3 rounded border border-ctp-surface1 transition-colors"
        >
          Export CSV
        </button>
      </div>

      <div className="overflow-x-auto mb-4">
        <table className="w-full text-left text-xs">
          <caption className="sr-only">Results of run A and run B</caption>
          <thead>
            <tr className="border-b border-ctp-surface1">
              <th
                scope="col"
                className="py-1.5 pr-2 text-ctp-subtext0 font-medium"
              >
                Result
              </th>
              <th scope="col" className="py-1.5 pr-2 text-ctp-blue font-medium">
                A
              </th>
              <th
                scope="col"
                className="py-1.5 pr-2 text-ctp-peach font-medium"
              >
                B
              </th>
              <th scope="col" className="py-1.5 text-ctp-subtext0 font-medium">
                B - A
              </th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m) => (
              <tr key={m.key} className="border-b border-ctp-surface0">
                <th
                  scope="row"
                  className="py-1.5 pr-2 text-ctp-subtext1 font-normal"
                >
                  {m.label}
                </th>
                <td className="py-1.5 pr-2 font-mono text-ctp-text">
                  {fmt(m.a, m.digits)}
                </td>
                <td className="py-1.5 pr-2 font-mono text-ctp-text">
                  {fmt(m.b, m.digits)}
                </td>
                <td className="py-1.5 font-mono text-ctp-subtext1">
                  {fmtDiff(m)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="overflow-x-auto mb-4">
        <table className="w-full text-left text-xs">
          <caption className="sr-only">Parameters of run A and run B</caption>
          <thead>
            <tr className="border-b border-ctp-surface1">
              <th
                scope="col"
                className="py-1.5 pr-2 text-ctp-subtext0 font-medium"
              >
                Parameter
              </th>
              <th scope="col" className="py-1.5 pr-2 text-ctp-blue font-medium">
                A
              </th>
              <th scope="col" className="py-1.5 text-ctp-peach font-medium">
                B
              </th>
            </tr>
          </thead>
          <tbody>
            {params.map((p) => (
              <tr
                key={p.key}
                className={`border-b border-ctp-surface0 ${p.differs ? 'bg-ctp-surface0/50' : ''}`}
              >
                <th
                  scope="row"
                  className="py-1.5 pr-2 text-ctp-subtext1 font-normal"
                >
                  {p.label}
                  {p.differs && <span className="sr-only"> (differs)</span>}
                </th>
                <td
                  className={`py-1.5 pr-2 ${p.differs ? 'text-ctp-yellow' : 'text-ctp-text'}`}
                >
                  {p.a}
                </td>
                <td
                  className={`py-1.5 ${p.differs ? 'text-ctp-yellow' : 'text-ctp-text'}`}
                >
                  {p.b}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <OverlayChart a={a.cdSeries} b={b.cdSeries} />
    </section>
  );
}
