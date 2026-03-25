'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import { useSurrogate, type Prediction } from '../../lib/surrogate';
import type { SimulationResult } from '../../components/ResultsPanel';
import { MODELS, MODEL_LABELS } from '../../lib/models';

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

// ---------------------------------------------------------------------------
// Stat card
// ---------------------------------------------------------------------------

function StatCard({
  label,
  mlValue,
  lbmValue,
}: {
  label: string;
  mlValue: number | null;
  lbmValue: number | null;
}) {
  const err =
    mlValue !== null && lbmValue !== null ? mlValue - lbmValue : null;
  const pctErr =
    err !== null && lbmValue !== null && Math.abs(lbmValue) > 1e-8
      ? (err / Math.abs(lbmValue)) * 100
      : null;

  return (
    <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
      <div className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-2">
        {label}
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-[10px] text-ctp-mauve uppercase mb-1">
            ML Surrogate
          </div>
          <div className="text-xl font-mono text-ctp-text">
            {mlValue !== null ? mlValue.toFixed(4) : '--'}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-ctp-blue uppercase mb-1">
            LBM Simulation
          </div>
          <div className="text-xl font-mono text-ctp-text">
            {lbmValue !== null ? lbmValue.toFixed(4) : '--'}
          </div>
        </div>
      </div>
      {err !== null && (
        <div className="mt-3 pt-3 border-t border-ctp-surface0 flex items-center gap-3 text-xs">
          <span className="text-ctp-overlay0">Error:</span>
          <span
            className={`font-mono ${Math.abs(err) < 0.01 ? 'text-ctp-green' : Math.abs(err) < 0.05 ? 'text-ctp-yellow' : 'text-ctp-red'}`}
          >
            {err >= 0 ? '+' : ''}
            {err.toFixed(6)}
          </span>
          {pctErr !== null && (
            <span className="text-ctp-overlay0 font-mono">
              ({pctErr >= 0 ? '+' : ''}
              {pctErr.toFixed(2)}%)
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Scatter chart: ML vs LBM for all historical results
// ---------------------------------------------------------------------------

function ComparisonChart({
  pairs,
  coeff,
}: {
  pairs: { ml: number; lbm: number; model: string }[];
  coeff: 'Cd' | 'Cl';
}) {
  if (pairs.length < 1) return null;

  const W = 300;
  const H = 300;
  const pad = { top: 12, right: 12, bottom: 28, left: 40 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const allVals = pairs.flatMap((p) => [p.ml, p.lbm]);
  const lo = Math.min(...allVals) * 0.95;
  const hi = Math.max(...allVals) * 1.05;
  const range = hi - lo || 1;

  const toX = (v: number) => pad.left + ((v - lo) / range) * plotW;
  const toY = (v: number) => pad.top + plotH - ((v - lo) / range) * plotH;

  const modelColors: Record<string, string> = {
    car: 'var(--color-ctp-red)',
    ahmed25: 'var(--color-ctp-blue)',
    ahmed35: 'var(--color-ctp-green)',
  };

  const ticks = [lo, lo + range * 0.5, hi];

  return (
    <div>
      <div className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-1">
        {coeff}: ML vs LBM
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxWidth: W }}>
        {/* Grid */}
        {ticks.map((v, i) => (
          <g key={i}>
            <line
              x1={pad.left}
              y1={toY(v)}
              x2={W - pad.right}
              y2={toY(v)}
              className="stroke-ctp-surface1"
              strokeWidth={0.5}
            />
            <text
              x={pad.left - 4}
              y={toY(v) + 3}
              textAnchor="end"
              className="fill-ctp-overlay0"
              fontSize={8}
            >
              {v.toFixed(3)}
            </text>
            <line
              x1={toX(v)}
              y1={pad.top}
              x2={toX(v)}
              y2={H - pad.bottom}
              className="stroke-ctp-surface1"
              strokeWidth={0.5}
            />
            <text
              x={toX(v)}
              y={H - pad.bottom + 12}
              textAnchor="middle"
              className="fill-ctp-overlay0"
              fontSize={8}
            >
              {v.toFixed(3)}
            </text>
          </g>
        ))}

        {/* Perfect agreement line */}
        <line
          x1={toX(lo)}
          y1={toY(lo)}
          x2={toX(hi)}
          y2={toY(hi)}
          className="stroke-ctp-overlay0"
          strokeWidth={0.8}
          strokeDasharray="4,3"
          opacity={0.5}
        />

        {/* Data points */}
        {pairs.map((p, i) => (
          <circle
            key={i}
            cx={toX(p.lbm)}
            cy={toY(p.ml)}
            r={4}
            fill={modelColors[p.model] ?? 'var(--color-ctp-text)'}
            opacity={0.8}
          />
        ))}

        {/* Axis labels */}
        <text
          x={pad.left + plotW / 2}
          y={H - 2}
          textAnchor="middle"
          className="fill-ctp-subtext0"
          fontSize={9}
        >
          LBM {coeff}
        </text>
        <text
          x={10}
          y={pad.top + plotH / 2}
          textAnchor="middle"
          className="fill-ctp-subtext0"
          fontSize={9}
          transform={`rotate(-90, 10, ${pad.top + plotH / 2})`}
        >
          ML {coeff}
        </text>
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sweep table: ML predictions across wind speeds for selected model
// ---------------------------------------------------------------------------

function SweepTable({
  model,
  reynolds,
  predict,
}: {
  model: string;
  reynolds: number;
  predict: (ws: number, re: number, m: string) => Prediction | null;
}) {
  const speeds = useMemo(() => {
    const out: number[] = [];
    for (let ws = 0.5; ws <= 5.0; ws += 0.25) {
      out.push(Math.round(ws * 100) / 100);
    }
    return out;
  }, []);

  const predictions = useMemo(
    () => speeds.map((ws) => ({ ws, pred: predict(ws, reynolds, model) })),
    [speeds, reynolds, model, predict],
  );

  return (
    <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
      <div className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-2">
        ML sweep: {MODEL_LABELS[model]} at Re={reynolds}
      </div>
      <div className="overflow-x-auto max-h-64 overflow-y-auto">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-ctp-mantle">
            <tr className="border-b border-ctp-surface1">
              <th className="py-1.5 pr-3 text-ctp-subtext0 font-medium">
                Wind
              </th>
              <th className="py-1.5 pr-3 text-ctp-subtext0 font-medium">
                C<sub>d</sub> (ML)
              </th>
              <th className="py-1.5 text-ctp-subtext0 font-medium">
                C<sub>l</sub> (ML)
              </th>
            </tr>
          </thead>
          <tbody>
            {predictions.map(({ ws, pred }) => (
              <tr
                key={ws}
                className="border-b border-ctp-surface0"
              >
                <td className="py-1 pr-3 font-mono text-ctp-subtext1">
                  {ws.toFixed(2)}
                </td>
                <td className="py-1 pr-3 font-mono text-ctp-mauve">
                  {pred ? pred.cd.toFixed(4) : '--'}
                </td>
                <td className="py-1 font-mono text-ctp-mauve">
                  {pred ? pred.cl.toFixed(4) : '--'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function ComparisonPage() {
  const { status: mlStatus, predict: mlPredict } = useSurrogate();

  const [model, setModel] = useState<string>('car');
  const [windSpeed, setWindSpeed] = useState(2.5);
  const [reynolds, setReynolds] = useState(5000);

  // Load LBM results from localStorage
  const lbmHistory: SimulationResult[] = useMemo(() => {
    if (typeof window === 'undefined') return [];
    try {
      const saved = localStorage.getItem('lattice_results');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  }, []);

  // ML prediction for current params
  const mlPred =
    mlStatus === 'ready' ? mlPredict(windSpeed, reynolds, model) : null;

  // Find closest LBM result for the selected params
  const matchingLbm = useMemo(() => {
    const candidates = lbmHistory.filter(
      (r) =>
        r.model === model &&
        r.cdValue !== null &&
        Math.abs(r.windSpeed - windSpeed) < 0.01,
    );
    return candidates.length > 0 ? candidates[candidates.length - 1] : null;
  }, [lbmHistory, model, windSpeed]);

  // Build paired data for scatter charts (all results that have both ML + LBM)
  const pairedData = useMemo(() => {
    if (mlStatus !== 'ready') return [];
    return lbmHistory
      .filter((r) => r.cdValue !== null && r.clValue !== null)
      .map((r) => {
        const pred = mlPredict(r.windSpeed, 0, r.model);
        if (!pred) return null;
        return {
          ml_cd: pred.cd,
          ml_cl: pred.cl,
          lbm_cd: r.cdValue!,
          lbm_cl: r.clValue!,
          model: r.model,
        };
      })
      .filter(Boolean) as {
      ml_cd: number;
      ml_cl: number;
      lbm_cd: number;
      lbm_cl: number;
      model: string;
    }[];
  }, [lbmHistory, mlStatus, mlPredict]);

  const cdPairs = pairedData.map((p) => ({
    ml: p.ml_cd,
    lbm: p.lbm_cd,
    model: p.model,
  }));
  const clPairs = pairedData.map((p) => ({
    ml: p.ml_cl,
    lbm: p.lbm_cl,
    model: p.model,
  }));

  return (
    <main className="min-h-screen p-4 md:p-6 lg:p-10 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-6 flex items-center gap-3">
        <Link
          href="/"
          className="text-ctp-overlay1 hover:text-ctp-text transition-colors"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15 19l-7-7 7-7"
            />
          </svg>
        </Link>
        <div>
          <h1 className="text-lg font-semibold text-ctp-text">
            ML vs LBM Comparison
          </h1>
          <p className="text-xs text-ctp-subtext0">
            Surrogate model predictions compared against Lattice Boltzmann results
          </p>
        </div>
      </div>

      {/* Model status */}
      {mlStatus !== 'ready' && (
        <div className="mb-4 p-3 bg-ctp-surface0 border border-ctp-surface1 rounded text-xs text-ctp-subtext0">
          {mlStatus === 'loading'
            ? 'Loading surrogate model weights...'
            : mlStatus === 'error'
              ? 'Failed to load surrogate model. Check that /models/model.bin exists.'
              : 'Initializing...'}
        </div>
      )}

      {/* Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
        <div className="border border-ctp-surface1 rounded-lg p-3 bg-ctp-mantle">
          <label className="text-[10px] text-ctp-overlay0 uppercase tracking-wider block mb-1">
            Model
          </label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full bg-ctp-surface0 border border-ctp-surface1 rounded px-2 py-1.5 text-sm text-ctp-text accent-ctp-mauve"
          >
            {MODELS.map((m) => (
              <option key={m} value={m}>
                {MODEL_LABELS[m]}
              </option>
            ))}
          </select>
        </div>
        <div className="border border-ctp-surface1 rounded-lg p-3 bg-ctp-mantle">
          <label className="text-[10px] text-ctp-overlay0 uppercase tracking-wider block mb-1">
            Wind Speed: {windSpeed.toFixed(2)}
          </label>
          <input
            type="range"
            min={0.5}
            max={5.0}
            step={0.25}
            value={windSpeed}
            onChange={(e) => setWindSpeed(parseFloat(e.target.value))}
            className="w-full accent-ctp-mauve"
          />
        </div>
        <div className="border border-ctp-surface1 rounded-lg p-3 bg-ctp-mantle">
          <label className="text-[10px] text-ctp-overlay0 uppercase tracking-wider block mb-1">
            Reynolds Number
          </label>
          <input
            type="number"
            min={0}
            max={100000}
            step={1000}
            value={reynolds}
            onChange={(e) =>
              setReynolds(clamp(parseInt(e.target.value) || 0, 0, 100000))
            }
            className="w-full bg-ctp-surface0 border border-ctp-surface1 rounded px-2 py-1.5 text-sm font-mono text-ctp-text"
          />
        </div>
      </div>

      {/* Side-by-side results */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <StatCard
          label="Drag Coefficient (Cd)"
          mlValue={mlPred?.cd ?? null}
          lbmValue={matchingLbm?.cdValue ?? null}
        />
        <StatCard
          label="Lift Coefficient (Cl)"
          mlValue={mlPred?.cl ?? null}
          lbmValue={matchingLbm?.clValue ?? null}
        />
      </div>

      {!matchingLbm && lbmHistory.length === 0 && (
        <div className="mb-6 p-3 bg-ctp-surface0 border border-ctp-surface1 rounded text-xs text-ctp-subtext0">
          No LBM results found. Run a simulation on the{' '}
          <Link href="/" className="text-ctp-blue hover:text-ctp-lavender">
            main page
          </Link>{' '}
          to see side-by-side comparisons.
        </div>
      )}

      {!matchingLbm && lbmHistory.length > 0 && (
        <div className="mb-6 p-3 bg-ctp-surface0 border border-ctp-surface1 rounded text-xs text-ctp-subtext0">
          No LBM result for {MODEL_LABELS[model]} at wind speed{' '}
          {windSpeed.toFixed(2)}. Adjust parameters to match a previous run, or
          run a new simulation.
        </div>
      )}

      {/* Charts + sweep */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        {/* Scatter charts from historical data */}
        {cdPairs.length > 0 && (
          <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
            <ComparisonChart pairs={cdPairs} coeff="Cd" />
            {/* Legend */}
            <div className="flex gap-3 mt-2 text-[10px] text-ctp-overlay0">
              <span className="flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full bg-ctp-red" />
                Car
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full bg-ctp-blue" />
                Ahmed 25
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full bg-ctp-green" />
                Ahmed 35
              </span>
            </div>
          </div>
        )}
        {clPairs.length > 0 && (
          <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
            <ComparisonChart pairs={clPairs} coeff="Cl" />
            <div className="flex gap-3 mt-2 text-[10px] text-ctp-overlay0">
              <span className="flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full bg-ctp-red" />
                Car
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full bg-ctp-blue" />
                Ahmed 25
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full bg-ctp-green" />
                Ahmed 35
              </span>
            </div>
          </div>
        )}

        {/* ML sweep table */}
        {mlStatus === 'ready' && (
          <div className={cdPairs.length > 0 ? '' : 'lg:col-span-3'}>
            <SweepTable
              model={model}
              reynolds={reynolds}
              predict={mlPredict}
            />
          </div>
        )}
      </div>

      {/* Historical results table */}
      {pairedData.length > 0 && (
        <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
          <div className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-2">
            All comparisons ({pairedData.length} runs)
          </div>
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
                    LBM C<sub>d</sub>
                  </th>
                  <th className="py-1.5 pr-2 text-ctp-subtext0 font-medium">
                    ML C<sub>d</sub>
                  </th>
                  <th className="py-1.5 pr-2 text-ctp-subtext0 font-medium">
                    C<sub>d</sub> Err
                  </th>
                  <th className="py-1.5 pr-2 text-ctp-subtext0 font-medium">
                    LBM C<sub>l</sub>
                  </th>
                  <th className="py-1.5 pr-2 text-ctp-subtext0 font-medium">
                    ML C<sub>l</sub>
                  </th>
                  <th className="py-1.5 text-ctp-subtext0 font-medium">
                    C<sub>l</sub> Err
                  </th>
                </tr>
              </thead>
              <tbody>
                {pairedData.map((p, i) => {
                  const cdErr = p.ml_cd - p.lbm_cd;
                  const clErr = p.ml_cl - p.lbm_cl;
                  return (
                    <tr
                      key={i}
                      className="border-b border-ctp-surface0"
                    >
                      <td className="py-1 pr-2 text-ctp-subtext1">
                        {MODEL_LABELS[p.model] ?? p.model}
                      </td>
                      <td className="py-1 pr-2 font-mono text-ctp-subtext1">
                        --
                      </td>
                      <td className="py-1 pr-2 font-mono text-ctp-blue">
                        {p.lbm_cd.toFixed(4)}
                      </td>
                      <td className="py-1 pr-2 font-mono text-ctp-mauve">
                        {p.ml_cd.toFixed(4)}
                      </td>
                      <td
                        className={`py-1 pr-2 font-mono ${Math.abs(cdErr) < 0.01 ? 'text-ctp-green' : 'text-ctp-yellow'}`}
                      >
                        {cdErr >= 0 ? '+' : ''}
                        {cdErr.toFixed(4)}
                      </td>
                      <td className="py-1 pr-2 font-mono text-ctp-blue">
                        {p.lbm_cl.toFixed(4)}
                      </td>
                      <td className="py-1 pr-2 font-mono text-ctp-mauve">
                        {p.ml_cl.toFixed(4)}
                      </td>
                      <td
                        className={`py-1 font-mono ${Math.abs(clErr) < 0.01 ? 'text-ctp-green' : 'text-ctp-yellow'}`}
                      >
                        {clErr >= 0 ? '+' : ''}
                        {clErr.toFixed(4)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </main>
  );
}
