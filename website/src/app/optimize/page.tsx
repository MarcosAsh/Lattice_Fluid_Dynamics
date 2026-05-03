'use client';

import { useState, useRef, useCallback } from 'react';
import Link from 'next/link';
import {
  loadShapeOptSurrogate,
  predictCd,
  type ShapeOptModel,
} from '../../lib/shapeopt_surrogate';
import { MODEL_LABELS } from '../../lib/models';

const NUM_FFD = 108;

// Base model presets: all-zeros means the undeformed shape.
const BASE_MODELS: Record<string, string> = {
  ahmed25: 'Ahmed 25\u00b0',
  ahmed35: 'Ahmed 35\u00b0',
  car: 'Car',
};

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

// SPSA optimizer -- 2 forward passes per step regardless of dimension

interface SPSAConfig {
  a: number;     // step size scale
  c: number;     // perturbation size
  A: number;     // stability constant (typically ~10% of max iterations)
  alpha: number; // step size decay exponent
  gamma: number; // perturbation decay exponent
}

const DEFAULT_SPSA: SPSAConfig = {
  a: 0.05,
  c: 1e-3,
  A: 20,
  alpha: 0.602,
  gamma: 0.101,
};

interface OptState {
  params: number[];
  step: number;
  cdHistory: number[];
  clHistory: number[];
}

function spsaStep(
  model: ShapeOptModel,
  state: OptState,
  windSpeed: number,
  reynolds: number,
  cfg: SPSAConfig,
): OptState {
  const k = state.step + 1;
  const ak = cfg.a / Math.pow(k + cfg.A, cfg.alpha);
  const ck = cfg.c / Math.pow(k, cfg.gamma);

  // Random perturbation direction: +1 or -1 for each param
  const delta = new Array(NUM_FFD);
  for (let i = 0; i < NUM_FFD; i++) {
    delta[i] = Math.random() < 0.5 ? -1 : 1;
  }

  // Perturbed parameter vectors
  const pPlus = state.params.map((p, i) => p + ck * delta[i]);
  const pMinus = state.params.map((p, i) => p - ck * delta[i]);

  // Two forward passes
  const fPlus = predictCd(model, pPlus, windSpeed, reynolds);
  const fMinus = predictCd(model, pMinus, windSpeed, reynolds);

  // Gradient estimate (minimize cd)
  const gradScale = (fPlus.cd - fMinus.cd) / (2 * ck);

  // Update
  const newParams = state.params.map((p, i) =>
    clamp(p - ak * gradScale / delta[i], -0.15, 0.15),
  );

  const newPred = predictCd(model, newParams, windSpeed, reynolds);

  return {
    params: newParams,
    step: k,
    cdHistory: [...state.cdHistory, newPred.cd],
    clHistory: [...state.clHistory, newPred.cl],
  };
}

// Mini Cd-vs-iteration chart (inline SVG)

function ConvergenceChart({ cdHistory }: { cdHistory: number[] }) {
  if (cdHistory.length < 2) return null;

  const W = 480;
  const H = 200;
  const pad = { top: 16, right: 16, bottom: 28, left: 52 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const yMin = Math.min(...cdHistory) * 0.98;
  const yMax = Math.max(...cdHistory) * 1.02;
  const yRange = yMax - yMin || 1e-6;
  const xMax = cdHistory.length - 1;

  const toX = (i: number) => pad.left + (i / Math.max(xMax, 1)) * plotW;
  const toY = (v: number) => pad.top + plotH - ((v - yMin) / yRange) * plotH;

  const points = cdHistory.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');

  // Y-axis ticks
  const yTicks = [yMin, yMin + yRange * 0.5, yMax];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxWidth: W }}>
      {/* Grid lines */}
      {yTicks.map((v, i) => (
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
            x={pad.left - 6}
            y={toY(v) + 3}
            textAnchor="end"
            className="fill-ctp-overlay0"
            fontSize={9}
          >
            {v.toFixed(4)}
          </text>
        </g>
      ))}

      {/* The line */}
      <polyline
        points={points}
        fill="none"
        className="stroke-ctp-mauve"
        strokeWidth={1.5}
        strokeLinejoin="round"
      />

      {/* X-axis label */}
      <text
        x={pad.left + plotW / 2}
        y={H - 4}
        textAnchor="middle"
        className="fill-ctp-subtext0"
        fontSize={10}
      >
        Iteration
      </text>

      {/* Y-axis label */}
      <text
        x={12}
        y={pad.top + plotH / 2}
        textAnchor="middle"
        className="fill-ctp-subtext0"
        fontSize={10}
        transform={`rotate(-90, 12, ${pad.top + plotH / 2})`}
      >
        Cd
      </text>
    </svg>
  );
}

// Main page

type RunStatus = 'idle' | 'loading-model' | 'running' | 'done' | 'error';

export default function OptimizePage() {
  const [baseModel, setBaseModel] = useState<string>('ahmed25');
  const [windSpeed, setWindSpeed] = useState(2.5);
  const [reynolds, setReynolds] = useState(5000);
  const [maxSteps, setMaxSteps] = useState(200);

  const [status, setStatus] = useState<RunStatus>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [optState, setOptState] = useState<OptState | null>(null);
  const [baselineCd, setBaselineCd] = useState<number | null>(null);
  const [baselineCl, setBaselineCl] = useState<number | null>(null);

  const modelRef = useRef<ShapeOptModel | null>(null);
  const cancelRef = useRef(false);

  const runOptimization = useCallback(async () => {
    cancelRef.current = false;
    setErrorMsg(null);
    setOptState(null);
    setBaselineCd(null);
    setBaselineCl(null);

    // Load model if not cached
    if (!modelRef.current) {
      setStatus('loading-model');
      try {
        modelRef.current = await loadShapeOptSurrogate();
      } catch (err) {
        setStatus('error');
        setErrorMsg(
          err instanceof Error ? err.message : 'Failed to load surrogate model',
        );
        return;
      }
    }

    const surr = modelRef.current;

    // Baseline prediction (all-zeros FFD = undeformed shape)
    const zeroFFD = new Array(NUM_FFD).fill(0);
    const baseline = predictCd(surr, zeroFFD, windSpeed, reynolds);
    setBaselineCd(baseline.cd);
    setBaselineCl(baseline.cl);

    // Initialize SPSA
    let state: OptState = {
      params: [...zeroFFD],
      step: 0,
      cdHistory: [baseline.cd],
      clHistory: [baseline.cl],
    };

    setStatus('running');

    // Run optimization in batches to keep the UI responsive
    const BATCH = 10;
    for (let i = 0; i < maxSteps; i += BATCH) {
      if (cancelRef.current) break;

      const batchEnd = Math.min(i + BATCH, maxSteps);
      for (let j = i; j < batchEnd; j++) {
        state = spsaStep(surr, state, windSpeed, reynolds, DEFAULT_SPSA);
      }

      // Yield to the event loop so React can re-render
      setOptState({ ...state });
      await new Promise((r) => setTimeout(r, 0));
    }

    setStatus('done');
  }, [windSpeed, reynolds, maxSteps]);

  const cancel = useCallback(() => {
    cancelRef.current = true;
    setStatus('done');
  }, []);

  const isRunning = status === 'running' || status === 'loading-model';

  // Results
  const finalCd =
    optState && optState.cdHistory.length > 0
      ? optState.cdHistory[optState.cdHistory.length - 1]
      : null;
  const finalCl =
    optState && optState.clHistory.length > 0
      ? optState.clHistory[optState.clHistory.length - 1]
      : null;
  const improvement =
    baselineCd !== null && finalCd !== null && Math.abs(baselineCd) > 1e-10
      ? ((baselineCd - finalCd) / Math.abs(baselineCd)) * 100
      : null;

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
            Shape Optimization
          </h1>
          <p className="text-xs text-ctp-subtext0">
            Minimize drag via surrogate-driven SPSA optimization on FFD
            parameters, entirely in-browser
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
        <div className="border border-ctp-surface1 rounded-lg p-3 bg-ctp-mantle">
          <label className="text-[10px] text-ctp-overlay0 uppercase tracking-wider block mb-1">
            Base Model
          </label>
          <select
            value={baseModel}
            onChange={(e) => setBaseModel(e.target.value)}
            disabled={isRunning}
            className="w-full bg-ctp-surface0 border border-ctp-surface1 rounded px-2 py-1.5 text-sm text-ctp-text accent-ctp-mauve disabled:opacity-50"
          >
            {Object.entries(BASE_MODELS).map(([k, label]) => (
              <option key={k} value={k}>
                {label}
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
            disabled={isRunning}
            className="w-full accent-ctp-mauve disabled:opacity-50"
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
            disabled={isRunning}
            className="w-full bg-ctp-surface0 border border-ctp-surface1 rounded px-2 py-1.5 text-sm font-mono text-ctp-text disabled:opacity-50"
          />
        </div>

        <div className="border border-ctp-surface1 rounded-lg p-3 bg-ctp-mantle">
          <label className="text-[10px] text-ctp-overlay0 uppercase tracking-wider block mb-1">
            Max Iterations
          </label>
          <input
            type="number"
            min={10}
            max={1000}
            step={10}
            value={maxSteps}
            onChange={(e) =>
              setMaxSteps(clamp(parseInt(e.target.value) || 200, 10, 1000))
            }
            disabled={isRunning}
            className="w-full bg-ctp-surface0 border border-ctp-surface1 rounded px-2 py-1.5 text-sm font-mono text-ctp-text disabled:opacity-50"
          />
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={runOptimization}
          disabled={isRunning}
          className="px-5 py-2 rounded-lg bg-ctp-mauve text-ctp-base text-sm font-medium
                     hover:bg-ctp-pink transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {status === 'loading-model'
            ? 'Loading model...'
            : status === 'running'
              ? `Running (${optState?.step ?? 0}/${maxSteps})...`
              : 'Run Optimization'}
        </button>

        {status === 'running' && (
          <button
            onClick={cancel}
            className="px-5 py-2 rounded-lg border border-ctp-surface1 text-ctp-subtext0 text-sm
                       hover:border-ctp-red hover:text-ctp-red transition-colors"
          >
            Stop
          </button>
        )}

        {status === 'done' && optState && (
          <button
            onClick={() => {
              // TODO: submit optimized FFD params to LBM validation endpoint
              alert(
                'Validation not yet connected. The optimized FFD parameters are ready for export.',
              );
            }}
            className="px-5 py-2 rounded-lg border border-ctp-blue text-ctp-blue text-sm
                       hover:bg-ctp-blue hover:text-ctp-base transition-colors"
          >
            Validate with LBM
          </button>
        )}
      </div>

      {/* Error display */}
      {status === 'error' && errorMsg && (
        <div className="mb-6 p-3 bg-ctp-surface0 border border-ctp-red/40 rounded text-xs text-ctp-red">
          {errorMsg}
        </div>
      )}

      {/* Progress bar */}
      {(status === 'running' || status === 'done') && optState && (
        <div className="mb-6">
          <div className="flex justify-between text-[10px] text-ctp-overlay0 mb-1">
            <span>Progress</span>
            <span>
              {optState.step} / {maxSteps} iterations
            </span>
          </div>
          <div className="h-1.5 bg-ctp-surface0 rounded-full overflow-hidden">
            <div
              className="h-full bg-ctp-mauve rounded-full transition-all duration-150"
              style={{ width: `${(optState.step / maxSteps) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Results cards */}
      {baselineCd !== null && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {/* Baseline */}
          <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
            <div className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-2">
              Baseline ({MODEL_LABELS[baseModel] ?? baseModel})
            </div>
            <div className="space-y-2">
              <div>
                <span className="text-[10px] text-ctp-subtext0 uppercase">
                  Cd
                </span>
                <div className="text-xl font-mono text-ctp-text">
                  {baselineCd.toFixed(4)}
                </div>
              </div>
              <div>
                <span className="text-[10px] text-ctp-subtext0 uppercase">
                  Cl
                </span>
                <div className="text-xl font-mono text-ctp-text">
                  {baselineCl !== null ? baselineCl.toFixed(4) : '--'}
                </div>
              </div>
            </div>
          </div>

          {/* Optimized */}
          <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
            <div className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-2">
              Optimized
            </div>
            <div className="space-y-2">
              <div>
                <span className="text-[10px] text-ctp-subtext0 uppercase">
                  Cd
                </span>
                <div className="text-xl font-mono text-ctp-text">
                  {finalCd !== null ? finalCd.toFixed(4) : '--'}
                </div>
              </div>
              <div>
                <span className="text-[10px] text-ctp-subtext0 uppercase">
                  Cl
                </span>
                <div className="text-xl font-mono text-ctp-text">
                  {finalCl !== null ? finalCl.toFixed(4) : '--'}
                </div>
              </div>
            </div>
          </div>

          {/* Improvement */}
          <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
            <div className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-2">
              Improvement
            </div>
            <div className="flex items-baseline gap-2">
              <span
                className={`text-3xl font-mono ${
                  improvement !== null && improvement > 0
                    ? 'text-ctp-green'
                    : improvement !== null && improvement < 0
                      ? 'text-ctp-red'
                      : 'text-ctp-text'
                }`}
              >
                {improvement !== null
                  ? `${improvement >= 0 ? '-' : '+'}${Math.abs(improvement).toFixed(2)}%`
                  : '--'}
              </span>
              <span className="text-xs text-ctp-subtext0">drag reduction</span>
            </div>
            {improvement !== null && improvement > 0 && (
              <p className="mt-2 text-xs text-ctp-subtext0">
                Cd reduced from {baselineCd!.toFixed(4)} to {finalCd!.toFixed(4)}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Convergence chart */}
      {optState && optState.cdHistory.length > 1 && (
        <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle mb-6">
          <div className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-2">
            Convergence
          </div>
          <ConvergenceChart cdHistory={optState.cdHistory} />
        </div>
      )}

      {/* FFD parameter summary (collapsed by default) */}
      {status === 'done' && optState && (
        <details className="border border-ctp-surface1 rounded-lg bg-ctp-mantle mb-6">
          <summary className="p-4 cursor-pointer text-xs text-ctp-subtext0 hover:text-ctp-text transition-colors">
            Optimized FFD Parameters ({NUM_FFD} values)
          </summary>
          <div className="px-4 pb-4">
            <div className="overflow-x-auto max-h-48 overflow-y-auto">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-ctp-mantle">
                  <tr className="border-b border-ctp-surface1">
                    <th className="py-1 pr-3 text-ctp-subtext0 font-medium w-16">
                      Index
                    </th>
                    <th className="py-1 text-ctp-subtext0 font-medium">
                      Displacement
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {optState.params.map((v, i) => (
                    <tr key={i} className="border-b border-ctp-surface0">
                      <td className="py-0.5 pr-3 font-mono text-ctp-subtext1">
                        {i}
                      </td>
                      <td
                        className={`py-0.5 font-mono ${
                          Math.abs(v) > 0.01 ? 'text-ctp-mauve' : 'text-ctp-overlay0'
                        }`}
                      >
                        {v.toFixed(6)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </details>
      )}

      {/* Explanation */}
      {status === 'idle' && (
        <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle text-xs text-ctp-subtext0 leading-relaxed">
          <p className="mb-2">
            This tool optimizes the shape of a vehicle body to minimize aerodynamic
            drag. It works by adjusting {NUM_FFD} Free-Form Deformation (FFD)
            control point displacements while evaluating the drag coefficient
            through a neural network surrogate.
          </p>
          <p className="mb-2">
            The optimizer uses SPSA (Simultaneous Perturbation Stochastic
            Approximation), which estimates the gradient with just 2 forward
            passes per step regardless of the number of parameters. This makes it
            practical to run hundreds of iterations in real time.
          </p>
          <p>
            After optimization, you can validate the result by running a full
            Lattice Boltzmann simulation on the deformed geometry.
          </p>
        </div>
      )}
    </main>
  );
}
