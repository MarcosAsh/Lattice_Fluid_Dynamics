import { computeStrouhal } from './spectral';
import { MODEL_LABELS } from './models';
import type { SimulationResult } from './types';

export interface MetricRow {
  key: string;
  label: string;
  a: number | null;
  b: number | null;
  // b minus a, null when either side is missing
  diff: number | null;
  // diff relative to a, null when a is missing or zero
  pct: number | null;
  digits: number;
}

export interface ParamRow {
  key: string;
  label: string;
  a: string;
  b: string;
  differs: boolean;
}

const VIZ_LABELS: Record<number, string> = {
  0: 'Depth',
  1: 'Velocity Magnitude',
  2: 'Velocity Direction',
  3: 'Particle Lifetime',
  4: 'Turbulence',
  5: 'Flow Progress',
  6: 'Vorticity',
};

const COLLISION_LABELS: Record<number, string> = {
  0: 'Off',
  1: 'AABB',
  2: 'Per-Triangle',
  3: 'Voxel',
};

export function runLabel(run: SimulationResult): string {
  return `${MODEL_LABELS[run.model] ?? run.model} at ${run.windSpeed.toFixed(1)} m/s`;
}

export function strouhalOf(run: SimulationResult): number | null {
  if (!run.charLength || run.clSeries.length < 8) return null;
  return (
    computeStrouhal(
      run.clSeries,
      run.charLength,
      run.sampleInterval ?? undefined,
    )?.strouhal ?? null
  );
}

function metric(
  key: string,
  label: string,
  a: number | null,
  b: number | null,
  digits = 4,
): MetricRow {
  const diff = a !== null && b !== null ? b - a : null;
  const pct =
    diff !== null && a !== null && Math.abs(a) > 1e-12
      ? (diff / Math.abs(a)) * 100
      : null;
  return { key, label, a, b, diff, pct, digits };
}

export function compareMetrics(
  a: SimulationResult,
  b: SimulationResult,
): MetricRow[] {
  return [
    metric('cd', 'Drag coefficient (Cd)', a.cdValue, b.cdValue),
    metric('cl', 'Lift coefficient (Cl)', a.clValue, b.clValue),
    metric('st', 'Strouhal number', strouhalOf(a), strouhalOf(b)),
    metric('tStar', 'Dimensionless time (t*)', a.tStar, b.tStar, 3),
    metric('flowThroughs', 'Flow-throughs', a.flowThroughs, b.flowThroughs, 2),
    metric('cfl', 'CFL', a.cfl, b.cfl),
  ];
}

function reynoldsText(re: number): string {
  return re === 0 ? 'Auto' : re.toFixed(0);
}

export function compareParams(
  a: SimulationResult,
  b: SimulationResult,
): ParamRow[] {
  const unknown = 'Not recorded';
  const rows: [string, string, string, string][] = [
    [
      'model',
      'Model',
      MODEL_LABELS[a.model] ?? a.model,
      MODEL_LABELS[b.model] ?? b.model,
    ],
    [
      'windSpeed',
      'Wind speed',
      `${a.windSpeed.toFixed(1)} m/s`,
      `${b.windSpeed.toFixed(1)} m/s`,
    ],
    [
      'reynolds',
      'Reynolds number',
      a.params ? reynoldsText(a.params.reynolds) : unknown,
      b.params ? reynoldsText(b.params.reynolds) : unknown,
    ],
    [
      'duration',
      'Duration',
      a.params ? `${a.params.duration} s` : unknown,
      b.params ? `${b.params.duration} s` : unknown,
    ],
    [
      'vizMode',
      'Visualization',
      a.params
        ? (VIZ_LABELS[a.params.vizMode] ?? `${a.params.vizMode}`)
        : unknown,
      b.params
        ? (VIZ_LABELS[b.params.vizMode] ?? `${b.params.vizMode}`)
        : unknown,
    ],
    [
      'collisionMode',
      'Collision',
      a.params
        ? (COLLISION_LABELS[a.params.collisionMode] ??
          `${a.params.collisionMode}`)
        : unknown,
      b.params
        ? (COLLISION_LABELS[b.params.collisionMode] ??
          `${b.params.collisionMode}`)
        : unknown,
    ],
  ];
  return rows.map(([key, label, av, bv]) => ({
    key,
    label,
    a: av,
    b: bv,
    differs: av !== bv,
  }));
}

function cell(series: number[], i: number): string {
  const v = series[i];
  return v === undefined ? '' : v.toFixed(6);
}

// Both runs' series in one file, aligned by measurement step
export function comparisonCsv(
  a: SimulationResult,
  b: SimulationResult,
): string {
  const rows: string[] = [`# Run A: ${runLabel(a)}`, `# Run B: ${runLabel(b)}`];
  for (const m of compareMetrics(a, b)) {
    const fmt = (v: number | null) => (v === null ? '' : v.toFixed(6));
    rows.push(`# ${m.label}: A=${fmt(m.a)} B=${fmt(m.b)}`);
  }
  rows.push('step,cd_a,cl_a,cd_b,cl_b');
  const len = Math.max(
    a.cdSeries.length,
    a.clSeries.length,
    b.cdSeries.length,
    b.clSeries.length,
  );
  for (let i = 0; i < len; i++) {
    rows.push(
      `${i},${cell(a.cdSeries, i)},${cell(a.clSeries, i)},${cell(b.cdSeries, i)},${cell(b.clSeries, i)}`,
    );
  }
  return rows.join('\n');
}
