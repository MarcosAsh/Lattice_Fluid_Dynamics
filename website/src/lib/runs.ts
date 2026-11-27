import { useSyncExternalStore } from 'react';
import type { SimulationParams, SimulationResult } from './types';

export const RUNS_KEY = 'lattice_results';
export const MAX_SAVED_RUNS = 50;

const EMPTY: SimulationResult[] = [];

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function numberArray(v: unknown): number[] {
  return Array.isArray(v) ? v.filter((x) => typeof x === 'number') : [];
}

function parseParams(v: unknown): SimulationParams | undefined {
  if (!isRecord(v) || typeof v.model !== 'string') return undefined;
  const keys = [
    'windSpeed',
    'vizMode',
    'duration',
    'collisionMode',
    'reynolds',
  ] as const;
  for (const k of keys) {
    if (typeof v[k] !== 'number') return undefined;
  }
  return {
    windSpeed: v.windSpeed as number,
    vizMode: v.vizMode as number,
    duration: v.duration as number,
    collisionMode: v.collisionMode as number,
    reynolds: v.reynolds as number,
    model: v.model,
  };
}

// Skip malformed entries and backfill newer fields
export function parseRuns(raw: string | null): SimulationResult[] {
  if (!raw) return EMPTY;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return EMPTY;
  }
  if (!Array.isArray(parsed)) return EMPTY;
  const runs: SimulationResult[] = [];
  for (const r of parsed) {
    if (
      !isRecord(r) ||
      typeof r.model !== 'string' ||
      typeof r.windSpeed !== 'number' ||
      typeof r.timestamp !== 'number'
    ) {
      continue;
    }
    const params = parseParams(r.params);
    runs.push({
      videoUrl: typeof r.videoUrl === 'string' ? r.videoUrl : '',
      cdValue: numberOrNull(r.cdValue),
      clValue: numberOrNull(r.clValue),
      cdSeries: numberArray(r.cdSeries),
      clSeries: numberArray(r.clSeries),
      charLength: numberOrNull(r.charLength),
      sampleInterval: numberOrNull(r.sampleInterval),
      tStar: numberOrNull(r.tStar),
      flowThroughs: numberOrNull(r.flowThroughs),
      cfl: numberOrNull(r.cfl),
      cdPressureSeries: numberArray(r.cdPressureSeries),
      cdFrictionSeries: numberArray(r.cdFrictionSeries),
      model: r.model,
      windSpeed: r.windSpeed,
      timestamp: r.timestamp,
      ...(params ? { params } : {}),
    });
  }
  return runs;
}

const listeners = new Set<() => void>();
let runs: SimulationResult[] | null = null;

function readStorage(): string | null {
  try {
    return window.localStorage.getItem(RUNS_KEY);
  } catch {
    return null;
  }
}

function notify() {
  for (const l of listeners) l();
}

function onStorage(e: StorageEvent) {
  if (e.key !== null && e.key !== RUNS_KEY) return;
  runs = parseRuns(e.newValue);
  notify();
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  if (listeners.size === 1) window.addEventListener('storage', onStorage);
  return () => {
    listeners.delete(cb);
    if (listeners.size === 0) window.removeEventListener('storage', onStorage);
  };
}

export function getRuns(): SimulationResult[] {
  if (runs === null) runs = parseRuns(readStorage());
  return runs;
}

export function setRuns(next: SimulationResult[]) {
  runs = next.slice(-MAX_SAVED_RUNS);
  try {
    window.localStorage.setItem(RUNS_KEY, JSON.stringify(runs));
  } catch {
    // Storage full or blocked, keep the runs in memory
  }
  notify();
}

export function addRun(run: SimulationResult): SimulationResult {
  const current = getRuns();
  const last = current[current.length - 1];
  // Timestamps double as run ids, so keep them unique
  const timestamp =
    last && run.timestamp <= last.timestamp
      ? last.timestamp + 1
      : run.timestamp;
  const stored = { ...run, timestamp };
  setRuns([...current, stored]);
  return stored;
}

// Make the next read go back to storage
export function resetRunsCache() {
  runs = null;
}

export function useRuns(): SimulationResult[] {
  return useSyncExternalStore(subscribe, getRuns, () => EMPTY);
}
