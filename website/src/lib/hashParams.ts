import { useCallback, useSyncExternalStore } from 'react';
import type { SimulationParams } from './types';

export const DEFAULT_PARAMS: SimulationParams = {
  windSpeed: 1.0,
  vizMode: 1,
  duration: 5,
  collisionMode: 2,
  model: 'car',
  reynolds: 0,
};

const VALID_MODELS = new Set(['car', 'ahmed25', 'ahmed35', 'custom']);

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function parseNum(raw: string, fallback: number, radix?: number): number {
  const v = radix ? parseInt(raw, radix) : parseFloat(raw);
  return Number.isNaN(v) ? fallback : v;
}

export function parseHash(hash: string): Partial<SimulationParams> {
  const body = hash.startsWith('#') ? hash.slice(1) : hash;
  if (!body) return {};
  const p = new URLSearchParams(body);
  const result: Partial<SimulationParams> = {};
  const ws = p.get('ws');
  const vm = p.get('vm');
  const cm = p.get('cm');
  const d = p.get('d');
  const m = p.get('m');
  const re = p.get('re');
  if (ws !== null) result.windSpeed = clamp(parseNum(ws, 1.0), 0, 5);
  if (vm !== null) result.vizMode = clamp(parseNum(vm, 1, 10), 0, 6);
  if (cm !== null) result.collisionMode = clamp(parseNum(cm, 1, 10), 0, 3);
  if (d !== null) result.duration = clamp(parseNum(d, 10, 10), 5, 30);
  if (m !== null && VALID_MODELS.has(m)) result.model = m;
  if (re !== null) result.reynolds = clamp(parseNum(re, 0), 0, 100000);
  return result;
}

export function formatHash(params: SimulationParams): string {
  return `ws=${params.windSpeed}&vm=${params.vizMode}&cm=${params.collisionMode}&d=${params.duration}&m=${params.model}&re=${params.reynolds}`;
}

// Params live in the URL hash so shared links survive hydration
const listeners = new Set<() => void>();
let cachedHash: string | null = null;
let cachedParams: SimulationParams = DEFAULT_PARAMS;

function notify() {
  for (const l of listeners) l();
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  if (listeners.size === 1) window.addEventListener('hashchange', notify);
  return () => {
    listeners.delete(cb);
    if (listeners.size === 0) window.removeEventListener('hashchange', notify);
  };
}

function getSnapshot(): SimulationParams {
  const hash = window.location.hash;
  if (hash !== cachedHash) {
    cachedHash = hash;
    cachedParams = { ...DEFAULT_PARAMS, ...parseHash(hash) };
  }
  return cachedParams;
}

function getServerSnapshot(): SimulationParams {
  return DEFAULT_PARAMS;
}

export function useHashParams(): [
  SimulationParams,
  (next: SimulationParams) => void,
] {
  const params = useSyncExternalStore(
    subscribe,
    getSnapshot,
    getServerSnapshot,
  );
  const setParams = useCallback((next: SimulationParams) => {
    window.history.replaceState(null, '', `#${formatHash(next)}`);
    notify();
  }, []);
  return [params, setParams];
}
