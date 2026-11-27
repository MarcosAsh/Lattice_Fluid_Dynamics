import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  MAX_SAVED_RUNS,
  RUNS_KEY,
  addRun,
  getRuns,
  parseRuns,
  resetRunsCache,
  setRuns,
  useRuns,
} from '../src/lib/runs';
import type { SimulationResult } from '../src/lib/types';

function run(timestamp: number, extra: Partial<SimulationResult> = {}) {
  return {
    videoUrl: `https://example.com/${timestamp}.mp4`,
    cdValue: 0.3,
    clValue: 0.01,
    cdSeries: [0.3],
    clSeries: [0.01],
    charLength: 32,
    sampleInterval: 10,
    tStar: null,
    flowThroughs: null,
    cfl: null,
    cdPressureSeries: [],
    cdFrictionSeries: [],
    model: 'car',
    windSpeed: 1,
    timestamp,
    ...extra,
  } satisfies SimulationResult;
}

beforeEach(() => {
  localStorage.clear();
  resetRunsCache();
});

describe('parseRuns', () => {
  it('handles missing and corrupt storage', () => {
    expect(parseRuns(null)).toEqual([]);
    expect(parseRuns('not json')).toEqual([]);
    expect(parseRuns('{"a":1}')).toEqual([]);
  });

  it('backfills fields older saves lack and drops junk entries', () => {
    const raw = JSON.stringify([
      { model: 'car', windSpeed: 1, timestamp: 5, cdValue: 0.4 },
      { model: 42 },
      null,
    ]);
    const [only, ...rest] = parseRuns(raw);
    expect(rest).toEqual([]);
    expect(only).toMatchObject({
      model: 'car',
      cdValue: 0.4,
      tStar: null,
      cdPressureSeries: [],
      cdSeries: [],
    });
    expect(only?.params).toBeUndefined();
  });

  it('keeps stored params when they are well formed', () => {
    const params = {
      windSpeed: 2,
      vizMode: 1,
      duration: 10,
      collisionMode: 2,
      model: 'car',
      reynolds: 0,
    };
    const [r] = parseRuns(JSON.stringify([run(1, { params })]));
    expect(r?.params).toEqual(params);
  });
});

describe('run store', () => {
  it('loads what is already in storage', () => {
    localStorage.setItem(RUNS_KEY, JSON.stringify([run(1), run(2)]));
    expect(getRuns().map((r) => r.timestamp)).toEqual([1, 2]);
  });

  it('persists added runs and keeps timestamps unique', () => {
    addRun(run(10));
    const second = addRun(run(10));
    expect(second.timestamp).toBe(11);
    const saved = JSON.parse(localStorage.getItem(RUNS_KEY) ?? '[]');
    expect(saved).toHaveLength(2);
  });

  it('caps the saved history', () => {
    setRuns(Array.from({ length: MAX_SAVED_RUNS + 5 }, (_, i) => run(i)));
    expect(getRuns()).toHaveLength(MAX_SAVED_RUNS);
    expect(getRuns()[0]?.timestamp).toBe(5);
  });

  it('notifies hook users of new runs', () => {
    const { result } = renderHook(() => useRuns());
    expect(result.current).toEqual([]);
    act(() => {
      addRun(run(1));
    });
    expect(result.current).toHaveLength(1);
  });

  it('picks up runs saved in another tab', () => {
    const { result } = renderHook(() => useRuns());
    act(() => {
      window.dispatchEvent(
        new StorageEvent('storage', {
          key: RUNS_KEY,
          newValue: JSON.stringify([run(7)]),
        }),
      );
    });
    expect(result.current[0]?.timestamp).toBe(7);
  });
});
