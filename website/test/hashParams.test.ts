import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import {
  DEFAULT_PARAMS,
  formatHash,
  parseHash,
  useHashParams,
} from '../src/lib/hashParams';

describe('parseHash', () => {
  it('returns nothing for an empty hash', () => {
    expect(parseHash('')).toEqual({});
    expect(parseHash('#')).toEqual({});
  });

  it('reads every field', () => {
    expect(parseHash('#ws=2.5&vm=3&cm=1&d=15&m=ahmed35&re=5000')).toEqual({
      windSpeed: 2.5,
      vizMode: 3,
      collisionMode: 1,
      duration: 15,
      model: 'ahmed35',
      reynolds: 5000,
    });
  });

  it('clamps out of range values and drops unknown models', () => {
    expect(parseHash('ws=99&vm=-4&d=1&m=rocket&re=1e9')).toEqual({
      windSpeed: 5,
      vizMode: 0,
      duration: 5,
      reynolds: 100000,
    });
  });

  it('falls back on garbage numbers', () => {
    expect(parseHash('ws=abc&cm=xyz')).toEqual({
      windSpeed: 1,
      collisionMode: 1,
    });
  });

  it('round trips through formatHash', () => {
    const params = { ...DEFAULT_PARAMS, windSpeed: 3.2, model: 'ahmed25' };
    expect({ ...DEFAULT_PARAMS, ...parseHash(formatHash(params)) }).toEqual(
      params,
    );
  });
});

describe('useHashParams', () => {
  afterEach(() => {
    window.history.replaceState(null, '', '/');
  });

  it('reads params from the current hash', () => {
    window.history.replaceState(null, '', '/#ws=4&m=ahmed25');
    const { result } = renderHook(() => useHashParams());
    expect(result.current[0].windSpeed).toBe(4);
    expect(result.current[0].model).toBe('ahmed25');
  });

  it('writes the hash and re-renders on change', () => {
    const { result } = renderHook(() => useHashParams());
    expect(result.current[0]).toEqual(DEFAULT_PARAMS);
    act(() => result.current[1]({ ...DEFAULT_PARAMS, windSpeed: 2 }));
    expect(result.current[0].windSpeed).toBe(2);
    expect(window.location.hash).toContain('ws=2');
  });
});
