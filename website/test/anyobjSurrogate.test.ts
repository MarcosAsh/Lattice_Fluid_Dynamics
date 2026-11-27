import { renderHook, waitFor } from '@testing-library/react';
import { readFile } from 'node:fs/promises';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useSurrogate } from '../src/lib/anyobj_surrogate';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('useSurrogate', () => {
  it('predicts finite coefficients when Reynolds is on Auto', async () => {
    const model = await readFile('public/models/anyobj.json', 'utf8');
    vi.stubGlobal(
      'fetch',
      vi.fn<typeof fetch>(async () => new Response(model)),
    );

    const { result } = renderHook(() => useSurrogate());
    await waitFor(() => expect(result.current.status).toBe('ready'));

    const auto = result.current.predict(1, 0, 'car');
    expect(Number.isFinite(auto?.cd)).toBe(true);
    expect(Number.isFinite(auto?.cl)).toBe(true);
    expect(result.current.predict(1, 0, 'rocket')).toBeNull();
  });
});
