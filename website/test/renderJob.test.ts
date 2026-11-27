import { describe, expect, it, vi } from 'vitest';
import {
  MAX_SWEEP_RUNS,
  encodeObjFile,
  runRenderJob,
  sweepSpeeds,
} from '../src/lib/renderJob';
import { DEFAULT_PARAMS } from '../src/lib/hashParams';

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function fakeFetch(responses: Response[]) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const impl = vi.fn(
    async (url: string | URL | Request, init?: RequestInit) => {
      calls.push({ url: String(url), init });
      const next = responses.shift();
      if (!next) throw new Error('unexpected fetch');
      return next;
    },
  );
  return { impl: impl as unknown as typeof fetch, calls };
}

const instant = { sleep: async () => {}, pollIntervalMs: 0 };

describe('sweepSpeeds', () => {
  it('steps from min to max inclusive', () => {
    expect(sweepSpeeds(0.5, 2, 0.5)).toEqual([0.5, 1, 1.5, 2]);
  });

  it('rejects steps that would never finish', () => {
    expect(sweepSpeeds(0, 5, 0)).toEqual([]);
    expect(sweepSpeeds(0, 5, -1)).toEqual([]);
    expect(sweepSpeeds(3, 1, 0.5)).toEqual([]);
    expect(sweepSpeeds(0, Number.NaN, 0.5)).toEqual([]);
  });

  it('caps the number of runs', () => {
    expect(sweepSpeeds(0, 5, 0.01)).toHaveLength(MAX_SWEEP_RUNS);
  });
});

describe('encodeObjFile', () => {
  it('encodes non-ASCII bytes without throwing', async () => {
    const text = '# Modell für Windkanal\nv 0 0 0\n';
    const b64 = await encodeObjFile(new Blob([text]));
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    expect(new TextDecoder().decode(bytes)).toBe(text);
  });
});

describe('runRenderJob', () => {
  it('polls until the job completes and maps the result', async () => {
    const { impl, calls } = fakeFetch([
      json({ status: 'accepted', jobId: 'job 1' }),
      json({ status: 'rendering', cd_series: [0.5] }),
      new Response('busy', { status: 502 }),
      json({
        status: 'complete',
        video_url: 'https://example.com/v.mp4',
        cd_value: 0.31,
        cl_value: 0.02,
        cd_series: [0.5, 0.31],
        t_star: 1.5,
      }),
    ]);
    const partials: number[][] = [];
    const result = await runRenderJob(
      { ...DEFAULT_PARAMS, windSpeed: 2 },
      {
        ...instant,
        fetchImpl: impl,
        onPartial: (p) => partials.push(p.cdSeries),
      },
    );

    expect(calls[0]?.init?.method).toBe('POST');
    expect(calls[1]?.url).toBe('/api/render?jobId=job%201');
    expect(partials).toEqual([[0.5]]);
    expect(result).toMatchObject({
      videoUrl: 'https://example.com/v.mp4',
      cdValue: 0.31,
      clValue: 0.02,
      tStar: 1.5,
      flowThroughs: null,
      model: 'car',
      windSpeed: 2,
      params: { ...DEFAULT_PARAMS, windSpeed: 2 },
    });
  });

  it('sends the OBJ data with the params', async () => {
    const { impl, calls } = fakeFetch([
      json({ status: 'accepted', jobId: 'j' }),
      json({ status: 'complete', video_url: 'v' }),
    ]);
    await runRenderJob(
      { ...DEFAULT_PARAMS, model: 'custom' },
      { ...instant, fetchImpl: impl, objData: 'AAAA' },
    );
    const body = JSON.parse(String(calls[0]?.init?.body));
    expect(body).toMatchObject({ model: 'custom', objData: 'AAAA' });
  });

  it('surfaces the server error message on submit', async () => {
    const { impl } = fakeFetch([
      json(
        { status: 'error', error: 'Render backend is not configured.' },
        503,
      ),
    ]);
    await expect(
      runRenderJob(DEFAULT_PARAMS, { ...instant, fetchImpl: impl }),
    ).rejects.toThrow('Render backend is not configured.');
  });

  it('fails when the worker reports an error', async () => {
    const { impl } = fakeFetch([
      json({ status: 'accepted', jobId: 'j' }),
      json({ status: 'error', error: 'GPU crashed' }),
    ]);
    await expect(
      runRenderJob(DEFAULT_PARAMS, { ...instant, fetchImpl: impl }),
    ).rejects.toThrow('GPU crashed');
  });

  it('times out after the poll budget', async () => {
    let t = 0;
    const { impl } = fakeFetch([
      json({ status: 'accepted', jobId: 'j' }),
      json({ status: 'rendering' }),
    ]);
    await expect(
      runRenderJob(DEFAULT_PARAMS, {
        fetchImpl: impl,
        pollIntervalMs: 0,
        sleep: async () => {
          t += 10 * 60 * 1000;
        },
        now: () => t,
      }),
    ).rejects.toThrow('Render timed out');
  });
});
