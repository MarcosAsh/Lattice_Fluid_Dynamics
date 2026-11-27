// @vitest-environment node
import { afterEach, describe, expect, it, vi } from 'vitest';
import { parseRenderRequest } from '../src/lib/renderRequest';
import { GET, POST } from '../src/app/api/render/route';

describe('parseRenderRequest', () => {
  it('fills defaults for a bare object', () => {
    expect(parseRenderRequest({})).toEqual({
      ok: true,
      body: {
        wind_speed: 1,
        viz_mode: 1,
        collision_mode: 2,
        duration: 5,
        model: 'car',
        reynolds: 0,
      },
    });
  });

  it('clamps numbers into range', () => {
    const r = parseRenderRequest({ windSpeed: 50, duration: 1, reynolds: -3 });
    expect(r.ok && r.body).toMatchObject({
      wind_speed: 5,
      duration: 5,
      reynolds: 0,
    });
  });

  it('rejects wrong types and unknown models', () => {
    expect(parseRenderRequest(null).ok).toBe(false);
    expect(parseRenderRequest([]).ok).toBe(false);
    expect(parseRenderRequest({ windSpeed: '2' }).ok).toBe(false);
    expect(parseRenderRequest({ vizMode: 1.5 }).ok).toBe(false);
    expect(parseRenderRequest({ model: 'rocket' }).ok).toBe(false);
  });

  it('requires base64 OBJ data for custom models', () => {
    expect(parseRenderRequest({ model: 'custom' }).ok).toBe(false);
    expect(parseRenderRequest({ model: 'custom', objData: 'v 0 0 0' }).ok).toBe(
      false,
    );
    const r = parseRenderRequest({ model: 'custom', objData: 'djAgMCAw' });
    expect(r.ok && r.body.obj_data).toBe('djAgMCAw');
  });

  it('drops OBJ data for built-in models', () => {
    const r = parseRenderRequest({ model: 'car', objData: 'djAgMCAw' });
    expect(r.ok && r.body.obj_data).toBeUndefined();
  });

  it('reports oversized OBJ data as 413', () => {
    const r = parseRenderRequest({
      model: 'custom',
      objData: 'A'.repeat(5 * 1024 * 1024 + 4),
    });
    expect(r).toMatchObject({ ok: false, status: 413 });
  });
});

function post(body: string, ip = '10.0.0.1') {
  return new Request('http://localhost/api/render', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-forwarded-for': ip },
    body,
  });
}

describe('/api/render', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it('reports a missing backend on GET', async () => {
    vi.stubEnv('MODAL_RENDER_ENDPOINT', '');
    const res = await GET(new Request('http://localhost/api/render'));
    expect(await res.json()).toEqual({
      available: false,
      status: 'not_configured',
    });
  });

  it('answers POST with 503 in demo mode', async () => {
    vi.stubEnv('MODAL_RENDER_ENDPOINT', '');
    const res = await POST(post('{}'));
    expect(res.status).toBe(503);
  });

  it('rejects malformed JSON with 400', async () => {
    vi.stubEnv(
      'MODAL_RENDER_ENDPOINT',
      'https://u--fluid-sim-render-endpoint.modal.run',
    );
    const res = await POST(post('{not json', '10.0.0.2'));
    expect(res.status).toBe(400);
  });

  it('forwards a validated body and rate limits repeats', async () => {
    vi.stubEnv(
      'MODAL_RENDER_ENDPOINT',
      'https://u--fluid-sim-render-endpoint.modal.run',
    );
    const fetchMock = vi.fn<typeof fetch>(
      async () => new Response(JSON.stringify({ job_id: 'abc' })),
    );
    vi.stubGlobal('fetch', fetchMock);

    const body = JSON.stringify({ windSpeed: 9, model: 'ahmed25' });
    const res = await POST(post(body, '10.0.0.3'));
    expect(await res.json()).toEqual({ status: 'accepted', jobId: 'abc' });

    const sent = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body));
    expect(sent).toMatchObject({ wind_speed: 5, model: 'ahmed25' });
    expect(typeof sent.job_id).toBe('string');

    const again = await POST(post(body, '10.0.0.3'));
    expect(again.status).toBe(429);
  });
});
