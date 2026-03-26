import { NextResponse } from 'next/server';

export const maxDuration = 30; // Polling endpoints are fast

const RATE_LIMIT = new Map<string, number>();
const RATE_LIMIT_MAX_ENTRIES = 10_000;
const RATE_LIMIT_WINDOW_MS = 60_000;
const MAX_OBJ_SIZE = 5 * 1024 * 1024; // 5 MB base64

function modalUrl(path: string): string | null {
  const endpoint = process.env.MODAL_RENDER_ENDPOINT;
  if (!endpoint) return null;
  // endpoint: https://<user>--fluid-sim-render-endpoint.modal.run
  // job_status: https://<user>--fluid-sim-job-status.modal.run
  return endpoint.replace('render-endpoint', path);
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const jobId = searchParams.get('jobId');

  // If no jobId, return backend health status
  if (!jobId) {
    const healthUrl = modalUrl('health');
    if (!healthUrl) {
      return NextResponse.json({ available: false, status: 'not_configured' });
    }

    try {
      const start = Date.now();
      const res = await fetch(healthUrl, {
        signal: AbortSignal.timeout(8000),
      });
      const latency = Date.now() - start;
      const data = await res.json();

      return NextResponse.json({
        available: true,
        status: latency < 5000 ? 'healthy' : 'slow',
        latency,
        grid: data.grid,
        gpu: data.gpu,
      });
    } catch {
      return NextResponse.json({
        available: true,
        status: 'down',
      });
    }
  }

  // Poll for job status
  const statusUrl = modalUrl('job-status');
  if (!statusUrl) {
    return NextResponse.json(
      { status: 'error', error: 'Backend not configured' },
      { status: 503 },
    );
  }

  try {
    const res = await fetch(`${statusUrl}?job_id=${encodeURIComponent(jobId)}`, {
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) {
      throw new Error(`status ${res.status}`);
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Status poll failed:', error);
    return NextResponse.json(
      { status: 'error', error: 'Failed to check job status' },
      { status: 502 },
    );
  }
}

export async function POST(request: Request) {
  const ip = request.headers.get('x-forwarded-for') || 'unknown';
  const now = Date.now();
  const lastRequest = RATE_LIMIT.get(ip) || 0;

  if (now - lastRequest < RATE_LIMIT_WINDOW_MS) {
    const waitSec = Math.ceil((RATE_LIMIT_WINDOW_MS - (now - lastRequest)) / 1000);
    return NextResponse.json(
      {
        status: 'error',
        error: `Rate limited -- please wait ${waitSec}s before submitting another render.`,
      },
      { status: 429 },
    );
  }
  RATE_LIMIT.set(ip, now);

  if (RATE_LIMIT.size > RATE_LIMIT_MAX_ENTRIES) {
    for (const [key, ts] of RATE_LIMIT) {
      if (now - ts > RATE_LIMIT_WINDOW_MS) RATE_LIMIT.delete(key);
    }
  }

  const params = await request.json();

  const endpoint = process.env.MODAL_RENDER_ENDPOINT;
  if (!endpoint) {
    return NextResponse.json(
      {
        status: 'error',
        error: 'Render backend is not configured. This is a demo instance.',
      },
      { status: 503 },
    );
  }

  if (params.objData && params.objData.length > MAX_OBJ_SIZE) {
    return NextResponse.json(
      { status: 'error', error: 'OBJ file too large (max 5 MB)' },
      { status: 413 },
    );
  }

  try {
    const body: Record<string, unknown> = {
      job_id: `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      wind_speed: params.windSpeed ?? 1.0,
      viz_mode: params.vizMode ?? 1,
      collision_mode: params.collisionMode ?? 2,
      duration: params.duration ?? 5,
      model: params.model ?? 'car',
      reynolds: params.reynolds ?? 0,
    };

    if (params.objData) {
      body.obj_data = params.objData;
    }

    // Fire-and-forget: Modal spawns the render in the background
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(15000),
    });

    const text = await response.text();
    let result;
    try {
      result = JSON.parse(text);
    } catch {
      console.error('Modal returned non-JSON:', text.slice(0, 500));
      return NextResponse.json(
        { status: 'error', error: 'Backend returned invalid response' },
        { status: 502 },
      );
    }

    // Return the job_id so the frontend can poll
    return NextResponse.json({
      status: 'accepted',
      jobId: result.job_id || body.job_id,
    });
  } catch (error) {
    console.error('Modal call failed:', error);
    return NextResponse.json(
      { status: 'error', error: 'Failed to start render' },
      { status: 500 },
    );
  }
}
