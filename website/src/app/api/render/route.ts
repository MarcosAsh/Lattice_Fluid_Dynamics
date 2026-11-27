import { NextResponse } from 'next/server';
import { parseRenderRequest } from '../../../lib/renderRequest';

export const maxDuration = 30; // Polling endpoints are fast

const RATE_LIMIT = new Map<string, number>();
const RATE_LIMIT_MAX_ENTRIES = 10_000;
const RATE_LIMIT_WINDOW_MS = 60_000;

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
      const data = (await res.json()) as { grid?: unknown; gpu?: unknown };

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
    const res = await fetch(
      `${statusUrl}?job_id=${encodeURIComponent(jobId)}`,
      {
        signal: AbortSignal.timeout(10000),
      },
    );
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

function clientIp(request: Request): string {
  const forwarded = request.headers.get('x-forwarded-for');
  return forwarded?.split(',')[0]?.trim() || 'unknown';
}

function rateLimited(ip: string, now: number): number | null {
  const lastRequest = RATE_LIMIT.get(ip) || 0;
  if (now - lastRequest < RATE_LIMIT_WINDOW_MS) {
    return Math.ceil((RATE_LIMIT_WINDOW_MS - (now - lastRequest)) / 1000);
  }
  RATE_LIMIT.set(ip, now);

  if (RATE_LIMIT.size > RATE_LIMIT_MAX_ENTRIES) {
    for (const [key, ts] of RATE_LIMIT) {
      if (now - ts > RATE_LIMIT_WINDOW_MS) RATE_LIMIT.delete(key);
    }
    // Hard cap: if cleanup didn't shrink enough, drop oldest entries
    if (RATE_LIMIT.size > RATE_LIMIT_MAX_ENTRIES) {
      const excess = RATE_LIMIT.size - RATE_LIMIT_MAX_ENTRIES;
      let dropped = 0;
      for (const key of RATE_LIMIT.keys()) {
        if (dropped >= excess) break;
        RATE_LIMIT.delete(key);
        dropped++;
      }
    }
  }
  return null;
}

function errorResponse(error: string, status: number) {
  return NextResponse.json({ status: 'error', error }, { status });
}

export async function POST(request: Request) {
  const endpoint = process.env.MODAL_RENDER_ENDPOINT;
  if (!endpoint) {
    return errorResponse(
      'Render backend is not configured. This is a demo instance.',
      503,
    );
  }

  let input: unknown;
  try {
    input = await request.json();
  } catch {
    return errorResponse('Request body must be JSON', 400);
  }
  const parsed = parseRenderRequest(input);
  if (!parsed.ok) return errorResponse(parsed.error, parsed.status);

  const waitSec = rateLimited(clientIp(request), Date.now());
  if (waitSec !== null) {
    return errorResponse(
      `Rate limited. Please wait ${waitSec}s before submitting another render.`,
      429,
    );
  }

  const jobId = `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  try {
    // Fire-and-forget: Modal spawns the render in the background
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId, ...parsed.body }),
      signal: AbortSignal.timeout(15000),
    });

    const text = await response.text();
    let result: { job_id?: unknown };
    try {
      result = JSON.parse(text);
    } catch {
      console.error('Modal returned non-JSON:', text.slice(0, 500));
      return errorResponse('Backend returned invalid response', 502);
    }

    // Return the job_id so the frontend can poll
    return NextResponse.json({
      status: 'accepted',
      jobId: typeof result.job_id === 'string' ? result.job_id : jobId,
    });
  } catch (error) {
    console.error('Modal call failed:', error);
    return errorResponse('Failed to start render', 500);
  }
}
