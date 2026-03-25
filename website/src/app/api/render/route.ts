import { NextResponse } from 'next/server';

export const maxDuration = 600; // 10 min -- renders take 3-6 min

const RATE_LIMIT = new Map<string, number>();
const RATE_LIMIT_MAX_ENTRIES = 10_000;
const RATE_LIMIT_WINDOW_MS = 60_000;
const MAX_OBJ_SIZE = 5 * 1024 * 1024; // 5 MB base64

export async function GET() {
  const endpoint = process.env.MODAL_RENDER_ENDPOINT;
  if (!endpoint) {
    return NextResponse.json({ available: false, status: 'not_configured' });
  }

  // Derive the health URL from the render endpoint.
  // Render: https://<user>--fluid-sim-render-endpoint-dev.modal.run
  // Health: https://<user>--fluid-sim-health-dev.modal.run
  const healthUrl = endpoint.replace('render-endpoint', 'health');

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

  // Evict expired entries to prevent unbounded growth
  if (RATE_LIMIT.size > RATE_LIMIT_MAX_ENTRIES) {
    for (const [key, ts] of RATE_LIMIT) {
      if (now - ts > RATE_LIMIT_WINDOW_MS) RATE_LIMIT.delete(key);
    }
  }

  const params = await request.json();

  const modalEndpoint = process.env.MODAL_RENDER_ENDPOINT;

  if (!modalEndpoint) {
    return NextResponse.json(
      {
        status: 'error',
        error: 'Render backend is not configured. This is a demo instance.',
      },
      { status: 503 },
    );
  }

  // Validate custom OBJ size
  if (params.objData && params.objData.length > MAX_OBJ_SIZE) {
    return NextResponse.json(
      { status: 'error', error: 'OBJ file too large (max 5 MB)' },
      { status: 413 },
    );
  }

  try {
    const body: Record<string, unknown> = {
      job_id: `job_${Date.now()}`,
      wind_speed: params.windSpeed ?? 1.0,
      viz_mode: params.vizMode ?? 1,
      collision_mode: params.collisionMode ?? 2,
      duration: params.duration ?? 10,
      model: params.model ?? 'car',
      reynolds: params.reynolds ?? 0,
    };

    if (params.objData) {
      body.obj_data = params.objData;
    }

    const timeoutMs = ((params.duration ?? 10) * 40 + 180) * 1000;
    const response = await fetch(modalEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
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
    return NextResponse.json(result);
  } catch (error) {
    console.error('Modal call failed:', error);
    return NextResponse.json(
      { status: 'error', error: 'Render failed' },
      { status: 500 },
    );
  }
}
