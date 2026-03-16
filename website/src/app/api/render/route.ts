import { NextResponse } from 'next/server';

const RATE_LIMIT = new Map<string, number>();
const MAX_OBJ_SIZE = 5 * 1024 * 1024; // 5 MB base64

export async function GET() {
  const endpoint = process.env.MODAL_RENDER_ENDPOINT;
  if (!endpoint) {
    return NextResponse.json({ available: false, status: 'not_configured' });
  }

  // Derive the health URL from the render endpoint.
  // Render: https://<user>--fluid-sim-render-endpoint-dev.modal.run
  // Health: https://<user>--fluid-sim-health-dev.modal.run
  const healthUrl = endpoint.replace(
    /render-endpoint(-\w+)?\.modal\.run/,
    'health$1.modal.run',
  );

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

  if (now - lastRequest < 60000) {
    return NextResponse.json({ error: 'Rate limited' }, { status: 429 });
  }
  RATE_LIMIT.set(ip, now);

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
      collision_mode: params.collisionMode ?? 1,
      duration: params.duration ?? 10,
      model: params.model ?? 'car',
      reynolds: params.reynolds ?? 0,
    };

    if (params.objData) {
      body.obj_data = params.objData;
    }

    const response = await fetch(modalEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const result = await response.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Modal call failed:', error);
    return NextResponse.json(
      { status: 'error', error: 'Render failed' },
      { status: 500 },
    );
  }
}
