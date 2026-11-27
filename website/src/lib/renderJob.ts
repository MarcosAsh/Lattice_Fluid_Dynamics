import type { SimulationParams, SimulationResult } from './types';

// Poll response, passed through from the Modal worker
export interface JobStatusResponse {
  status?: string;
  error?: string;
  video_url?: string;
  cd_value?: number | null;
  cl_value?: number | null;
  cd_series?: number[];
  cl_series?: number[];
  char_length?: number | null;
  sample_interval?: number | null;
  t_star?: number | null;
  flow_throughs?: number | null;
  cfl?: number | null;
  cd_pressure_series?: number[];
  cd_friction_series?: number[];
}

interface SubmitResponse {
  status?: string;
  error?: string;
  jobId?: string;
}

export interface RunJobOptions {
  objData?: string;
  onPartial?: (partial: SimulationResult) => void;
  pollIntervalMs?: number;
  fetchImpl?: typeof fetch;
  sleep?: (ms: number) => Promise<void>;
  now?: () => number;
}

const defaultSleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

// Base64 of the raw file bytes, safe for non-ASCII content
export async function encodeObjFile(file: Blob): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

export function resultFromResponse(
  data: JobStatusResponse,
  params: SimulationParams,
  timestamp: number,
): SimulationResult {
  return {
    videoUrl: data.video_url ?? '',
    cdValue: data.cd_value ?? null,
    clValue: data.cl_value ?? null,
    cdSeries: data.cd_series ?? [],
    clSeries: data.cl_series ?? [],
    charLength: data.char_length ?? null,
    sampleInterval: data.sample_interval ?? null,
    tStar: data.t_star ?? null,
    flowThroughs: data.flow_throughs ?? null,
    cfl: data.cfl ?? null,
    cdPressureSeries: data.cd_pressure_series ?? [],
    cdFrictionSeries: data.cd_friction_series ?? [],
    model: params.model,
    windSpeed: params.windSpeed,
    timestamp,
    params,
  };
}

// Submit one render and poll until it ends
export async function runRenderJob(
  params: SimulationParams,
  options: RunJobOptions = {},
): Promise<SimulationResult> {
  const {
    objData,
    onPartial,
    pollIntervalMs = 3000,
    fetchImpl = fetch,
    sleep = defaultSleep,
    now = Date.now,
  } = options;

  const submitRes = await fetchImpl('/api/render', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(objData ? { ...params, objData } : params),
  });
  const submitData = (await submitRes
    .json()
    .catch(() => ({}))) as SubmitResponse;
  if (!submitRes.ok || submitData.status === 'error') {
    throw new Error(submitData.error ?? `Server error (${submitRes.status})`);
  }
  const jobId = submitData.jobId;
  if (!jobId) throw new Error('No job ID returned');

  // Generous limit, a 5 s clip gets 5 minutes
  const maxPollTime = params.duration * 60 * 1000;
  const pollStart = now();

  while (now() - pollStart < maxPollTime) {
    await sleep(pollIntervalMs);

    const pollRes = await fetchImpl(
      `/api/render?jobId=${encodeURIComponent(jobId)}`,
    );
    // Retry on transient errors
    if (!pollRes.ok) continue;

    const data = (await pollRes.json()) as JobStatusResponse;

    if (data.status === 'rendering') {
      if (onPartial && data.cd_series?.length) {
        onPartial(resultFromResponse(data, params, now()));
      }
      continue;
    }
    if (data.status === 'complete' && data.video_url) {
      return resultFromResponse(data, params, now());
    }
    if (data.status === 'error') {
      throw new Error(data.error || 'Render failed');
    }
  }

  throw new Error('Render timed out');
}

export const MAX_SWEEP_RUNS = 50;

// Wind speeds for a sweep, rounded to 0.1 m/s
export function sweepSpeeds(min: number, max: number, step: number): number[] {
  if (![min, max, step].every(Number.isFinite) || step <= 0 || min > max) {
    return [];
  }
  const speeds: number[] = [];
  for (
    let w = min;
    w <= max + 0.001 && speeds.length < MAX_SWEEP_RUNS;
    w += step
  ) {
    speeds.push(Math.round(w * 10) / 10);
  }
  return speeds;
}
