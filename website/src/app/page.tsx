'use client';
import { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import ControlPanel from '../components/ControlPanel';
import VideoPlayer from '../components/VideoPlayer';
import StatusDisplay from '../components/StatusDisplay';
import ResultsPanel, { SimulationResult } from '../components/ResultsPanel';
import { useSurrogate } from '../lib/surrogate';
import { DEMOS, type DemoEntry } from '../lib/demos';

export type JobStatus =
  | 'idle'
  | 'starting'
  | 'rendering'
  | 'complete'
  | 'error';

export interface SimulationParams {
  windSpeed: number;
  vizMode: number;
  duration: number;
  collisionMode: number;
  model: string;
  reynolds: number;
}

const VALID_MODELS = new Set(['car', 'ahmed25', 'ahmed35', 'custom']);

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function parseNum(raw: string, fallback: number, radix?: number): number {
  const v = radix ? parseInt(raw, radix) : parseFloat(raw);
  return Number.isNaN(v) ? fallback : v;
}

function parseHash(): Partial<SimulationParams> {
  if (typeof window === 'undefined') return {};
  const hash = window.location.hash.slice(1);
  if (!hash) return {};
  const p = new URLSearchParams(hash);
  const result: Partial<SimulationParams> = {};
  if (p.has('ws')) result.windSpeed = clamp(parseNum(p.get('ws')!, 1.0), 0, 5);
  if (p.has('vm')) result.vizMode = clamp(parseNum(p.get('vm')!, 1, 10), 0, 6);
  if (p.has('cm')) result.collisionMode = clamp(parseNum(p.get('cm')!, 1, 10), 0, 3);
  if (p.has('d')) result.duration = clamp(parseNum(p.get('d')!, 10, 10), 5, 30);
  if (p.has('m') && VALID_MODELS.has(p.get('m')!)) result.model = p.get('m')!;
  if (p.has('re')) result.reynolds = clamp(parseNum(p.get('re')!, 0), 0, 100000);
  return result;
}

function writeHash(params: SimulationParams) {
  const hash = `ws=${params.windSpeed}&vm=${params.vizMode}&cm=${params.collisionMode}&d=${params.duration}&m=${params.model}&re=${params.reynolds}`;
  window.history.replaceState(null, '', `#${hash}`);
}

export default function Home() {
  const [params, setParams] = useState<SimulationParams>(() => {
    const defaults: SimulationParams = {
      windSpeed: 1.0,
      vizMode: 1,
      duration: 5,
      collisionMode: 2,
      model: 'car',
      reynolds: 0,
    };
    return { ...defaults, ...parseHash() };
  });

  const [status, setStatus] = useState<JobStatus>('idle');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [liveResult, setLiveResult] = useState<SimulationResult | null>(null);
  const [results, setResults] = useState<SimulationResult[]>(() => {
    if (typeof window === 'undefined') return [];
    try {
      const saved = localStorage.getItem('lattice_results');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [backendAvailable, setBackendAvailable] = useState(true);
  const [backendStatus, setBackendStatus] = useState<
    'checking' | 'healthy' | 'slow' | 'down' | 'not_configured'
  >('checking');
  const [objFile, setObjFile] = useState<File | null>(null);
  const renderStartTime = useRef<number | null>(null);
  const { status: mlStatus, predict: mlPredict } = useSurrogate();

  const mlPrediction = mlStatus === 'ready'
    ? mlPredict(params.windSpeed, params.reynolds, params.model)
    : null;

  useEffect(() => {
    fetch('/api/render')
      .then((r) => {
        if (!r.ok) throw new Error(`status ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setBackendAvailable(data.available);
        setBackendStatus(data.status ?? (data.available ? 'healthy' : 'not_configured'));
      })
      .catch(() => {
        setBackendAvailable(false);
        setBackendStatus('down');
      });
  }, []);

  // Persist results to localStorage
  useEffect(() => {
    try {
      // Keep last 50 results max
      const toSave = results.slice(-50);
      localStorage.setItem(
        'lattice_results',
        JSON.stringify(toSave),
      );
    } catch {
      // localStorage full or unavailable
    }
  }, [results]);

  // Write hash on param change
  const initialized = useRef(false);
  useEffect(() => {
    if (!initialized.current) {
      initialized.current = true;
      return;
    }
    writeHash(params);
  }, [params]);

  const currentResult = liveResult ?? (results.length > 0 ? results[results.length - 1] : null);

  const startRender = useCallback(async () => {
    if (params.model === 'custom' && !objFile) {
      setError('Please select an OBJ file first');
      return;
    }

    try {
      setStatus('rendering');
      setError(null);
      setVideoUrl(null);
      renderStartTime.current = Date.now();

      const body: Record<string, unknown> = { ...params };

      if (params.model === 'custom' && objFile) {
        const text = await objFile.text();
        const b64 = btoa(text);
        body.objData = b64;
      }

      // Step 1: Submit render job
      const submitRes = await fetch('/api/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!submitRes.ok) {
        throw new Error(`Server error (${submitRes.status})`);
      }

      const submitData = await submitRes.json();

      if (submitData.status === 'error') {
        throw new Error(submitData.error || 'Failed to start render');
      }

      const jobId = submitData.jobId;
      if (!jobId) {
        throw new Error('No job ID returned');
      }

      // Step 2: Poll for completion
      const maxPollTime = (params.duration ?? 5) * 60 * 1000; // generous limit
      const pollStart = Date.now();

      while (Date.now() - pollStart < maxPollTime) {
        await new Promise((r) => setTimeout(r, 3000));

        const pollRes = await fetch(`/api/render?jobId=${encodeURIComponent(jobId)}`);
        if (!pollRes.ok) continue; // retry on transient errors

        const data = await pollRes.json();

        if (data.status === 'rendering') {
          // Update live chart with partial Cd/Cl data
          if (data.cd_series?.length) {
            setLiveResult({
              videoUrl: '',
              cdValue: null,
              clValue: null,
              cdSeries: data.cd_series,
              clSeries: data.cl_series ?? [],
              charLength: null,
              sampleInterval: null,
              tStar: null,
              flowThroughs: null,
              cfl: null,
              cdPressureSeries: [],
              cdFrictionSeries: [],
              model: params.model,
              windSpeed: params.windSpeed,
              timestamp: Date.now(),
            });
          }
          continue;
        }

        if (data.status === 'complete' && data.video_url) {
          setLiveResult(null);
          setVideoUrl(data.video_url);
          setStatus('complete');

          const result: SimulationResult = {
            videoUrl: data.video_url,
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
            timestamp: Date.now(),
          };
          setResults((prev) => [...prev, result]);
          return;
        }

        if (data.status === 'error') {
          throw new Error(data.error || 'Render failed');
        }
      }

      throw new Error('Render timed out');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setStatus('error');
    } finally {
      renderStartTime.current = null;
    }
  }, [params, objFile]);

  const [sweepProgress, setSweepProgress] = useState<string | null>(null);

  const startSweep = useCallback(async (windMin: number, windMax: number, windStep: number) => {
    const speeds: number[] = [];
    for (let w = windMin; w <= windMax + 0.001; w += windStep) {
      speeds.push(Math.round(w * 10) / 10);
    }
    if (speeds.length === 0) return;

    setError(null);
    setStatus('rendering');

    try {
      for (let i = 0; i < speeds.length; i++) {
        const ws = speeds[i];
        setSweepProgress(`${i + 1}/${speeds.length}: wind ${ws.toFixed(1)} m/s`);

        const sweepParams = { ...params, windSpeed: ws };
        const body: Record<string, unknown> = { ...sweepParams };
        if (params.model === 'custom' && objFile) {
          body.objData = await objFile.text();
        }

        const submitRes = await fetch('/api/render', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (!submitRes.ok) throw new Error(`Sweep submit failed at wind ${ws}`);
        const submitData = await submitRes.json();
        const jobId = submitData.jobId;
        if (!jobId) throw new Error('No job ID');

        const maxPollTime = (params.duration ?? 5) * 60 * 1000;
        const pollStart = Date.now();
        let done = false;

        while (Date.now() - pollStart < maxPollTime) {
          await new Promise((r) => setTimeout(r, 3000));
          const pollRes = await fetch(`/api/render?jobId=${encodeURIComponent(jobId)}`);
          if (!pollRes.ok) continue;
          const data = await pollRes.json();
          if (data.status === 'rendering') continue;
          if (data.status === 'complete' && data.video_url) {
            const result: SimulationResult = {
              videoUrl: data.video_url,
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
              windSpeed: ws,
              timestamp: Date.now(),
            };
            setResults((prev) => [...prev, result]);
            setVideoUrl(data.video_url);
            done = true;
            break;
          }
          if (data.status === 'error') throw new Error(data.error || `Sweep failed at wind ${ws}`);
        }
        if (!done) throw new Error(`Sweep timed out at wind ${ws}`);
      }

      setStatus('complete');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setStatus('error');
    } finally {
      setSweepProgress(null);
    }
  }, [params, objFile]);

  const selectResult = useCallback((result: SimulationResult) => {
    setVideoUrl(result.videoUrl);
    setStatus('complete');
  }, []);

  const loadDemo = useCallback((demo: DemoEntry) => {
    setVideoUrl(demo.result.videoUrl);
    setStatus('complete');
    setError(null);
    const result: SimulationResult = {
      ...demo.result,
      timestamp: Date.now(),
    };
    setResults((prev) => [...prev, result]);
    setParams((prev) => ({ ...prev, ...demo.params }));
  }, []);

  return (
    <main className="min-h-screen p-4 md:p-6 lg:p-10 max-w-7xl mx-auto">
      <div className="mb-4 lg:mb-8 flex items-center gap-3 lg:gap-4">
        <Image src="/logo.png" alt="Lattice" width={64} height={64} className="h-12 lg:h-16 w-auto" />
        <p className="text-xs lg:text-sm text-ctp-subtext0">
          GPU-accelerated wind tunnel simulation{' '}
          <a
            href="https://github.com/MarcosAsh/3dFluidDynamicsInC"
            target="_blank"
            rel="noopener noreferrer"
            className="text-ctp-blue hover:text-ctp-lavender transition-colors"
          >
            (GitHub)
          </a>
        </p>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-ctp-surface0 border border-ctp-red/30 rounded text-ctp-red text-sm">
          {error}
        </div>
      )}

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="w-full lg:w-60 shrink-0 space-y-4">
          <ControlPanel
            params={params}
            setParams={setParams}
            onRender={startRender}
            onSweep={startSweep}
            sweepProgress={sweepProgress}
            disabled={!backendAvailable || status === 'rendering'}
            objFile={objFile}
            onObjFileChange={setObjFile}
            mlPrediction={mlPrediction}
            mlStatus={mlStatus}
          />
          {backendStatus !== 'checking' && (
            <div className="flex items-center gap-2 text-xs px-3 py-2 rounded border border-ctp-surface1 bg-ctp-mantle">
              <div
                className={`w-2 h-2 rounded-full ${
                  backendStatus === 'healthy'
                    ? 'bg-ctp-green'
                    : backendStatus === 'slow'
                      ? 'bg-ctp-yellow'
                      : 'bg-ctp-red'
                }`}
              />
              <span className="text-ctp-subtext0">
                Backend:{' '}
                {backendStatus === 'healthy'
                  ? 'Healthy'
                  : backendStatus === 'slow'
                    ? 'Slow'
                    : backendStatus === 'down'
                      ? 'Down'
                      : 'Not configured'}
              </span>
            </div>
          )}
          <StatusDisplay
            status={status}
            error={error}
            duration={params.duration}
            renderStartTime={renderStartTime.current}
          />
        </div>
        <div className="flex-1 flex flex-col gap-4">
          <VideoPlayer
            videoUrl={videoUrl}
            status={status}
            backendAvailable={backendAvailable}
          />

          <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
            {!videoUrl && status !== 'rendering' && (
              <>
                <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-2">
                  Examples
                </h2>
                <div className="grid grid-cols-3 gap-2 mb-4">
                  {DEMOS.map((demo) => (
                    <button
                      key={demo.params.model}
                      onClick={() => loadDemo(demo)}
                      className="rounded p-2 text-left hover:bg-ctp-surface0 transition-colors"
                    >
                      <div className="text-sm font-medium text-ctp-text">{demo.label}</div>
                      <div className="text-xs text-ctp-overlay0 mt-0.5">{demo.description}</div>
                    </button>
                  ))}
                </div>
              </>
            )}

            <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-2">
              Learn More
            </h2>
            <div className="flex gap-3">
              <Link
                href="/docs"
                className="flex-1 rounded p-2 hover:bg-ctp-surface0 transition-colors"
              >
                <div className="text-sm font-medium text-ctp-text">White Paper</div>
                <div className="text-xs text-ctp-overlay0 mt-0.5">LBM theory, compute shaders, drag validation</div>
              </Link>
              <Link
                href="/comparison"
                className="flex-1 rounded p-2 hover:bg-ctp-surface0 transition-colors"
              >
                <div className="text-sm font-medium text-ctp-text">ML vs LBM</div>
                <div className="text-xs text-ctp-overlay0 mt-0.5">Surrogate model predictions vs simulation</div>
              </Link>
            </div>
          </div>

          <ResultsPanel
            current={currentResult}
            history={results}
            onSelect={selectResult}
          />
        </div>
      </div>

      <footer className="mt-8 py-4 border-t border-ctp-surface1 flex items-center justify-between text-xs text-ctp-overlay0">
        <span>Built by Marcos Ashton, CS at University of Exeter</span>
        <a
          href="https://github.com/MarcosAsh/3dFluidDynamicsInC"
          target="_blank"
          rel="noopener noreferrer"
          className="text-ctp-blue hover:text-ctp-lavender transition-colors"
        >
          Source Code
        </a>
      </footer>
    </main>
  );
}
