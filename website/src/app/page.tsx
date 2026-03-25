'use client';
import { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import ControlPanel from '../components/ControlPanel';
import VideoPlayer from '../components/VideoPlayer';
import StatusDisplay from '../components/StatusDisplay';
import AboutSection from '../components/AboutSection';
import ResultsPanel, { SimulationResult } from '../components/ResultsPanel';
import { useSurrogate } from '../lib/surrogate';

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
  if (p.has('cm')) result.collisionMode = clamp(parseNum(p.get('cm')!, 1, 10), 0, 2);
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
      duration: 10,
      collisionMode: 1,
      model: 'car',
      reynolds: 0,
    };
    return { ...defaults, ...parseHash() };
  });

  const [status, setStatus] = useState<JobStatus>('idle');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
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
      .then((r) => r.json())
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

  const currentResult = results.length > 0 ? results[results.length - 1] : null;

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

      const response = await fetch('/api/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      const data = await response.json();

      if (data.status === 'complete' && data.video_url) {
        setVideoUrl(data.video_url);
        setStatus('complete');

        const result: SimulationResult = {
          videoUrl: data.video_url,
          cdValue: data.cd_value ?? null,
          clValue: data.cl_value ?? null,
          cdSeries: data.cd_series ?? [],
          clSeries: data.cl_series ?? [],
          charLength: data.char_length ?? null,
          model: params.model,
          windSpeed: params.windSpeed,
          timestamp: Date.now(),
        };
        setResults((prev) => [...prev, result]);
      } else if (data.status === 'error') {
        setError(data.error || 'Render failed');
        setStatus('error');
      } else {
        setError('Unexpected response');
        setStatus('error');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setStatus('error');
    } finally {
      renderStartTime.current = null;
    }
  }, [params, objFile]);

  const selectResult = useCallback((result: SimulationResult) => {
    setVideoUrl(result.videoUrl);
    setStatus('complete');
  }, []);

  return (
    <main className="min-h-screen p-4 md:p-6 lg:p-10 max-w-7xl mx-auto">
      <div className="mb-4 lg:mb-8 flex items-center gap-3 lg:gap-4">
        <img src="/logo.png" alt="Lattice" className="h-12 lg:h-16" />
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
          <AboutSection />
        </div>
        <div className="flex-1 flex flex-col gap-4">
          <VideoPlayer
            videoUrl={videoUrl}
            status={status}
            backendAvailable={backendAvailable}
          />
          <Link
            href="/docs"
            className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle flex items-center justify-between hover:border-ctp-mauve transition-colors"
          >
            <div>
              <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-1">
                White Paper
              </h2>
              <p className="text-xs text-ctp-overlay0">
                LBM theory, compute shader implementation, and drag coefficient
                validation
              </p>
            </div>
            <svg
              className="w-4 h-4 text-ctp-overlay1 shrink-0 ml-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9 5l7 7-7 7"
              />
            </svg>
          </Link>
          <Link
            href="/comparison"
            className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle flex items-center justify-between hover:border-ctp-mauve transition-colors"
          >
            <div>
              <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-1">
                ML vs LBM
              </h2>
              <p className="text-xs text-ctp-overlay0">
                Compare surrogate model predictions against simulation results
              </p>
            </div>
            <svg
              className="w-4 h-4 text-ctp-overlay1 shrink-0 ml-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9 5l7 7-7 7"
              />
            </svg>
          </Link>
          <ResultsPanel
            current={currentResult}
            history={results}
            onSelect={selectResult}
          />
        </div>
      </div>
    </main>
  );
}
