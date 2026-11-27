'use client';
import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import ControlPanel from '../components/ControlPanel';
import VideoPlayer from '../components/VideoPlayer';
import StatusDisplay from '../components/StatusDisplay';
import ResultsPanel from '../components/ResultsPanel';
import { useSurrogate } from '../lib/anyobj_surrogate';
import { DEMOS, type DemoEntry } from '../lib/demos';
import { useHashParams } from '../lib/hashParams';
import { encodeObjFile, runRenderJob, sweepSpeeds } from '../lib/renderJob';
import { addRun, useRuns } from '../lib/runs';
import type { JobStatus, SimulationResult } from '../lib/types';

export default function Home() {
  const [params, setParams] = useHashParams();
  const results = useRuns();

  const [status, setStatus] = useState<JobStatus>('idle');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [liveResult, setLiveResult] = useState<SimulationResult | null>(null);
  const [backendAvailable, setBackendAvailable] = useState(true);
  const [backendStatus, setBackendStatus] = useState<
    'checking' | 'healthy' | 'slow' | 'down' | 'not_configured'
  >('checking');
  const [objFile, setObjFile] = useState<File | null>(null);
  const [renderStartTime, setRenderStartTime] = useState<number | null>(null);
  const { status: mlStatus, predict: mlPredict } = useSurrogate();

  const mlPrediction =
    mlStatus === 'ready'
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
        setBackendStatus(
          data.status ?? (data.available ? 'healthy' : 'not_configured'),
        );
      })
      .catch(() => {
        setBackendAvailable(false);
        setBackendStatus('down');
      });
  }, []);

  const currentResult =
    liveResult ?? (results.length > 0 ? results[results.length - 1] : null);

  const startRender = useCallback(async () => {
    if (params.model === 'custom' && !objFile) {
      setError('Please select an OBJ file first');
      return;
    }

    setStatus('rendering');
    setError(null);
    setVideoUrl(null);
    setRenderStartTime(Date.now());
    try {
      const objData =
        params.model === 'custom' && objFile
          ? await encodeObjFile(objFile)
          : undefined;
      const result = await runRenderJob(params, {
        objData,
        onPartial: setLiveResult,
      });
      addRun(result);
      setVideoUrl(result.videoUrl);
      setStatus('complete');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setStatus('error');
    } finally {
      setLiveResult(null);
      setRenderStartTime(null);
    }
  }, [params, objFile]);

  const [sweepProgress, setSweepProgress] = useState<string | null>(null);

  const startSweep = useCallback(
    async (windMin: number, windMax: number, windStep: number) => {
      const speeds = sweepSpeeds(windMin, windMax, windStep);
      if (speeds.length === 0) return;

      setError(null);
      setStatus('rendering');
      try {
        const objData =
          params.model === 'custom' && objFile
            ? await encodeObjFile(objFile)
            : undefined;
        for (const [i, ws] of speeds.entries()) {
          setSweepProgress(
            `${i + 1}/${speeds.length}: wind ${ws.toFixed(1)} m/s`,
          );
          let result: SimulationResult;
          try {
            result = await runRenderJob(
              { ...params, windSpeed: ws },
              { objData },
            );
          } catch (err) {
            const msg = err instanceof Error ? err.message : 'Unknown error';
            throw new Error(
              `Sweep stopped at wind ${ws.toFixed(1)} m/s: ${msg}`,
            );
          }
          addRun(result);
          setVideoUrl(result.videoUrl);
        }
        setStatus('complete');
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        setStatus('error');
      } finally {
        setSweepProgress(null);
      }
    },
    [params, objFile],
  );

  const selectResult = useCallback((result: SimulationResult) => {
    setVideoUrl(result.videoUrl);
    setStatus('complete');
  }, []);

  const loadDemo = useCallback(
    (demo: DemoEntry) => {
      const demoParams = { ...params, ...demo.params };
      setVideoUrl(demo.result.videoUrl);
      setStatus('complete');
      setError(null);
      addRun({ ...demo.result, timestamp: Date.now(), params: demoParams });
      setParams(demoParams);
    },
    [params, setParams],
  );

  return (
    <main className="min-h-screen p-4 md:p-6 lg:p-10 max-w-7xl mx-auto">
      <div className="mb-4 lg:mb-8 flex items-center gap-3 lg:gap-4">
        <Image
          src="/logo.png"
          alt="Lattice"
          width={64}
          height={64}
          className="h-12 lg:h-16 w-auto"
        />
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
            renderStartTime={renderStartTime}
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
                <div
                  className="grid grid-cols-3 gap-2 mb-4"
                  data-testid="demo-gallery"
                >
                  {DEMOS.map((demo) => (
                    <button
                      key={demo.params.model}
                      onClick={() => loadDemo(demo)}
                      className="rounded p-2 text-left hover:bg-ctp-surface0 transition-colors"
                    >
                      <div className="text-sm font-medium text-ctp-text">
                        {demo.label}
                      </div>
                      <div className="text-xs text-ctp-overlay0 mt-0.5">
                        {demo.description}
                      </div>
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
                <div className="text-sm font-medium text-ctp-text">
                  White Paper
                </div>
                <div className="text-xs text-ctp-overlay0 mt-0.5">
                  LBM theory, compute shaders, drag validation
                </div>
              </Link>
              <Link
                href="/comparison"
                className="flex-1 rounded p-2 hover:bg-ctp-surface0 transition-colors"
              >
                <div className="text-sm font-medium text-ctp-text">
                  ML vs LBM
                </div>
                <div className="text-xs text-ctp-overlay0 mt-0.5">
                  Surrogate model predictions vs simulation
                </div>
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
