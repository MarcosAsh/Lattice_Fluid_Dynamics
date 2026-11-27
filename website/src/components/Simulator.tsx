'use client';

import { useCallback, useState, type ReactNode } from 'react';
import ControlPanel from './ControlPanel';
import VideoPlayer from './VideoPlayer';
import StatusDisplay from './StatusDisplay';
import ResultsPanel from './ResultsPanel';
import { useSurrogate } from '../lib/anyobj_surrogate';
import { useBackendStatus } from '../lib/backend';
import { DEMOS, type DemoEntry } from '../lib/demos';
import { useHashParams } from '../lib/hashParams';
import { encodeObjFile, runRenderJob, sweepSpeeds } from '../lib/renderJob';
import { addRun, useRuns } from '../lib/runs';
import type { BackendStatus, JobStatus, SimulationResult } from '../lib/types';

const BACKEND_LABELS: Record<BackendStatus, string> = {
  checking: 'Checking',
  healthy: 'Healthy',
  slow: 'Slow',
  down: 'Down',
  not_configured: 'Not configured',
};

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : 'Unknown error';
}

export default function Simulator({ learnMore }: { learnMore: ReactNode }) {
  const [params, setParams] = useHashParams();
  const results = useRuns();
  const backend = useBackendStatus();
  const { status: mlStatus, predict: mlPredict } = useSurrogate();

  const [status, setStatus] = useState<JobStatus>('idle');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [liveResult, setLiveResult] = useState<SimulationResult | null>(null);
  const [objFile, setObjFile] = useState<File | null>(null);
  const [renderStartTime, setRenderStartTime] = useState<number | null>(null);
  const [sweepProgress, setSweepProgress] = useState<string | null>(null);

  const mlPrediction =
    mlStatus === 'ready'
      ? mlPredict(params.windSpeed, params.reynolds, params.model)
      : null;

  const currentResult = liveResult ?? results[results.length - 1] ?? null;

  const showRun = useCallback((run: SimulationResult) => {
    setVideoUrl(run.videoUrl || null);
    setStatus('complete');
  }, []);

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
      showRun(addRun(result));
    } catch (err) {
      setError(errorMessage(err));
      setStatus('error');
    } finally {
      setLiveResult(null);
      setRenderStartTime(null);
    }
  }, [params, objFile, showRun]);

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
            throw new Error(
              `Sweep stopped at wind ${ws.toFixed(1)} m/s: ${errorMessage(err)}`,
            );
          }
          addRun(result);
          setVideoUrl(result.videoUrl || null);
        }
        setStatus('complete');
      } catch (err) {
        setError(errorMessage(err));
        setStatus('error');
      } finally {
        setSweepProgress(null);
      }
    },
    [params, objFile],
  );

  const loadDemo = useCallback(
    (demo: DemoEntry) => {
      const demoParams = { ...params, ...demo.params };
      setError(null);
      setParams(demoParams);
      showRun(
        addRun({ ...demo.result, timestamp: Date.now(), params: demoParams }),
      );
    },
    [params, setParams, showRun],
  );

  const rendering = status === 'rendering';
  const renderHint =
    backend.status === 'not_configured'
      ? 'No GPU backend is connected, so rendering is off. Load an example to see results.'
      : !backend.available && backend.status === 'down'
        ? 'Could not reach the server, so rendering is off.'
        : null;

  return (
    <>
      {error && (
        <div
          role="alert"
          className="mb-4 p-3 bg-ctp-surface0 border border-ctp-red/30 rounded text-ctp-red text-sm"
        >
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
            disabled={rendering}
            canRender={backend.available}
            renderHint={renderHint}
            objFile={objFile}
            onObjFileChange={setObjFile}
            mlPrediction={mlPrediction}
            mlStatus={mlStatus}
          />
          {backend.status !== 'checking' && (
            <div className="flex items-center gap-2 text-xs px-3 py-2 rounded border border-ctp-surface1 bg-ctp-mantle">
              <div
                aria-hidden="true"
                className={`w-2 h-2 rounded-full ${
                  backend.status === 'healthy'
                    ? 'bg-ctp-green'
                    : backend.status === 'slow'
                      ? 'bg-ctp-yellow'
                      : 'bg-ctp-red'
                }`}
              />
              <span className="text-ctp-subtext0">
                Backend: {BACKEND_LABELS[backend.status]}
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
        <div className="flex-1 min-w-0 flex flex-col gap-4">
          <VideoPlayer
            videoUrl={videoUrl}
            status={status}
            backendStatus={backend.status}
          />

          <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
            {!rendering && (
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
            {learnMore}
          </div>

          <ResultsPanel
            current={currentResult}
            history={results}
            onSelect={showRun}
          />
        </div>
      </div>
    </>
  );
}
