'use client';

import { useState, useEffect } from 'react';
import { JobStatus } from '../app/page';

/* eslint-disable react-hooks/set-state-in-effect */
function useTickElapsed(active: boolean, key: number | null): number {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    setTick(0);
    if (!active) return;
    const id = setInterval(() => setTick((t) => t + 1), 500);
    return () => clearInterval(id);
  }, [active, key]);

  return active ? tick * 0.5 : 0;
}
/* eslint-enable react-hooks/set-state-in-effect */

interface StatusDisplayProps {
  status: JobStatus;
  error: string | null;
  duration: number;
  renderStartTime: number | null;
}

interface Phase {
  label: string;
  hint: string;
}

function getPhase(
  elapsed: number,
  duration: number,
): { phase: Phase; progress: number } {
  const startupTime = 15;
  const simTime = duration * 1.5;
  const encodeTime = 10;
  const total = startupTime + simTime + encodeTime;

  const progress = Math.min(elapsed / total, 0.95);

  if (elapsed < startupTime) {
    return {
      phase: {
        label: 'Starting GPU instance',
        hint: 'Cold starting the container...',
      },
      progress,
    };
  }
  if (elapsed < startupTime + simTime) {
    return {
      phase: {
        label: 'Running simulation',
        hint: `Computing ${duration}s of fluid dynamics`,
      },
      progress,
    };
  }
  return {
    phase: { label: 'Encoding video', hint: 'Packaging frames into MP4' },
    progress,
  };
}

export default function StatusDisplay({
  status,
  error,
  duration,
  renderStartTime,
}: StatusDisplayProps) {
  const isRendering = status === 'rendering' && renderStartTime != null;
  const elapsed = useTickElapsed(isRendering, renderStartTime);

  const { phase, progress } = getPhase(elapsed, duration);

  const statusConfig = {
    idle: { color: 'bg-ctp-overlay0', text: 'Ready' },
    starting: { color: 'bg-ctp-yellow', text: 'Starting...' },
    rendering: { color: 'bg-ctp-blue', text: phase.label },
    complete: { color: 'bg-ctp-green', text: 'Complete!' },
    error: { color: 'bg-ctp-red', text: 'Error' },
  };

  const config = statusConfig[status];

  return (
    <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
      <div className="flex items-center gap-3">
        <div
          className={`w-2.5 h-2.5 rounded-full ${config.color} ${
            isRendering ? 'animate-pulse' : ''
          }`}
        />
        <span className="text-sm text-ctp-text">{config.text}</span>
      </div>

      {isRendering && (
        <div className="mt-3">
          <div className="h-1.5 bg-ctp-surface0 rounded-full overflow-hidden">
            <div
              className="h-full bg-ctp-mauve rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
          <p className="text-[10px] text-ctp-overlay0 mt-1.5">{phase.hint}</p>
        </div>
      )}

      {error && (
        <div className="mt-3 p-2 bg-ctp-surface0 border border-ctp-red/30 rounded text-ctp-red text-xs">
          {error}
        </div>
      )}
    </div>
  );
}
