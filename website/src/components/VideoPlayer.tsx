'use client';

import { useEffect, useRef, useState } from 'react';
import { JobStatus } from '../app/page';

interface VideoPlayerProps {
  videoUrl: string | null;
  status: JobStatus;
  backendAvailable: boolean;
}

export default function VideoPlayer({
  videoUrl,
  status,
  backendAvailable,
}: VideoPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [demoAvailable, setDemoAvailable] = useState(true);

  useEffect(() => {
    const onChange = () => {
      setIsFullscreen(document.fullscreenElement === containerRef.current);
    };
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  const toggleFullscreen = async () => {
    const el = containerRef.current;
    if (!el) return;

    try {
      if (document.fullscreenElement === el) {
        await document.exitFullscreen();
      } else {
        await el.requestFullscreen();
      }
    } catch (err) {
      console.error('Fullscreen toggle failed', err);
    }
  };

  const showDemo = !videoUrl && status === 'idle' && demoAvailable;
  const showPlaceholder = !videoUrl && !showDemo;

  return (
    <div
      ref={containerRef}
      className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle"
    >
      <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-3">
        Preview
      </h2>

      <div className="aspect-video bg-ctp-crust rounded-lg overflow-hidden flex items-center justify-center border border-ctp-surface1 relative">
        {videoUrl ? (
          <video
            src={videoUrl}
            controls
            autoPlay
            loop
            className="w-full h-full object-contain"
          />
        ) : showDemo ? (
          <>
            <video
              src="/demo.mp4"
              autoPlay
              loop
              muted
              playsInline
              onError={() => setDemoAvailable(false)}
              className="w-full h-full object-contain"
            />
            <div className="absolute bottom-2 right-2 bg-ctp-crust/80 text-ctp-overlay0 text-[10px] px-2 py-0.5 rounded">
              Demo
            </div>
          </>
        ) : showPlaceholder ? (
          <div className="text-center text-ctp-overlay0 text-sm px-6">
            {!backendAvailable && status === 'idle' && (
              <>
                <p className="text-ctp-subtext0 mb-2">Demo Mode</p>
                <p className="text-xs text-ctp-overlay0 leading-relaxed">
                  The GPU render backend is not connected. You can explore the
                  controls and read the white paper. To run live simulations,
                  configure the MODAL_RENDER_ENDPOINT environment variable.
                </p>
              </>
            )}
            {backendAvailable && status === 'idle' && (
              <p>Configure parameters and click Start Render</p>
            )}
            {status === 'rendering' && (
              <>
                <p className="animate-pulse">Rendering simulation...</p>
                <p className="text-xs text-ctp-overlay0 mt-1">
                  This may take a minute
                </p>
              </>
            )}
            {status === 'error' && (
              <p className="text-ctp-red">Render failed</p>
            )}
          </div>
        ) : null}
      </div>

      {videoUrl && (
        <div className="mt-3 flex gap-2">
          <a
            href={videoUrl}
            download
            className="bg-ctp-mauve hover:bg-ctp-lavender text-ctp-crust text-xs font-medium py-1.5 px-3 rounded transition-colors"
          >
            Download
          </a>
          <button
            onClick={() => navigator.clipboard.writeText(videoUrl)}
            className="bg-ctp-surface0 hover:bg-ctp-surface1 text-ctp-text text-xs font-medium py-1.5 px-3 rounded border border-ctp-surface1 transition-colors"
          >
            Copy URL
          </button>
          <button
            onClick={toggleFullscreen}
            className="bg-ctp-surface0 hover:bg-ctp-surface1 text-ctp-text text-xs font-medium py-1.5 px-3 rounded border border-ctp-surface1 transition-colors"
          >
            {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          </button>
        </div>
      )}
    </div>
  );
}
