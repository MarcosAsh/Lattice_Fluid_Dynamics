'use client';

import { useEffect } from 'react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Route error boundary caught:', error);
  }, [error]);

  const clearAndReload = () => {
    try {
      localStorage.clear();
      sessionStorage.clear();
    } catch (e) {
      console.error('Failed to clear storage:', e);
    }
    window.location.reload();
  };

  return (
    <main className="min-h-screen flex items-center justify-center p-6">
      <div className="max-w-lg w-full rounded-lg border border-ctp-surface1 bg-ctp-mantle p-6 text-ctp-text">
        <h1 className="text-xl font-semibold mb-2">Something went wrong</h1>
        <p className="text-ctp-subtext1 mb-4">
          The page hit an unexpected error. This sometimes happens after an
          update if your browser has stale data from a previous version.
        </p>
        {error.message && (
          <pre className="text-xs bg-ctp-crust text-ctp-subtext0 p-3 rounded mb-4 overflow-auto max-h-32">
            {error.message}
            {error.digest ? `\n\ndigest: ${error.digest}` : ''}
          </pre>
        )}
        <div className="flex gap-2">
          <button
            onClick={reset}
            className="px-4 py-2 rounded bg-ctp-surface1 hover:bg-ctp-surface2 text-ctp-text"
          >
            Try again
          </button>
          <button
            onClick={clearAndReload}
            className="px-4 py-2 rounded bg-ctp-red text-ctp-base hover:opacity-90"
          >
            Clear data and reload
          </button>
        </div>
      </div>
    </main>
  );
}
