'use client';

import './globals.css';

// Replaces the root layout on failure, so it renders its own html and body
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body className="antialiased bg-ctp-base text-ctp-text">
        <main className="min-h-screen flex items-center justify-center p-6">
          <div className="max-w-lg w-full rounded-lg border border-ctp-surface1 bg-ctp-mantle p-6">
            <h1 className="text-xl font-semibold mb-2">Something went wrong</h1>
            <p className="text-ctp-subtext1 mb-4">
              The site failed to load. Try again, and if it keeps happening,
              reload the page.
            </p>
            {error.digest && (
              <pre className="text-xs bg-ctp-crust text-ctp-subtext0 p-3 rounded mb-4 overflow-auto">
                digest: {error.digest}
              </pre>
            )}
            <button
              onClick={reset}
              className="px-4 py-2 rounded bg-ctp-surface1 hover:bg-ctp-surface2 text-ctp-text"
            >
              Try again
            </button>
          </div>
        </main>
      </body>
    </html>
  );
}
