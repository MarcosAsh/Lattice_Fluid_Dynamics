import type { Metadata } from 'next';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Page not found',
};

export default function NotFound() {
  return (
    <main className="min-h-screen flex items-center justify-center p-6">
      <div className="max-w-lg w-full rounded-lg border border-ctp-surface1 bg-ctp-mantle p-6 text-ctp-text">
        <h1 className="text-xl font-semibold mb-2">Page not found</h1>
        <p className="text-ctp-subtext1 mb-4">
          There is nothing at this address. It may have moved, or the link may
          be mistyped.
        </p>
        <div className="flex flex-wrap gap-3 text-sm">
          <Link
            href="/"
            className="px-4 py-2 rounded bg-ctp-surface1 hover:bg-ctp-surface2 text-ctp-text"
          >
            Open the simulator
          </Link>
          <Link
            href="/docs"
            className="px-4 py-2 rounded text-ctp-blue hover:text-ctp-lavender"
          >
            Read the docs
          </Link>
        </div>
      </div>
    </main>
  );
}
