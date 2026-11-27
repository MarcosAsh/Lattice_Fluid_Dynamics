import { useSyncExternalStore } from 'react';

const QUERY = '(prefers-reduced-motion: reduce)';

// jsdom and some old browsers lack matchMedia
function query(): MediaQueryList | null {
  return typeof window.matchMedia === 'function'
    ? window.matchMedia(QUERY)
    : null;
}

function subscribe(cb: () => void) {
  const mql = query();
  mql?.addEventListener('change', cb);
  return () => mql?.removeEventListener('change', cb);
}

export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => query()?.matches ?? false,
    () => false,
  );
}
