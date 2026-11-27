import { useEffect, useState } from 'react';
import type { BackendStatus } from './types';

interface HealthResponse {
  available: boolean;
  status?: BackendStatus;
}

export interface BackendState {
  status: BackendStatus;
  available: boolean;
}

// Fetched on the client so the page can stay static
export function useBackendStatus(): BackendState {
  const [state, setState] = useState<BackendState>({
    status: 'checking',
    available: false,
  });

  useEffect(() => {
    const ctrl = new AbortController();
    fetch('/api/render', { signal: ctrl.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`status ${r.status}`);
        return r.json() as Promise<HealthResponse>;
      })
      .then((data) => {
        setState({
          available: data.available,
          status:
            data.status ?? (data.available ? 'healthy' : 'not_configured'),
        });
      })
      .catch(() => {
        if (!ctrl.signal.aborted)
          setState({ available: false, status: 'down' });
      });
    return () => ctrl.abort();
  }, []);

  return state;
}
