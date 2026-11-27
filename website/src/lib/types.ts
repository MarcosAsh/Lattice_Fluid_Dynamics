export type JobStatus =
  'idle' | 'starting' | 'rendering' | 'complete' | 'error';

export type BackendStatus =
  'checking' | 'healthy' | 'slow' | 'down' | 'not_configured';

export interface SimulationParams {
  windSpeed: number;
  vizMode: number;
  duration: number;
  collisionMode: number;
  model: string;
  reynolds: number;
}

export interface SimulationResult {
  videoUrl: string;
  cdValue: number | null;
  clValue: number | null;
  cdSeries: number[];
  clSeries: number[];
  charLength: number | null;
  sampleInterval: number | null;
  tStar: number | null;
  flowThroughs: number | null;
  cfl: number | null;
  cdPressureSeries: number[];
  cdFrictionSeries: number[];
  model: string;
  windSpeed: number;
  timestamp: number;
  // Missing on runs saved before params were stored
  params?: SimulationParams;
}
