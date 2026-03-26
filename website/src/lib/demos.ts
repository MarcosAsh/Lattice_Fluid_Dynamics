import type { SimulationResult } from '../components/ResultsPanel';

export interface DemoEntry {
  label: string;
  description: string;
  params: { model: string; windSpeed: number; duration: number };
  result: SimulationResult;
}

const S3 = 'https://fluid-sim-renders.s3.eu-west-2.amazonaws.com/renders';

export const DEMOS: DemoEntry[] = [
  {
    label: 'Car',
    description: 'Generic car at 1.0 m/s',
    params: { model: 'car', windSpeed: 1.0, duration: 5 },
    result: {
      videoUrl: `${S3}/demo_car.mp4`,
      cdValue: 0.7772,
      clValue: 0.465,
      cdSeries: [1.946,1.063,3.889,6.249,0.04,0.197,10.777,0.508,4.939,1.095,5.407,0.925,0.838,1.519,0.023,0.581],
      clSeries: [0.243,0.042,0.122,0.505,0.362,0.258,1.004,0.178,0.239,0.398,0.499,0.492,0.681,0.291,0.228,0.633],
      charLength: 32.0,
      model: 'car',
      windSpeed: 1.0,
      timestamp: 0,
    },
  },
  {
    label: 'Ahmed 25\u00b0',
    description: 'Ahmed body, 25\u00b0 slant',
    params: { model: 'ahmed25', windSpeed: 1.0, duration: 5 },
    result: {
      videoUrl: `${S3}/demo_ahmed25.mp4`,
      cdValue: 2.3002,
      clValue: 0.32,
      cdSeries: [3.779,0.452,11.024,15.884,2.283,2.996,24.54,2.915,12.28,2.944,11.711,1.83,1.31,4.051,3.095,1.215],
      clSeries: [0.296,0.403,0.077,0.015,0.181,0.12,0.25,0.499,0.185,0.311,0.716,0.074,0.271,0.467,0.644,0.144],
      charLength: 32.0,
      model: 'ahmed25',
      windSpeed: 1.0,
      timestamp: 0,
    },
  },
  {
    label: 'Ahmed 35\u00b0',
    description: 'Ahmed body, 35\u00b0 slant',
    params: { model: 'ahmed35', windSpeed: 1.0, duration: 5 },
    result: {
      videoUrl: `${S3}/demo_ahmed35.mp4`,
      cdValue: 2.6058,
      clValue: 0.3106,
      cdSeries: [3.911,0.386,10.672,16.019,2.937,2.847,24.755,3.046,13.362,1.658,11.375,1.928,1.755,4.445,3.501,1.4],
      clSeries: [0.074,0.001,0.031,0.053,0.667,0.234,0.182,0.573,0.167,0.275,0.253,0.206,0.603,0.08,0.337,0.327],
      charLength: 32.0,
      model: 'ahmed35',
      windSpeed: 1.0,
      timestamp: 0,
    },
  },
];
