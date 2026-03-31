/**
 * Spectral analysis for vortex shedding frequency extraction.
 *
 * Computes the Strouhal number from a Cl time series using FFT:
 *   St = f_peak * L / U
 *
 * The Cl series is sampled every 20 lattice timesteps in the simulation.
 * The lattice velocity is fixed at 0.05.
 */

const LATTICE_VELOCITY = 0.05;
const DEFAULT_SAMPLE_INTERVAL = 10; // lattice timesteps between Cl samples

export interface SpectrumResult {
  /** Strouhal number (dimensionless) */
  strouhal: number;
  /** Frequency axis (in lattice time units) */
  frequencies: number[];
  /** Power spectrum (magnitude squared) */
  power: number[];
  /** Index of the dominant peak in the spectrum */
  peakIndex: number;
}

/**
 * Compute Strouhal number and power spectrum from a Cl time series.
 *
 * @param clSeries  Raw Cl values from the simulation
 * @param charLength  Characteristic length in lattice units
 * @returns Spectrum result, or null if the series is too short
 */
export function computeStrouhal(
  clSeries: number[],
  charLength: number,
  sampleInterval: number = DEFAULT_SAMPLE_INTERVAL,
): SpectrumResult | null {
  if (clSeries.length < 8 || charLength <= 0) return null;

  // Discard initial transient (first 35%), keep the developed flow
  const startIdx = Math.floor(clSeries.length * 0.35);
  const trimmed = clSeries.slice(startIdx);
  if (trimmed.length < 6) return null;

  // Remove mean (DC component)
  const mean = trimmed.reduce((a, b) => a + b, 0) / trimmed.length;
  const centered = trimmed.map((v) => v - mean);

  // Apply Hann window
  const N = centered.length;
  const windowed = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
    windowed[i] = centered[i] * w;
  }

  // Zero-pad to next power of 2
  const nfft = nextPow2(N);
  const re = new Float64Array(nfft);
  const im = new Float64Array(nfft);
  for (let i = 0; i < N; i++) re[i] = windowed[i];

  fft(re, im);

  // Build one-sided power spectrum (skip DC at index 0)
  const nFreqs = Math.floor(nfft / 2);
  const fs = 1 / sampleInterval;
  const frequencies: number[] = [];
  const power: number[] = [];

  for (let k = 1; k < nFreqs; k++) {
    frequencies.push((k * fs) / nfft);
    power.push(re[k] * re[k] + im[k] * im[k]);
  }

  // Find dominant peak
  let peakIndex = 0;
  let peakVal = 0;
  for (let i = 0; i < power.length; i++) {
    if (power[i] > peakVal) {
      peakVal = power[i];
      peakIndex = i;
    }
  }

  const peakFreq = frequencies[peakIndex];
  const strouhal = (peakFreq * charLength) / LATTICE_VELOCITY;

  return { strouhal, frequencies, power, peakIndex };
}

/** Next power of 2 >= n */
function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * In-place Cooley-Tukey radix-2 FFT.
 * Arrays must have length that is a power of 2.
 */
function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n <= 1) return;

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  // Butterfly stages
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let j = 0; j < halfLen; j++) {
        const a = i + j;
        const b = a + halfLen;
        const tRe = curRe * re[b] - curIm * im[b];
        const tIm = curRe * im[b] + curIm * re[b];
        re[b] = re[a] - tRe;
        im[b] = im[a] - tIm;
        re[a] += tRe;
        im[a] += tIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}
