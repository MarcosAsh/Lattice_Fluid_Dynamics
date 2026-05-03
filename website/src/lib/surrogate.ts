// Pure TypeScript inference for the Cd/Cl surrogate model.
//
// Architecture (matches ml/train.cpp):
//   Linear(3,256) -> SwiGLU(256,512) -> Linear(256,128)
//   -> SwiGLU(128,256) -> Linear(128,2)
//
// Loads model.bin + model_norm.bin from /models/ and runs the forward
// pass entirely in the browser. ~525K params, runs in < 1ms.

'use client';

import { useState, useEffect, useRef, useCallback } from 'react';

// Binary format constants (must match ml/framework/src/weights_io.cpp)

const MAGIC = 0x4c545753;
const VERSION = 1;
const NUM_PARAMS = 12;

// Model dimensions
const IN = 3;
const H1 = 256;
const HID1 = 512;
const H2 = 128;
const HID2 = 256;
const OUT = 2;

// Model ID mapping (must match ml/data_gen.py)
const MODEL_IDS: Record<string, number> = {
  car: 0,
  ahmed25: 1,
  ahmed35: 2,
};

// Math helpers

function matvec(W: Float32Array, x: Float32Array, rows: number, cols: number): Float32Array {
  const y = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    const off = r * cols;
    for (let c = 0; c < cols; c++) {
      sum += W[off + c] * x[c];
    }
    y[r] = sum;
  }
  return y;
}

function addBias(y: Float32Array, b: Float32Array): void {
  for (let i = 0; i < y.length; i++) y[i] += b[i];
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function swigluForward(
  Wg: Float32Array,
  Wu: Float32Array,
  Wd: Float32Array,
  x: Float32Array,
  embedDim: number,
  hiddenDim: number,
): Float32Array {
  const gate = matvec(Wg, x, hiddenDim, embedDim);
  const up = matvec(Wu, x, hiddenDim, embedDim);

  // swish(gate) * up
  for (let i = 0; i < hiddenDim; i++) {
    gate[i] = gate[i] * sigmoid(gate[i]) * up[i];
  }

  return matvec(Wd, gate, embedDim, hiddenDim);
}

// Weight loading

interface Weights {
  fc1W: Float32Array;
  fc1b: Float32Array;
  act1Wg: Float32Array;
  act1Wu: Float32Array;
  act1Wd: Float32Array;
  fc2W: Float32Array;
  fc2b: Float32Array;
  act2Wg: Float32Array;
  act2Wu: Float32Array;
  act2Wd: Float32Array;
  fc3W: Float32Array;
  fc3b: Float32Array;
}

interface Normalizer {
  mean: Float32Array;
  stdDev: Float32Array;
}

function parseWeights(buf: ArrayBuffer): Weights {
  const view = new DataView(buf);
  let off = 0;

  const magic = view.getUint32(off, true);
  off += 4;
  if (magic !== MAGIC) throw new Error(`Bad magic: 0x${magic.toString(16)}`);

  const version = view.getUint32(off, true);
  off += 4;
  if (version !== VERSION) throw new Error(`Bad version: ${version}`);

  const count = view.getUint32(off, true);
  off += 4;
  if (count !== NUM_PARAMS) throw new Error(`Expected ${NUM_PARAMS} params, got ${count}`);

  function readParam(): Float32Array {
    if (off + 4 > buf.byteLength) throw new Error('Truncated weights file');
    const ndim = view.getUint32(off, true);
    off += 4;
    let numel = 1;
    for (let d = 0; d < ndim; d++) {
      if (off + 4 > buf.byteLength) throw new Error('Truncated weights file');
      numel *= view.getUint32(off, true);
      off += 4;
    }
    if (off + numel * 4 > buf.byteLength) throw new Error('Truncated weights file');
    const data = new Float32Array(buf, off, numel);
    off += numel * 4;
    return new Float32Array(data); // copy so we own the buffer
  }

  return {
    fc1W: readParam(),
    fc1b: readParam(),
    act1Wg: readParam(),
    act1Wu: readParam(),
    act1Wd: readParam(),
    fc2W: readParam(),
    fc2b: readParam(),
    act2Wg: readParam(),
    act2Wu: readParam(),
    act2Wd: readParam(),
    fc3W: readParam(),
    fc3b: readParam(),
  };
}

function parseNorm(buf: ArrayBuffer): Normalizer {
  const f = new Float32Array(buf);
  return {
    mean: new Float32Array(f.buffer, 0, 3),
    stdDev: new Float32Array(f.buffer, 12, 3),
  };
}

// Forward pass

export interface Prediction {
  cd: number;
  cl: number;
}

function forward(w: Weights, norm: Normalizer, windSpeed: number, reynolds: number, modelId: number): Prediction {
  // Z-score normalize inputs
  const x = new Float32Array(IN);
  x[0] = (windSpeed - norm.mean[0]) / norm.stdDev[0];
  x[1] = (reynolds - norm.mean[1]) / norm.stdDev[1];
  x[2] = (modelId - norm.mean[2]) / norm.stdDev[2];

  // fc1 + bias
  let h = matvec(w.fc1W, x, H1, IN);
  addBias(h, w.fc1b);

  // swiglu1
  h = swigluForward(w.act1Wg, w.act1Wu, w.act1Wd, h, H1, HID1);

  // fc2 + bias
  h = matvec(w.fc2W, h, H2, H1);
  addBias(h, w.fc2b);

  // swiglu2
  h = swigluForward(w.act2Wg, w.act2Wu, w.act2Wd, h, H2, HID2);

  // fc3 + bias
  const out = matvec(w.fc3W, h, OUT, H2);
  addBias(out, w.fc3b);

  return { cd: out[0], cl: out[1] };
}

// React hook

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'error';

export function useSurrogate(): {
  status: ModelStatus;
  predict: (windSpeed: number, reynolds: number, model: string) => Prediction | null;
} {
  const [status, setStatus] = useState<ModelStatus>('loading');
  const weightsRef = useRef<Weights | null>(null);
  const normRef = useRef<Normalizer | null>(null);

  useEffect(() => {
    let cancelled = false;

    Promise.all([
      fetch('/models/model.bin').then((r) => {
        if (!r.ok) throw new Error(`model.bin: ${r.status}`);
        return r.arrayBuffer();
      }),
      fetch('/models/model_norm.bin').then((r) => {
        if (!r.ok) throw new Error(`model_norm.bin: ${r.status}`);
        return r.arrayBuffer();
      }),
    ])
      .then(([wBuf, nBuf]) => {
        if (cancelled) return;
        weightsRef.current = parseWeights(wBuf);
        normRef.current = parseNorm(nBuf);
        setStatus('ready');
      })
      .catch((err) => {
        if (cancelled) return;
        console.error('Failed to load surrogate model:', err);
        setStatus('error');
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const predict = useCallback(
    (windSpeed: number, reynolds: number, model: string): Prediction | null => {
      if (!weightsRef.current || !normRef.current) return null;
      const id = MODEL_IDS[model];
      if (id === undefined) return null;
      return forward(weightsRef.current, normRef.current, windSpeed, reynolds, id);
    },
    [],
  );

  return { status, predict };
}
