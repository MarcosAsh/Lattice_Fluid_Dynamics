// Pure TypeScript inference for the shape-optimization Cd/Cl surrogate.
//
// Architecture (same layer types as the base surrogate, wider input):
//   Linear(110,256) -> SwiGLU(256,512) -> Linear(256,128)
//   -> SwiGLU(128,256) -> Linear(128,2)
//
// Input: 108 FFD displacements + wind_speed + reynolds = 110
// Output: [cd, cl]
//
// Loads shapeopt_model.bin + shapeopt_norm.bin from /models/ and runs
// the forward pass entirely in the browser.

'use client';

// Binary format constants (must match ml/framework/src/weights_io.cpp)

const MAGIC = 0x4c545753;
const VERSION = 1;
const NUM_PARAMS = 12;

// Model dimensions
const IN = 110; // 108 FFD displacements + wind_speed + reynolds
const H1 = 256;
const HID1 = 512;
const H2 = 128;
const HID2 = 256;
const OUT = 2;

const NUM_FFD = 108;

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
  // Layout: 110 floats for mean, then 110 floats for stddev
  return {
    mean: f.slice(0, IN),
    stdDev: f.slice(IN, IN * 2),
  };
}

// Forward pass

export interface ShapeOptModel {
  weights: Weights;
  norm: Normalizer;
}

function buildInput(
  norm: Normalizer,
  ffdDisplacements: number[],
  windSpeed: number,
  reynolds: number,
): Float32Array {
  const x = new Float32Array(IN);
  for (let i = 0; i < NUM_FFD; i++) {
    x[i] = (ffdDisplacements[i] - norm.mean[i]) / norm.stdDev[i];
  }
  x[NUM_FFD] = (windSpeed - norm.mean[NUM_FFD]) / norm.stdDev[NUM_FFD];
  x[NUM_FFD + 1] = (reynolds - norm.mean[NUM_FFD + 1]) / norm.stdDev[NUM_FFD + 1];
  return x;
}

function forward(w: Weights, x: Float32Array): { cd: number; cl: number } {
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

// Public API

export async function loadShapeOptSurrogate(): Promise<ShapeOptModel> {
  const [wBuf, nBuf] = await Promise.all([
    fetch('/models/shapeopt_model.bin').then((r) => {
      if (!r.ok) throw new Error(`shapeopt_model.bin: ${r.status}`);
      return r.arrayBuffer();
    }),
    fetch('/models/shapeopt_norm.bin').then((r) => {
      if (!r.ok) throw new Error(`shapeopt_norm.bin: ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);

  return {
    weights: parseWeights(wBuf),
    norm: parseNorm(nBuf),
  };
}

export function predictCd(
  model: ShapeOptModel,
  ffdDisplacements: number[],
  windSpeed: number,
  reynolds: number,
): { cd: number; cl: number } {
  if (ffdDisplacements.length !== NUM_FFD) {
    throw new Error(`Expected ${NUM_FFD} FFD displacements, got ${ffdDisplacements.length}`);
  }
  const x = buildInput(model.norm, ffdDisplacements, windSpeed, reynolds);
  return forward(model.weights, x);
}
