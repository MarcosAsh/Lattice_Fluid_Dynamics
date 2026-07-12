// Pure TypeScript inference for the any-mesh Cd/Cl surrogate.
//
// Architecture (matches ml/anyobj/train.py):
//   Linear(20, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 2)
//
// Inputs are 19 geometry descriptors plus log10 Reynolds, z-score
// normalized. Outputs are denormalized back to raw Cd and Cl. Descriptors
// for the bundled geometries are precomputed by ml/anyobj/export_web.py
// and shipped inside /models/anyobj.json, so the browser does no mesh
// processing. Wind speed is accepted for interface compatibility but not
// used, Cd at a fixed Reynolds number is independent of it.

'use client';

import { useState, useEffect, useRef, useCallback } from 'react';

interface Layer {
  in: number;
  out: number;
  w: number[];
  b: number[];
}

interface AnyobjModel {
  layers: Layer[];
  xMean: number[];
  xStd: number[];
  yMean: number[];
  yStd: number[];
  featureNames: string[];
  targetNames: string[];
  descriptors: Record<string, number[]>;
}

export interface Prediction {
  cd: number;
  cl: number;
}

function forward(model: AnyobjModel, geometry: string, reynolds: number): Prediction | null {
  const desc = model.descriptors[geometry];
  if (!desc) return null;

  let x = new Float64Array(desc.length + 1);
  x.set(desc);
  x[desc.length] = Math.log10(reynolds);
  for (let i = 0; i < x.length; i++) {
    x[i] = (x[i] - model.xMean[i]) / model.xStd[i];
  }

  model.layers.forEach((layer, li) => {
    const y = new Float64Array(layer.out);
    for (let j = 0; j < layer.out; j++) {
      let sum = layer.b[j];
      for (let i = 0; i < layer.in; i++) {
        sum += x[i] * layer.w[i * layer.out + j];
      }
      y[j] = li < model.layers.length - 1 ? Math.max(0, sum) : sum;
    }
    x = y;
  });

  return {
    cd: x[0] * model.yStd[0] + model.yMean[0],
    cl: x[1] * model.yStd[1] + model.yMean[1],
  };
}

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'error';

export function useSurrogate(): {
  status: ModelStatus;
  predict: (windSpeed: number, reynolds: number, model: string) => Prediction | null;
} {
  const [status, setStatus] = useState<ModelStatus>('loading');
  const modelRef = useRef<AnyobjModel | null>(null);

  useEffect(() => {
    let cancelled = false;

    fetch('/models/anyobj.json')
      .then((r) => {
        if (!r.ok) throw new Error(`anyobj.json: ${r.status}`);
        return r.json();
      })
      .then((m: AnyobjModel) => {
        if (cancelled) return;
        modelRef.current = m;
        setStatus('ready');
      })
      .catch((err) => {
        console.error('Failed to load anyobj surrogate:', err);
        if (!cancelled) setStatus('error');
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const predict = useCallback(
    (_windSpeed: number, reynolds: number, model: string): Prediction | null => {
      if (!modelRef.current) return null;
      return forward(modelRef.current, model, reynolds);
    },
    [],
  );

  return { status, predict };
}
