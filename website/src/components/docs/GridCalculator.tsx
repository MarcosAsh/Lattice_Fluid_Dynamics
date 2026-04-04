'use client';

import { useState, useMemo } from 'react';

export function GridCalculator() {
  const [gridX, setGridX] = useState(256);
  const [gridY, setGridY] = useState(128);
  const [gridZ, setGridZ] = useState(128);

  const results = useMemo(() => {
    const scaleX = gridX / 8;
    const bodyWorldLen = 2.4; // auto-scaled model
    const charLength = bodyWorldLen * scaleX;

    const U = 0.05;
    const tauMin = 0.52;
    const nuMin = (tauMin - 0.5) / 3;
    const reMax = (U * charLength) / nuMin;

    // Stable tau = 0.65 for reliable single-sample Cd
    const nuStable = (0.65 - 0.5) / 3;
    const reStable = (U * charLength) / nuStable;

    // Memory: 19 floats * 2 (f + f_new) + 4 floats (vel) + 1 int (solid) + 19 floats (q)
    const cells = gridX * gridY * gridZ;
    const memBytes = cells * ((19 * 2 + 4 + 19) * 4 + 4);
    const memMB = memBytes / (1024 * 1024);

    // SSBO per buffer (largest is f: 19 * cells * 4 bytes)
    const ssboMB = (19 * cells * 4) / (1024 * 1024);

    return {
      charLength: charLength.toFixed(1),
      reMax: Math.round(reMax),
      reStable: Math.round(reStable),
      cells: (cells / 1e6).toFixed(1),
      memMB: Math.round(memMB),
      ssboMB: Math.round(ssboMB),
      bodyWidth: (bodyWorldLen * gridY / 4).toFixed(0),
      bodyHeight: ((bodyWorldLen * 0.66 / 2.4) * gridZ / 4).toFixed(0),
    };
  }, [gridX, gridY, gridZ]);

  const presets = [
    { label: '128x64x64', x: 128, y: 64, z: 64 },
    { label: '128x96x96', x: 128, y: 96, z: 96 },
    { label: '256x128x128', x: 256, y: 128, z: 128 },
    { label: '512x256x256', x: 512, y: 256, z: 256 },
  ];

  return (
    <div className="rounded-lg border border-ctp-surface1 bg-ctp-mantle p-4 my-4">
      <h4 className="text-sm font-semibold mb-3 text-ctp-text">
        Grid Resolution Calculator
      </h4>

      <div className="flex flex-wrap gap-2 mb-4">
        {presets.map((p) => (
          <button
            key={p.label}
            onClick={() => { setGridX(p.x); setGridY(p.y); setGridZ(p.z); }}
            className={`px-3 py-1 text-xs rounded-md border transition-colors ${
              gridX === p.x && gridY === p.y && gridZ === p.z
                ? 'bg-ctp-blue text-ctp-base border-ctp-blue'
                : 'border-ctp-surface2 text-ctp-subtext0 hover:border-ctp-blue'
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        {[
          { label: 'X (streamwise)', value: gridX, set: setGridX },
          { label: 'Y (lateral)', value: gridY, set: setGridY },
          { label: 'Z (vertical)', value: gridZ, set: setGridZ },
        ].map(({ label, value, set }) => (
          <div key={label}>
            <label className="text-xs text-ctp-overlay1 block mb-1">{label}</label>
            <input
              type="number"
              min={16}
              max={1024}
              step={16}
              value={value}
              onChange={(e) => set(Number(e.target.value) || 16)}
              className="w-full px-2 py-1 text-sm bg-ctp-base border border-ctp-surface2 rounded text-ctp-text"
            />
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
        <div className="text-ctp-overlay1">Total cells</div>
        <div className="text-ctp-text font-mono">{results.cells}M</div>

        <div className="text-ctp-overlay1">Body length (lattice)</div>
        <div className="text-ctp-text font-mono">{results.charLength} cells</div>

        <div className="text-ctp-overlay1">Re (max, &tau;=0.52)</div>
        <div className="text-ctp-text font-mono">{results.reMax}</div>

        <div className="text-ctp-overlay1">Re (stable, &tau;=0.65)</div>
        <div className="text-ctp-text font-mono">{results.reStable}</div>

        <div className="text-ctp-overlay1">GPU memory</div>
        <div className="text-ctp-text font-mono">{results.memMB} MB</div>

        <div className="text-ctp-overlay1">Largest SSBO buffer</div>
        <div className={`font-mono ${
          results.ssboMB > 128 ? 'text-ctp-red' : 'text-ctp-green'
        }`}>
          {results.ssboMB} MB
          {results.ssboMB > 128 && ' (needs EGL/native GPU)'}
        </div>
      </div>
    </div>
  );
}
