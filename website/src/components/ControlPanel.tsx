'use client';

import { useState, useRef } from 'react';
import { SimulationParams } from '../app/page';
import type { Prediction, ModelStatus } from '../lib/surrogate';
import { validateObj, ObjValidationResult } from '../lib/validateObj';

interface ControlPanelProps {
  params: SimulationParams;
  setParams: (params: SimulationParams) => void;
  onRender: () => void;
  disabled: boolean;
  objFile: File | null;
  onObjFileChange: (file: File | null) => void;
  mlPrediction?: Prediction | null;
  mlStatus?: ModelStatus;
}

const vizModes = [
  { value: 0, label: 'Depth', description: 'Color by distance from camera' },
  {
    value: 1,
    label: 'Velocity Magnitude',
    description: 'Blue (slow) to Red (fast)',
  },
  { value: 2, label: 'Velocity Direction', description: 'RGB = XYZ velocity' },
  { value: 3, label: 'Particle Lifetime', description: 'Age of particles' },
  { value: 4, label: 'Turbulence', description: 'Laminar vs turbulent flow' },
  { value: 5, label: 'Flow Progress', description: 'Rainbow by X position' },
  { value: 6, label: 'Vorticity', description: 'Lateral motion indicator' },
];

const collisionModes = [
  { value: 0, label: 'Off', description: 'No collision detection' },
  { value: 1, label: 'AABB', description: 'Fast bounding box' },
  { value: 2, label: 'Per-Triangle', description: 'Accurate mesh collision' },
];

const models = [
  { value: 'car', label: 'Car Model', description: 'Generic car shape' },
  {
    value: 'ahmed25',
    label: 'Ahmed Body 25deg',
    description: 'CFD benchmark, low drag',
  },
  {
    value: 'ahmed35',
    label: 'Ahmed Body 35deg',
    description: 'CFD benchmark, high drag',
  },
  {
    value: 'custom',
    label: 'Custom OBJ',
    description: 'Upload your own model',
  },
];

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  display,
  hint,
  disabled,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  display?: string;
  hint?: string;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-ctp-subtext0">{label}</span>
        <span className="text-ctp-text font-mono">{display ?? value}</span>
      </div>
      {hint && (
        <p className="text-[10px] text-ctp-overlay0 leading-tight">{hint}</p>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-ctp-mauve"
        disabled={disabled}
      />
    </div>
  );
}

const rePresets = [
  { value: 0, label: 'Auto', tag: '' },
  { value: 100, label: '100', tag: 'laminar' },
  { value: 500, label: '500', tag: 'transitional' },
  { value: 1000, label: '1k', tag: 'turbulent' },
  { value: 5000, label: '5k', tag: 'high Re' },
  { value: 10000, label: '10k', tag: 'very high' },
] as const;

function ReynoldsPresets({
  value,
  onChange,
  disabled,
}: {
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  const activeIdx = rePresets.findIndex((p) => p.value === value);
  const display = value === 0 ? 'Auto' : value.toFixed(0);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-ctp-subtext0">Reynolds Number</span>
        <span className="text-ctp-text font-mono">{display}</span>
      </div>
      <p className="text-[10px] text-ctp-overlay0 leading-tight">
        Target Re. Auto derives from wind speed.
      </p>
      <input
        type="range"
        min={0}
        max={rePresets.length - 1}
        step={1}
        value={activeIdx >= 0 ? activeIdx : 0}
        onChange={(e) => onChange(rePresets[parseInt(e.target.value)].value)}
        className="w-full accent-ctp-mauve"
        disabled={disabled}
      />
      {activeIdx >= 0 && rePresets[activeIdx].tag && (
        <div className="text-[10px] text-ctp-overlay1 -mt-0.5">
          {rePresets[activeIdx].tag}
        </div>
      )}
    </div>
  );
}

export default function ControlPanel({
  params,
  setParams,
  onRender,
  disabled,
  objFile,
  onObjFileChange,
  mlPrediction,
  mlStatus,
}: ControlPanelProps) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [objValidation, setObjValidation] = useState<ObjValidationResult | null>(null);
  const [validating, setValidating] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    setObjValidation(null);

    if (!file) {
      onObjFileChange(null);
      return;
    }

    setValidating(true);
    try {
      const text = await file.text();
      const result = validateObj(text);
      setObjValidation(result);
      onObjFileChange(result.valid ? file : null);
    } catch {
      setObjValidation({
        valid: false,
        errors: ['Could not read file'],
        warnings: [],
        stats: { vertices: 0, faces: 0, boundingBox: null },
      });
      onObjFileChange(null);
    } finally {
      setValidating(false);
    }
  };

  const isCustom = params.model === 'custom';
  const renderDisabled = disabled || (isCustom && !objFile) || validating;

  return (
    <div className="flex flex-col gap-4 p-4 border border-ctp-surface1 rounded-lg overflow-y-auto bg-ctp-mantle">
      {/* Mobile: compact header with toggle + run button */}
      <div className="flex items-center justify-between lg:hidden">
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="flex items-center gap-2 text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider"
        >
          <svg
            className={`w-3 h-3 transition-transform ${mobileOpen ? 'rotate-90' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9 5l7 7-7 7"
            />
          </svg>
          Controls
        </button>
        <button
          onClick={onRender}
          disabled={renderDisabled}
          className="bg-ctp-mauve hover:bg-ctp-lavender disabled:bg-ctp-surface1 disabled:text-ctp-overlay0 text-ctp-crust text-xs font-medium py-1.5 px-3 rounded transition-colors"
        >
          {disabled ? 'Rendering...' : 'Render'}
        </button>
      </div>

      {/* Desktop: always-visible header */}
      <h2 className="hidden lg:block text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider">
        Simulation Parameters
      </h2>

      {/* Controls body */}
      <div
        className={`flex flex-col gap-4 ${mobileOpen ? '' : 'hidden'} lg:flex`}
      >
        <div className="flex flex-col gap-1">
          <label className="text-xs text-ctp-subtext0">Model</label>
          <select
            value={params.model}
            onChange={(e) => {
              setParams({ ...params, model: e.target.value });
              if (e.target.value !== 'custom') {
                onObjFileChange(null);
                setObjValidation(null);
              }
            }}
            className="bg-ctp-surface0 text-ctp-text text-sm rounded px-2 py-1.5 border border-ctp-surface1 focus:border-ctp-mauve outline-none"
            disabled={disabled}
          >
            {models.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
          <p className="text-[10px] text-ctp-overlay0 leading-tight">
            {models.find((m) => m.value === params.model)?.description}
          </p>
        </div>

        {/* Custom OBJ upload */}
        {isCustom && (
          <div className="flex flex-col gap-1">
            <label className="text-xs text-ctp-subtext0">OBJ File</label>
            <input
              ref={fileInputRef}
              type="file"
              accept=".obj"
              onChange={handleFileChange}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled || validating}
              className="bg-ctp-surface0 hover:bg-ctp-surface1 text-ctp-text text-xs font-medium py-1.5 px-3 rounded border border-ctp-surface1 transition-colors text-left"
            >
              {validating ? 'Validating...' : objFile ? objFile.name : 'Choose file...'}
            </button>
            {objFile && objValidation && (
              <p className="text-[10px] text-ctp-overlay0">
                {(objFile.size / 1024).toFixed(0)} KB
                {' · '}
                {objValidation.stats.vertices.toLocaleString()} verts
                {' · '}
                {objValidation.stats.faces.toLocaleString()} tris
              </p>
            )}
            {!objFile && !objValidation && (
              <p className="text-[10px] text-ctp-overlay0 leading-tight">
                Upload a Wavefront .obj file (max ~5 MB)
              </p>
            )}
            {objValidation && objValidation.errors.length > 0 && (
              <div className="flex flex-col gap-0.5 mt-0.5">
                {objValidation.errors.map((err, i) => (
                  <p key={i} className="text-[10px] text-ctp-red leading-tight">
                    {err}
                  </p>
                ))}
              </div>
            )}
            {objValidation && objValidation.warnings.length > 0 && (
              <div className="flex flex-col gap-0.5 mt-0.5">
                {objValidation.warnings.map((warn, i) => (
                  <p key={i} className="text-[10px] text-ctp-yellow leading-tight">
                    {warn}
                  </p>
                ))}
              </div>
            )}
          </div>
        )}

        <Slider
          label="Wind Speed"
          value={params.windSpeed}
          min={0}
          max={5}
          step={0.1}
          onChange={(v) => setParams({ ...params, windSpeed: v })}
          display={params.windSpeed.toFixed(1)}
          hint="Freestream velocity of the simulated wind."
          disabled={disabled}
        />

        <Slider
          label="Duration"
          value={params.duration}
          min={5}
          max={30}
          step={5}
          onChange={(v) => setParams({ ...params, duration: v })}
          display={`${params.duration}s`}
          hint="Length of the rendered video clip."
          disabled={disabled}
        />

        <ReynoldsPresets
          value={params.reynolds}
          onChange={(v) => setParams({ ...params, reynolds: v })}
          disabled={disabled}
        />

        {mlStatus === 'ready' && mlPrediction && (
          <div className="border border-ctp-surface1 rounded p-3 bg-ctp-surface0/50">
            <div className="text-[10px] text-ctp-overlay0 uppercase tracking-wider mb-2">
              ML Estimate
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="text-[10px] text-ctp-overlay0">
                  C<sub>d</sub>
                </div>
                <div className="text-sm font-mono text-ctp-text">
                  {mlPrediction.cd.toFixed(4)}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-ctp-overlay0">
                  C<sub>l</sub>
                </div>
                <div className="text-sm font-mono text-ctp-text">
                  {mlPrediction.cl.toFixed(4)}
                </div>
              </div>
            </div>
            <div className="text-[9px] text-ctp-overlay0 mt-1.5">
              Surrogate prediction, updates instantly
            </div>
          </div>
        )}

        <hr className="border-ctp-surface1" />

        <div className="flex flex-col gap-1">
          <label className="text-xs text-ctp-subtext0">
            Visualization Mode
          </label>
          <select
            value={params.vizMode}
            onChange={(e) =>
              setParams({ ...params, vizMode: parseInt(e.target.value) })
            }
            className="bg-ctp-surface0 text-ctp-text text-sm rounded px-2 py-1.5 border border-ctp-surface1 focus:border-ctp-mauve outline-none"
            disabled={disabled}
          >
            {vizModes.map((mode) => (
              <option key={mode.value} value={mode.value}>
                {mode.label}
              </option>
            ))}
          </select>
          <p className="text-[10px] text-ctp-overlay0 leading-tight">
            {vizModes.find((m) => m.value === params.vizMode)?.description}
          </p>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-ctp-subtext0">Collision Mode</label>
          <select
            value={params.collisionMode}
            onChange={(e) =>
              setParams({ ...params, collisionMode: parseInt(e.target.value) })
            }
            className="bg-ctp-surface0 text-ctp-text text-sm rounded px-2 py-1.5 border border-ctp-surface1 focus:border-ctp-mauve outline-none"
            disabled={disabled}
          >
            {collisionModes.map((mode) => (
              <option key={mode.value} value={mode.value}>
                {mode.label}
              </option>
            ))}
          </select>
          <p className="text-[10px] text-ctp-overlay0 leading-tight">
            {
              collisionModes.find((m) => m.value === params.collisionMode)
                ?.description
            }
          </p>
        </div>

        <button
          onClick={onRender}
          disabled={renderDisabled}
          className="mt-2 bg-ctp-mauve hover:bg-ctp-lavender disabled:bg-ctp-surface1 disabled:text-ctp-overlay0 text-ctp-crust text-sm font-medium py-2 px-4 rounded transition-colors lg:block"
        >
          {disabled
            ? 'Rendering...'
            : isCustom && !objFile
              ? 'Select OBJ first'
              : 'Start Render'}
        </button>
      </div>
    </div>
  );
}
