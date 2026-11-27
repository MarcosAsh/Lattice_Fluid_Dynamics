// Checks POST /api/render bodies before they reach the GPU worker

export const MAX_OBJ_BASE64_LENGTH = 5 * 1024 * 1024;

const MODELS = new Set(['car', 'ahmed25', 'ahmed35', 'custom']);
const BASE64 = /^[A-Za-z0-9+/]*={0,2}$/;

export interface ModalRenderBody {
  wind_speed: number;
  viz_mode: number;
  collision_mode: number;
  duration: number;
  model: string;
  reynolds: number;
  obj_data?: string;
}

export type ParseResult =
  | { ok: true; body: ModalRenderBody }
  | { ok: false; error: string; status: number };

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function num(
  raw: unknown,
  fallback: number,
  lo: number,
  hi: number,
  integer = false,
): number | null {
  if (raw === undefined || raw === null) return fallback;
  if (typeof raw !== 'number' || !Number.isFinite(raw)) return null;
  if (integer && !Number.isInteger(raw)) return null;
  return Math.max(lo, Math.min(hi, raw));
}

export function parseRenderRequest(input: unknown): ParseResult {
  if (!isRecord(input)) {
    return {
      ok: false,
      error: 'Request body must be a JSON object',
      status: 400,
    };
  }

  const windSpeed = num(input.windSpeed, 1.0, 0, 5);
  const vizMode = num(input.vizMode, 1, 0, 7, true);
  const collisionMode = num(input.collisionMode, 2, 0, 3, true);
  const duration = num(input.duration, 5, 5, 30);
  const reynolds = num(input.reynolds, 0, 0, 100000);
  if (
    windSpeed === null ||
    vizMode === null ||
    collisionMode === null ||
    duration === null ||
    reynolds === null
  ) {
    return { ok: false, error: 'Parameters must be numbers', status: 400 };
  }

  const model = input.model ?? 'car';
  if (typeof model !== 'string' || !MODELS.has(model)) {
    return { ok: false, error: 'Unknown model', status: 400 };
  }

  const body: ModalRenderBody = {
    wind_speed: windSpeed,
    viz_mode: vizMode,
    collision_mode: collisionMode,
    duration,
    model,
    reynolds,
  };

  const objData = input.objData;
  if (model === 'custom') {
    if (typeof objData !== 'string' || objData.length === 0) {
      return {
        ok: false,
        error: 'Custom model needs an OBJ file',
        status: 400,
      };
    }
    if (objData.length > MAX_OBJ_BASE64_LENGTH) {
      return { ok: false, error: 'OBJ file too large (max 5 MB)', status: 413 };
    }
    if (!BASE64.test(objData)) {
      return { ok: false, error: 'OBJ data must be base64', status: 400 };
    }
    body.obj_data = objData;
  }

  return { ok: true, body };
}
