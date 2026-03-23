export interface ObjValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  stats: {
    vertices: number;
    faces: number;
    boundingBox: { min: [number, number, number]; max: [number, number, number] } | null;
  };
}

const MAX_VERTICES = 500_000;
const MAX_FACES = 1_000_000;
// Bounding box warnings -- OBJ has no unit spec, but these catch obvious scale issues
const BBOX_WARN_LARGE = 100; // meters -- "did you mean to scale this?"
const BBOX_WARN_TINY = 0.001; // meters -- probably micrometers or a nearly-flat mesh

function parseVertexIndex(token: string): number {
  // OBJ face tokens: "v", "v/vt", "v/vt/vn", "v//vn"
  const slash = token.indexOf('/');
  const raw = slash === -1 ? token : token.slice(0, slash);
  return parseInt(raw, 10);
}

export function validateObj(text: string): ObjValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const vertices: [number, number, number][] = [];
  const faces: [number, number, number][] = [];

  if (!text || text.trim().length === 0) {
    return {
      valid: false,
      errors: ['File is empty'],
      warnings: [],
      stats: { vertices: 0, faces: 0, boundingBox: null },
    };
  }

  const lines = text.split('\n');
  let hasAnyContent = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || line.startsWith('#')) continue;
    hasAnyContent = true;

    if (line.startsWith('v ')) {
      const parts = line.split(/\s+/);
      if (parts.length < 4) {
        errors.push(`Line ${i + 1}: vertex needs 3 coordinates, got ${parts.length - 1}`);
        continue;
      }
      const x = parseFloat(parts[1]);
      const y = parseFloat(parts[2]);
      const z = parseFloat(parts[3]);
      if (isNaN(x) || isNaN(y) || isNaN(z)) {
        errors.push(`Line ${i + 1}: vertex has non-numeric coordinates`);
        continue;
      }
      vertices.push([x, y, z]);
    } else if (line.startsWith('f ')) {
      const parts = line.split(/\s+/).slice(1);
      if (parts.length < 3) {
        errors.push(`Line ${i + 1}: face needs at least 3 vertices, got ${parts.length}`);
        continue;
      }
      // Triangulate quads and n-gons using fan triangulation
      const indices: number[] = [];
      for (const token of parts) {
        const idx = parseVertexIndex(token);
        if (isNaN(idx)) {
          errors.push(`Line ${i + 1}: face has non-numeric vertex index`);
          break;
        }
        indices.push(idx);
      }
      if (indices.length < 3) continue;
      // Fan triangulate: (0,1,2), (0,2,3), (0,3,4), ...
      for (let j = 1; j < indices.length - 1; j++) {
        faces.push([indices[0], indices[j], indices[j + 1]]);
      }
    }
    // Silently skip vt, vn, g, o, s, mtllib, usemtl, etc.
  }

  if (!hasAnyContent) {
    errors.push('File contains no OBJ data (only comments or blank lines)');
  }

  if (vertices.length === 0) {
    errors.push('Mesh has 0 vertices');
  }
  if (faces.length === 0) {
    errors.push('Mesh has 0 faces -- the simulation needs a closed surface');
  }
  if (vertices.length > MAX_VERTICES) {
    errors.push(
      `Mesh has ${vertices.length.toLocaleString()} vertices (max ${MAX_VERTICES.toLocaleString()}). Simplify the mesh in Blender (Decimate modifier) before uploading.`,
    );
  }
  if (faces.length > MAX_FACES) {
    errors.push(
      `Mesh has ${faces.length.toLocaleString()} triangles (max ${MAX_FACES.toLocaleString()}). Simplify the mesh before uploading.`,
    );
  }

  // Stop early if fundamental problems
  if (errors.length > 0) {
    return {
      valid: false,
      errors: errors.slice(0, 5), // cap to avoid wall of text
      warnings,
      stats: { vertices: vertices.length, faces: faces.length, boundingBox: null },
    };
  }

  // Check face indices are in bounds
  let outOfBounds = 0;
  for (const [v1, v2, v3] of faces) {
    for (const idx of [v1, v2, v3]) {
      // OBJ indices are 1-based; negative indices count from the end
      const resolved = idx > 0 ? idx : vertices.length + idx + 1;
      if (resolved < 1 || resolved > vertices.length) {
        outOfBounds++;
      }
    }
  }
  if (outOfBounds > 0) {
    errors.push(
      `${outOfBounds} face indices reference non-existent vertices. The file may be corrupted or incomplete.`,
    );
  }

  // Degenerate face check (all three indices identical)
  let degenerate = 0;
  for (const [v1, v2, v3] of faces) {
    if (v1 === v2 || v2 === v3 || v1 === v3) {
      degenerate++;
    }
  }
  if (degenerate > 0) {
    const pct = ((degenerate / faces.length) * 100).toFixed(1);
    if (degenerate === faces.length) {
      errors.push('All faces are degenerate (duplicate vertex indices). This is not a valid mesh.');
    } else {
      warnings.push(
        `${degenerate} degenerate faces (${pct}% of mesh). These will be ignored by the solver.`,
      );
    }
  }

  // Bounding box
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const [x, y, z] of vertices) {
    if (x < min[0]) min[0] = x;
    if (y < min[1]) min[1] = y;
    if (z < min[2]) min[2] = z;
    if (x > max[0]) max[0] = x;
    if (y > max[1]) max[1] = y;
    if (z > max[2]) max[2] = z;
  }

  const extentX = max[0] - min[0];
  const extentY = max[1] - min[1];
  const extentZ = max[2] - min[2];
  const maxExtent = Math.max(extentX, extentY, extentZ);

  if (maxExtent > BBOX_WARN_LARGE) {
    warnings.push(
      `Model is ${maxExtent.toFixed(1)} units tall/wide. If your units are meters, that's very large -- the solver will auto-scale it, but results may be unexpected.`,
    );
  } else if (maxExtent < BBOX_WARN_TINY) {
    warnings.push(
      `Model bounding box is only ${maxExtent.toFixed(6)} units. If your units are meters, the model is sub-millimeter. The solver will auto-scale, but check your export settings.`,
    );
  }

  // Non-manifold edge check: each edge should be shared by exactly 2 faces
  const edgeCount = new Map<string, number>();
  for (const [v1, v2, v3] of faces) {
    const edges: [number, number][] = [
      [Math.min(v1, v2), Math.max(v1, v2)],
      [Math.min(v2, v3), Math.max(v2, v3)],
      [Math.min(v1, v3), Math.max(v1, v3)],
    ];
    for (const [a, b] of edges) {
      const key = `${a},${b}`;
      edgeCount.set(key, (edgeCount.get(key) ?? 0) + 1);
    }
  }
  let boundaryEdges = 0;
  let nonManifoldEdges = 0;
  for (const count of edgeCount.values()) {
    if (count === 1) boundaryEdges++;
    else if (count > 2) nonManifoldEdges++;
  }
  if (nonManifoldEdges > 0) {
    warnings.push(
      `${nonManifoldEdges} non-manifold edges detected. The mesh may produce artifacts in the simulation.`,
    );
  }
  if (boundaryEdges > 0) {
    warnings.push(
      `Mesh is not watertight (${boundaryEdges} open boundary edges). The inside/outside test may give wrong results. Consider closing holes in Blender.`,
    );
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    stats: {
      vertices: vertices.length,
      faces: faces.length,
      boundingBox: { min, max },
    },
  };
}
