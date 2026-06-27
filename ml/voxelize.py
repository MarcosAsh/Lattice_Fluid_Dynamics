"""
Voxelize OBJ meshes into fixed-size binary occupancy grids for ML input.

A numpy port of simulation/src/voxelize.c. It rasterizes any mesh into a
deterministic, resolution-configurable occupancy grid in the solver's
frame -- centered, longest axis rotated to X (the flow direction), scaled
into the [-1, 1]^3 cube -- and reads/writes the same VOXB binary format.
On the same triangle input the two voxelizers agree bit-for-bit, so a
training grid built here is the grid the C inference path sees, and the
two CLIs (this and voxelize_obj) are interchangeable.

One caveat for raw quad meshes: this loader fan-triangulates each polygon,
while the minimal C OBJ loader keeps only a face's first triangle. Run the
preprocessing pipeline (ml/preprocess.py, issue #137) to triangulate up
front and the two paths converge on identical geometry.

Modes:
    solid    parity-fill X-ray casting: a cell is solid if its center is
             inside the mesh (correct only for watertight meshes)
    surface  mark every cell a triangle's bounding box touches
    auto     solid, falling back to surface when the solid fill comes out
             degenerate (non-watertight / open meshes); the default

Usage:
    from ml.voxelize import voxelize_obj, write_voxbin, read_voxbin

    grid = voxelize_obj("body.obj", resolution=32)   # (R, R, R) uint8
    write_voxbin(grid, "body.voxbin")

    python ml/voxelize.py body.obj --resolution 32 --output body.voxbin
    python ml/voxelize.py --selftest
"""

import argparse
import math
import struct
import sys

import numpy as np

VOXB_MAGIC = b"VOXB"
VOXB_VERSION = 1

DEFAULT_RESOLUTION = 32
DEFAULT_PADDING = 0.05  # longest axis spans 2 - 2*padding inside [-1, 1]^3
EPS = 1e-8

MODE_SOLID = "solid"
MODE_SURFACE = "surface"
MODE_AUTO = "auto"
MODES = (MODE_SOLID, MODE_SURFACE, MODE_AUTO)


# Mesh loading


def _read_text(source):
    """Return OBJ text from a path, raw bytes, or inline OBJ text."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source).decode("utf-8", "replace")
    if isinstance(source, str) and "\n" in source and "v " in source:
        return source
    with open(source) as f:
        return f.read()


def load_obj_triangles(source):
    """
    Parse an OBJ into a vertex array and a triangle index array.

    Accepts a path, OBJ bytes, or inline OBJ text. Face lines may use
    v, v/vt, v/vt/vn, or v//vn vertex references; negative (relative)
    indices are resolved. Polygons with more than three vertices are
    fan-triangulated, matching how the solver consumes only triangles.

    Returns:
        (vertices, triangles): float64 (N, 3) and int32 (M, 3), 0-based.
    """
    verts = []
    tris = []
    for line in _read_text(source).splitlines():
        if line.startswith("v "):
            parts = line.split()
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            tokens = line.split()[1:]
            idx = []
            for tok in tokens:
                raw = int(tok.split("/")[0])
                idx.append(raw - 1 if raw > 0 else len(verts) + raw)
            for i in range(1, len(idx) - 1):
                tris.append((idx[0], idx[i], idx[i + 1]))
    vertices = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(tris, dtype=np.int32).reshape(-1, 3)
    return vertices, triangles


def canonicalize(vertices, padding=DEFAULT_PADDING, align_longest_x=True):
    """
    Center the mesh, rotate its longest axis to X, scale into [-1, 1]^3.

    Mirrors copy_and_normalize_vertices in simulation/src/voxelize.c, so X
    is the flow axis and the body fills the same cube the solver voxelizes.

    Returns:
        A new float64 (N, 3) array, or None if the mesh has zero extent.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.shape[0] == 0:
        return None

    lo = v.min(axis=0)
    hi = v.max(axis=0)
    v = v - 0.5 * (lo + hi)
    size = hi - lo

    if align_longest_x:
        sx, sy, sz = size
        # Single swap, matching the solver: Y longest -> swap X/Y,
        # else Z longest -> swap X/Z.
        if sy > sx and sy >= sz:
            v = v[:, [1, 0, 2]]
            size = size[[1, 0, 2]]
        elif sz > sx and sz > sy:
            v = v[:, [2, 1, 0]]
            size = size[[2, 1, 0]]

    max_dim = float(size.max())
    if max_dim < EPS:
        return None
    v = v * ((2.0 - 2.0 * padding) / max_dim)
    return v


# Voxelization


def _voxelize_surface(verts, tris, R, grid):
    """Mark every cell whose extent overlaps a triangle's bounding box."""
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    lo = np.minimum(np.minimum(v0, v1), v2)
    hi = np.maximum(np.maximum(v0, v1), v2)

    g0 = np.floor((lo + 1.0) * 0.5 * R).astype(np.int64)
    g1 = np.ceil((hi + 1.0) * 0.5 * R).astype(np.int64)
    np.clip(g0, 0, R, out=g0)
    np.clip(g1, 0, R, out=g1)

    for i in range(tris.shape[0]):
        x0, y0, z0 = g0[i]
        x1, y1, z1 = g1[i]
        if x0 < x1 and y0 < y1 and z0 < z1:
            grid[x0:x1, y0:y1, z0:z1] = 1


def _voxelize_solid(verts, tris, R, grid):
    """
    Parity-fill voxelization via +X ray casting at cell centers.

    For each triangle, intersect the rays through the (y, z) cell-center
    columns it spans (Moller-Trumbore with direction (1, 0, 0)) and record
    the intersection x. Each column is then sorted and filled between
    consecutive entry/exit pairs -- exactly simulation/src/voxelize.c.
    """
    hits = [[] for _ in range(R * R)]  # per column, index gy * R + gz

    v0 = verts[tris[:, 0]]
    e1 = verts[tris[:, 1]] - v0
    e2 = verts[tris[:, 2]] - v0

    for f in range(tris.shape[0]):
        v0x, v0y, v0z = v0[f]
        e1x, e1y, e1z = e1[f]
        e2x, e2y, e2z = e2[f]

        # a = e1 . (dir x e2), dir = (1, 0, 0); skip rays parallel to face.
        a = -e1y * e2z + e1z * e2y
        if abs(a) < EPS:
            continue
        inv_a = 1.0 / a

        y_min = min(v0y, v0y + e1y, v0y + e2y)
        y_max = max(v0y, v0y + e1y, v0y + e2y)
        z_min = min(v0z, v0z + e1z, v0z + e2z)
        z_max = max(v0z, v0z + e1z, v0z + e2z)
        if y_max < -1.0 or y_min > 1.0 or z_max < -1.0 or z_min > 1.0:
            continue

        gy_lo = max(int(math.floor((y_min + 1.0) * 0.5 * R)), 0)
        gy_hi = min(int(math.ceil((y_max + 1.0) * 0.5 * R)) + 1, R)
        gz_lo = max(int(math.floor((z_min + 1.0) * 0.5 * R)), 0)
        gz_hi = min(int(math.ceil((z_max + 1.0) * 0.5 * R)) + 1, R)
        if gy_lo >= gy_hi or gz_lo >= gz_hi:
            continue

        gy = np.arange(gy_lo, gy_hi)
        gz = np.arange(gz_lo, gz_hi)
        wy = (gy + 0.5) / R * 2.0 - 1.0
        wz = (gz + 0.5) / R * 2.0 - 1.0
        sy = (wy - v0y)[:, None]  # (ny, 1)
        sz = (wz - v0z)[None, :]  # (1, nz)
        sx = -v0x  # ray origin x = 0

        u = inv_a * (sy * -e2z + sz * e2y)
        qx = sy * e1z - sz * e1y
        qy = sz * e1x - sx * e1z
        qz = sx * e1y - sy * e1x
        v_coord = inv_a * qx
        t = inv_a * (e2x * qx + e2y * qy + e2z * qz)

        ok = (u >= 0.0) & (u <= 1.0) & (v_coord >= 0.0) & (u + v_coord <= 1.0)
        ay, az = np.nonzero(ok)
        if ay.size == 0:
            continue
        cols = (gy_lo + ay) * R + (gz_lo + az)
        tvals = t[ay, az]
        for col, tv in zip(cols.tolist(), tvals.tolist()):
            hits[col].append(tv)

    for gy in range(R):
        for gz in range(R):
            col = hits[gy * R + gz]
            if len(col) < 2:
                continue
            col.sort()
            # Drop near-duplicate intersections (shared edge / vertex hits).
            dedup = [col[0]]
            for x in col[1:]:
                if x - dedup[-1] > 1e-6:
                    dedup.append(x)
            for i in range(0, len(dedup) - 1, 2):
                x0, x1 = dedup[i], dedup[i + 1]
                gx0 = max(int(math.ceil((x0 + 1.0) * 0.5 * R - 0.5)), 0)
                gx1 = min(int(math.floor((x1 + 1.0) * 0.5 * R - 0.5)) + 1, R)
                if gx0 < gx1:
                    grid[gx0:gx1, gy, gz] = 1


def voxelize_mesh(vertices, triangles, resolution=DEFAULT_RESOLUTION,
                  mode=MODE_AUTO, padding=DEFAULT_PADDING,
                  align_longest_x=True):
    """
    Voxelize an in-memory triangle mesh into an occupancy grid.

    Args:
        vertices: (N, 3) array of vertex positions.
        triangles: (M, 3) array of 0-based vertex indices.
        resolution: grid edge length R; the grid is (R, R, R).
        mode: one of "solid", "surface", "auto".
        padding: border left around the body inside the [-1, 1]^3 cube.
        align_longest_x: rotate the longest axis onto X (the flow axis).

    Returns:
        (R, R, R) uint8 occupancy grid (1 = solid, 0 = fluid).
    """
    R = int(resolution)
    if R <= 0:
        raise ValueError("resolution must be positive")
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}")
    padding = min(max(float(padding), 0.0), 0.499)

    tris = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if tris.shape[0] == 0:
        raise ValueError("mesh has no triangles")

    verts = canonicalize(vertices, padding, align_longest_x)
    if verts is None:
        raise ValueError("degenerate mesh: zero extent")

    grid = np.zeros((R, R, R), dtype=np.uint8)
    if mode == MODE_SURFACE:
        _voxelize_surface(verts, tris, R, grid)
    else:
        _voxelize_solid(verts, tris, R, grid)
        if mode == MODE_AUTO:
            frac = float(grid.mean())
            if frac < 0.001 or frac > 0.99:
                grid.fill(0)
                _voxelize_surface(verts, tris, R, grid)
    return grid


def voxelize_obj(source, resolution=DEFAULT_RESOLUTION, mode=MODE_AUTO,
                 padding=DEFAULT_PADDING, align_longest_x=True):
    """Load an OBJ (path, bytes, or text) and voxelize it. See voxelize_mesh."""
    vertices, triangles = load_obj_triangles(source)
    return voxelize_mesh(vertices, triangles, resolution, mode, padding,
                         align_longest_x)


def solid_fraction(grid):
    """Fraction of cells marked solid, in [0, 1]."""
    grid = np.asarray(grid)
    return float(grid.mean()) if grid.size else 0.0


# Binary I/O (VOXB, shared with simulation/src/voxelize.c)


def write_voxbin(grid, path):
    """
    Write a grid to the VOXB binary format read by the C inference path.

    Layout: magic "VOXB", u32 version, u32 resolution, then R*R*R bytes
    indexed x + R*(y + R*z) -- the column-major order the solver uses.
    """
    grid = np.ascontiguousarray(grid, dtype=np.uint8)
    R = grid.shape[0]
    if grid.shape != (R, R, R):
        raise ValueError("grid must be a cubic (R, R, R) array")
    with open(path, "wb") as f:
        f.write(VOXB_MAGIC)
        f.write(struct.pack("<I", VOXB_VERSION))
        f.write(struct.pack("<I", R))
        f.write(grid.tobytes(order="F"))


def read_voxbin(path):
    """Read a VOXB file written by this module or the C voxelizer."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != VOXB_MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        (version,) = struct.unpack("<I", f.read(4))
        if version != VOXB_VERSION:
            raise ValueError(f"unsupported VOXB version: {version}")
        (R,) = struct.unpack("<I", f.read(4))
        payload = f.read(R * R * R)
    if len(payload) != R * R * R:
        raise ValueError("truncated VOXB payload")
    grid = np.frombuffer(payload, dtype=np.uint8)
    return grid.reshape((R, R, R), order="F").copy()


# Self-tests (run in CI; mirror simulation/test/test_voxelize.c)


def _cube(h):
    """Watertight axis-aligned cube of half-extent h, centered at origin."""
    verts = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ], dtype=np.float64)
    # 1-based winding from test_voxelize.c, shifted to 0-based.
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ], dtype=np.int32)
    return verts, faces


def _selftest():
    failures = 0
    total = 0

    def check(cond, msg):
        nonlocal failures, total
        total += 1
        if not cond:
            failures += 1
            print(f"  FAIL: {msg}")

    verts, faces = _cube(1.0)

    g = voxelize_mesh(verts, faces, 16, MODE_SOLID)
    frac = solid_fraction(g)
    print(f"cube solid fraction: {frac:.3f}")
    check(frac > 0.80, "cube should be mostly solid")
    check(frac <= 1.0, "solid fraction <= 1")

    g = voxelize_mesh(verts, faces, 16, MODE_SURFACE)
    frac = solid_fraction(g)
    print(f"cube surface fraction: {frac:.3f}")
    check(0.20 < frac < 0.95, "surface fraction in plausible range")
    check(g[8, 8, 8] == 0, "surface mode leaves the interior empty")

    # Z-elongated slab must be realigned so it spans more cells in X.
    slab_v = np.array([
        [-0.1, -0.2, -0.8], [0.1, -0.2, -0.8], [0.1, 0.2, -0.8],
        [-0.1, 0.2, -0.8], [-0.1, -0.2, 0.8], [0.1, -0.2, 0.8],
        [0.1, 0.2, 0.8], [-0.1, 0.2, 0.8],
    ], dtype=np.float64)
    g = voxelize_mesh(slab_v, faces, 32, MODE_SOLID)
    xs = np.nonzero(g.any(axis=(1, 2)))[0]
    zs = np.nonzero(g.any(axis=(0, 1)))[0]
    x_span = xs.max() - xs.min()
    z_span = zs.max() - zs.min()
    print(f"slab x_span={x_span} z_span={z_span}")
    check(x_span > z_span, "longest axis realigned to X")

    g1 = voxelize_mesh(verts, faces, 16, MODE_SOLID)
    g2 = voxelize_mesh(verts, faces, 16, MODE_SOLID)
    check(np.array_equal(g1, g2), "voxelization is deterministic")

    # Quad faces must be fan-triangulated into a solid quad sheet.
    quad = "v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nf 1 2 3 4\n"
    qv, qt = load_obj_triangles(quad)
    check(qt.shape[0] == 2, "quad triangulated into two triangles")

    # AUTO must fall back to surface for an open (non-watertight) sheet.
    # A 3D-tilted quad has no enclosed volume, so solid fill comes out
    # empty and only the surface pass marks any cells.
    tilted = "v -1 -1 -1\nv 1 -1 -1\nv 1 1 1\nv -1 1 1\nf 1 2 3 4\n"
    tv, tt = load_obj_triangles(tilted)
    check(solid_fraction(voxelize_mesh(tv, tt, 16, MODE_SOLID)) < 0.001,
          "open sheet has no solid fill")
    check(solid_fraction(voxelize_mesh(tv, tt, 16, MODE_AUTO)) > 0.0,
          "auto falls back to surface for an open mesh")

    # VOXB round-trip, including the x + R*(y + R*z) byte order.
    import tempfile
    from pathlib import Path

    probe = np.zeros((8, 8, 8), dtype=np.uint8)
    probe[1, 2, 3] = 1
    with tempfile.NamedTemporaryFile(suffix=".voxbin", delete=False) as f:
        path = Path(f.name)
    write_voxbin(probe, path)
    back = read_voxbin(path)
    path.unlink()
    check(back.shape == (8, 8, 8), "round-trip resolution preserved")
    check(np.array_equal(back, probe), "round-trip preserves byte order")

    print(f"{total - failures}/{total} checks passed")
    return failures


def _build_parser():
    ap = argparse.ArgumentParser(
        description="Voxelize an OBJ mesh into a VOXB occupancy grid"
    )
    ap.add_argument("input", nargs="?", help="input OBJ path")
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    ap.add_argument("--padding", type=float, default=DEFAULT_PADDING)
    ap.add_argument("--mode", choices=MODES, default=MODE_AUTO)
    ap.add_argument("--no-align", dest="align", action="store_false",
                    help="do not rotate the longest axis onto X")
    ap.add_argument("--output", "-o", help="write the grid as a .voxbin file")
    ap.add_argument("--selftest", action="store_true",
                    help="run built-in tests and exit")
    return ap


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.selftest:
        return 1 if _selftest() else 0
    if not args.input:
        _build_parser().error("input OBJ required (or pass --selftest)")

    grid = voxelize_obj(args.input, args.resolution, args.mode, args.padding,
                        args.align)
    frac = solid_fraction(grid)
    total = grid.size
    solid = int(grid.sum())
    R = grid.shape[0]
    print(f"voxelized {args.input}: {R}x{R}x{R}, "
          f"solid={solid}/{total} ({100.0 * frac:.2f}%)")
    if args.output:
        write_voxbin(grid, args.output)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
