"""
OBJ model preprocessing pipeline.

The solver's OBJ loader is minimal -- it keeps only each face's first
triangle and does no validation -- so malformed or poorly scaled meshes
fail silently or give bad simulation results. This module cleans a mesh
before it reaches the solver:

    1. Triangulate   quads split along the shorter diagonal, n-gons fanned
    2. Center+scale  bounding box centered at the origin, longest axis
                     scaled to the simulation domain (matches the solver)
    3. Validate      degenerate triangles, boundary (open) edges,
                     non-manifold edges, inconsistent winding, and
                     disconnected components
    4. Normals       consistent, outward-facing per-face and per-vertex
    5. Stats         vertex/face counts, bounding box, surface area,
                     volume, watertightness

Run it server-side before a render, or as a check on an uploaded mesh.

Usage:
    from ml.preprocess import preprocess

    result = preprocess("body.obj")
    result["ok"]            # False if the mesh has no usable geometry
    result["validation"]    # issue counts
    result["stats"]         # mesh statistics

    python ml/preprocess.py body.obj --out clean.obj
    python ml/preprocess.py body.obj --json
    python ml/preprocess.py            # report on the bundled OBJs
    python ml/preprocess.py --selftest
"""

import argparse
import json
import sys

import numpy as np

DEFAULT_DOMAIN = 2.0  # longest axis scaled to this, matching [-1, 1]^3
EPS = 1e-12


# Mesh loading


def _read_text(source):
    """Return OBJ text from a path, raw bytes, or inline OBJ text."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source).decode("utf-8", "replace")
    if isinstance(source, str) and "\n" in source and "v " in source:
        return source
    with open(source) as f:
        return f.read()


def load_obj(source):
    """
    Parse an OBJ into vertices and raw (un-triangulated) faces.

    Face lines may use v, v/vt, v/vt/vn, or v//vn references; negative
    (relative) indices are resolved. Polygons are kept intact so the
    triangulation step can report what it split.

    Returns:
        (vertices, faces): float64 (N, 3) array and a list of 0-based
        index lists, one per face.
    """
    verts = []
    faces = []
    for line in _read_text(source).splitlines():
        if line.startswith("v "):
            p = line.split()
            verts.append((float(p[1]), float(p[2]), float(p[3])))
        elif line.startswith("f "):
            idx = []
            for tok in line.split()[1:]:
                raw = int(tok.split("/")[0])
                idx.append(raw - 1 if raw > 0 else len(verts) + raw)
            if len(idx) >= 3:
                faces.append(idx)
    vertices = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    return vertices, faces


# Triangulation


def triangulate(vertices, faces):
    """
    Triangulate faces: quads along the shorter diagonal, n-gons by a fan.

    Returns:
        (triangles, n_quads, n_ngons): int64 (M, 3) array and the counts
        of quad / >4-gon faces that were split.
    """
    tris = []
    n_quads = 0
    n_ngons = 0
    for face in faces:
        k = len(face)
        if k == 3:
            tris.append(tuple(face))
        elif k == 4:
            n_quads += 1
            a, b, c, d = face
            ac = vertices[a] - vertices[c]
            bd = vertices[b] - vertices[d]
            if ac.dot(ac) <= bd.dot(bd):
                tris.append((a, b, c))
                tris.append((a, c, d))
            else:
                tris.append((b, c, d))
                tris.append((b, d, a))
        elif k > 4:
            n_ngons += 1
            for i in range(1, k - 1):
                tris.append((face[0], face[i], face[i + 1]))
    triangles = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    return triangles, n_quads, n_ngons


# Center and scale


def center_and_scale(vertices, domain=DEFAULT_DOMAIN):
    """
    Center the bounding box at the origin and scale the longest axis.

    Centering uses the bounding-box center (not the vertex mean) so the
    result matches the solver's voxelization frame. The longest extent is
    scaled to `domain`; a zero-extent axis leaves the scale at 1.

    Returns:
        (vertices, transform): new (N, 3) array and a dict with the
        applied "translation" and "scale".
    """
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center = 0.5 * (lo + hi)
    max_dim = float((hi - lo).max())
    scale = domain / max_dim if max_dim > EPS else 1.0
    out = (vertices - center) * scale
    return out, {"translation": (-center).tolist(), "scale": scale}


# Geometry helpers


def face_areas(vertices, triangles):
    """Per-triangle area (float64, shape (M,))."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def signed_volume6(vertices, triangles):
    """Six times the signed volume (sign encodes winding orientation)."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum())


def _edge_counts(triangles):
    """Map each undirected edge to (count_ab, count_ba) directed totals."""
    directed = {}
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            directed[(u, v)] = directed.get((u, v), 0) + 1
    edges = {}
    for (u, v), n in directed.items():
        key = (u, v) if u < v else (v, u)
        fwd = n if u < v else 0
        bwd = 0 if u < v else n
        if key in edges:
            edges[key][0] += fwd
            edges[key][1] += bwd
        else:
            edges[key] = [fwd, bwd]
    return edges


def _count_components(triangles, n_vertices):
    """Number of connected components over vertices sharing a triangle."""
    parent = list(range(n_vertices))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b, c in triangles:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[rc] = ra

    used = np.unique(triangles)
    return len({find(int(v)) for v in used})


# Validation


def validate(vertices, triangles):
    """
    Check a triangle mesh for the defects that break the solver.

    Returns:
        dict with degenerate-triangle / boundary-edge / non-manifold-edge /
        flipped-edge counts, component count, and a `watertight` flag.
    """
    areas = face_areas(vertices, triangles)
    repeated = (
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    )
    area_eps = EPS + 1e-9 * float(areas.max(initial=0.0))
    degenerate = int((repeated | (areas < area_eps)).sum())

    edges = _edge_counts(triangles)
    boundary = 0
    non_manifold = 0
    flipped = 0
    for fwd, bwd in edges.values():
        total = fwd + bwd
        if total == 1:
            boundary += 1
        elif total > 2:
            non_manifold += 1
        elif total == 2 and not (fwd == 1 and bwd == 1):
            # Both incident triangles traverse the edge the same way:
            # one of them is wound backwards.
            flipped += 1

    components = _count_components(triangles, vertices.shape[0])
    watertight = boundary == 0 and non_manifold == 0 and flipped == 0

    return {
        "degenerate_triangles": degenerate,
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
        "flipped_edges": flipped,
        "components": components,
        "watertight": watertight,
    }


# Normals


def compute_normals(vertices, triangles):
    """
    Per-face and area-weighted per-vertex normals, oriented outward.

    For a watertight mesh wound inward (negative signed volume) every
    triangle is reversed so the normals point outward. Returns possibly
    reordered triangles alongside the normals.

    Returns:
        (triangles, face_normals, vertex_normals).
    """
    tris = triangles
    if signed_volume6(vertices, tris) < 0.0:
        tris = tris[:, ::-1].copy()

    v0 = vertices[tris[:, 0]]
    v1 = vertices[tris[:, 1]]
    v2 = vertices[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    face_normals = fn / np.maximum(norms, EPS)

    vn = np.zeros_like(vertices)
    # Accumulate area-weighted (un-normalized) face normals per vertex.
    for i in range(3):
        np.add.at(vn, tris[:, i], fn)
    vnorm = np.linalg.norm(vn, axis=1, keepdims=True)
    vertex_normals = vn / np.maximum(vnorm, EPS)
    return tris, face_normals, vertex_normals


# Statistics


def mesh_stats(vertices, triangles, n_quads=0, n_ngons=0):
    """Summary statistics for a triangulated mesh."""
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    areas = face_areas(vertices, triangles)
    return {
        "vertices": int(vertices.shape[0]),
        "triangles": int(triangles.shape[0]),
        "quads_split": int(n_quads),
        "ngons_split": int(n_ngons),
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "bbox_size": (hi - lo).tolist(),
        "surface_area": float(areas.sum()),
        "volume": abs(signed_volume6(vertices, triangles)) / 6.0,
        "centroid": (0.5 * (lo + hi)).tolist(),
    }


# Pipeline


def preprocess(source, domain=DEFAULT_DOMAIN, recenter=True):
    """
    Run the full preprocessing pipeline on an OBJ source.

    Args:
        source: OBJ path, bytes, or inline text.
        domain: longest-axis size after scaling.
        recenter: center and scale the mesh (off to keep input coordinates).

    Returns:
        dict with keys: ok, vertices, triangles, face_normals,
        vertex_normals, transform, validation, stats, warnings. `ok` is
        False when the mesh has no usable (non-degenerate) triangle.
    """
    vertices, faces = load_obj(source)
    triangles, n_quads, n_ngons = triangulate(vertices, faces)

    warnings = []
    if vertices.shape[0] == 0 or triangles.shape[0] == 0:
        return {
            "ok": False,
            "vertices": vertices,
            "triangles": triangles,
            "face_normals": np.zeros((0, 3)),
            "vertex_normals": np.zeros_like(vertices),
            "transform": {"translation": [0.0, 0.0, 0.0], "scale": 1.0},
            "validation": {},
            "stats": {},
            "warnings": ["mesh has no faces"],
        }

    transform = {"translation": [0.0, 0.0, 0.0], "scale": 1.0}
    if recenter:
        vertices, transform = center_and_scale(vertices, domain)

    validation = validate(vertices, triangles)
    triangles, face_normals, vertex_normals = compute_normals(
        vertices, triangles
    )
    stats = mesh_stats(vertices, triangles, n_quads, n_ngons)

    if validation["degenerate_triangles"]:
        warnings.append(
            f"{validation['degenerate_triangles']} degenerate triangle(s)"
        )
    if not validation["watertight"]:
        if validation["boundary_edges"]:
            warnings.append(
                f"{validation['boundary_edges']} boundary edge(s): open mesh"
            )
        if validation["non_manifold_edges"]:
            warnings.append(
                f"{validation['non_manifold_edges']} non-manifold edge(s)"
            )
        if validation["flipped_edges"]:
            warnings.append(
                f"{validation['flipped_edges']} inconsistently wound edge(s)"
            )
    if validation["components"] > 1:
        warnings.append(f"{validation['components']} disconnected components")

    ok = stats["triangles"] > validation["degenerate_triangles"]
    return {
        "ok": ok,
        "vertices": vertices,
        "triangles": triangles,
        "face_normals": face_normals,
        "vertex_normals": vertex_normals,
        "transform": transform,
        "validation": validation,
        "stats": stats,
        "warnings": warnings,
    }


def write_obj(path, vertices, triangles, normals=None):
    """Write a triangulated mesh (optionally with per-vertex normals)."""
    lines = []
    for x, y, z in vertices:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    if normals is not None:
        for x, y, z in normals:
            lines.append(f"vn {x:.6f} {y:.6f} {z:.6f}")
        for a, b, c in triangles + 1:  # OBJ is 1-based
            lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
    else:
        for a, b, c in triangles + 1:
            lines.append(f"f {a} {b} {c}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# Self-tests (run in CI; numpy-only, no asset or network dependency)


def _cube(h=1.0, center=(0.0, 0.0, 0.0)):
    cx, cy, cz = center
    verts = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ], dtype=np.float64) + np.array([cx, cy, cz])
    faces = [
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ]
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

    # Watertight cube: closed, manifold, consistently wound, one piece.
    verts, faces = _cube()
    tris, nq, _ = triangulate(verts, faces)
    val = validate(verts, tris)
    check(val["watertight"], "cube is watertight")
    check(val["boundary_edges"] == 0, "cube has no boundary edges")
    check(val["non_manifold_edges"] == 0, "cube has no non-manifold edges")
    check(val["flipped_edges"] == 0, "cube winding is consistent")
    check(val["components"] == 1, "cube is one component")
    check(val["degenerate_triangles"] == 0, "cube has no degenerate tris")

    # Outward normals: every face normal points away from the center.
    _, fn, _ = compute_normals(verts, tris)
    centroids = verts[tris].mean(axis=1)
    check(bool(np.all(np.einsum("ij,ij->i", fn, centroids) > 0)),
          "face normals point outward")

    # Quad triangulation along the shorter diagonal.
    quad_v = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]],
                      dtype=np.float64)
    quad_t, nq2, _ = triangulate(quad_v, [[0, 1, 2, 3]])
    check(quad_t.shape[0] == 2 and nq2 == 1, "quad split into two triangles")

    # Open box (cube minus its +Z cap) has boundary edges and is not closed.
    open_faces = [f for f in faces if f not in ([4, 6, 5], [4, 7, 6])]
    open_t, _, _ = triangulate(verts, open_faces)
    open_val = validate(verts, open_t)
    check(open_val["boundary_edges"] > 0, "open mesh has boundary edges")
    check(not open_val["watertight"], "open mesh is not watertight")

    # One reversed face makes the winding inconsistent.
    flip_faces = [list(f) for f in faces]
    flip_faces[0] = flip_faces[0][::-1]
    flip_t, _, _ = triangulate(verts, flip_faces)
    check(validate(verts, flip_t)["flipped_edges"] > 0,
          "reversed face flagged as inconsistent winding")

    # Degenerate triangle (repeated vertex) is detected.
    deg_t = np.vstack([tris, [[0, 0, 1]]])
    check(validate(verts, deg_t)["degenerate_triangles"] >= 1,
          "degenerate triangle detected")

    # Two separate cubes are two components.
    v2, f2 = _cube(center=(5.0, 0.0, 0.0))
    both_v = np.vstack([verts, v2])
    both_f = faces + [[i + 8 for i in f] for f in f2]
    both_t, _, _ = triangulate(both_v, both_f)
    check(validate(both_v, both_t)["components"] == 2,
          "two cubes are two components")

    # center_and_scale centers the bbox and scales the longest axis.
    raw = np.array([[1, 2, 3], [5, 2, 3], [1, 4, 3], [1, 2, 9]],
                   dtype=np.float64)
    scaled, tf = center_and_scale(raw, domain=2.0)
    lo, hi = scaled.min(axis=0), scaled.max(axis=0)
    check(np.allclose(0.5 * (lo + hi), 0.0), "scaled mesh is centered")
    check(np.isclose((hi - lo).max(), 2.0), "longest axis scaled to domain")

    # Full pipeline on a quad cube reports clean geometry.
    quad_cube = (
        "v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\n"
        "v -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1\n"
        "f 1 2 3 4\nf 5 8 7 6\nf 1 5 6 2\n"
        "f 3 7 8 4\nf 1 4 8 5\nf 2 6 7 3\n"
    )
    res = preprocess(quad_cube)
    check(res["ok"], "pipeline accepts a clean quad cube")
    check(res["stats"]["quads_split"] == 6, "all six quads triangulated")
    check(res["validation"]["watertight"], "quad cube is watertight")

    print(f"{total - failures}/{total} checks passed")
    return failures


def _format_report(name, result):
    s = result["stats"]
    v = result["validation"]
    lines = [
        f"{name}",
        f"  vertices={s['vertices']} triangles={s['triangles']} "
        f"(quads_split={s['quads_split']} ngons_split={s['ngons_split']})",
        f"  bbox_size={[round(x, 4) for x in s['bbox_size']]} "
        f"area={s['surface_area']:.4f} volume={s['volume']:.4f}",
        f"  watertight={v['watertight']} components={v['components']} "
        f"degenerate={v['degenerate_triangles']} "
        f"boundary={v['boundary_edges']} non_manifold={v['non_manifold_edges']}"
        f" flipped={v['flipped_edges']}",
    ]
    for w in result["warnings"]:
        lines.append(f"  WARN: {w}")
    return "\n".join(lines)


def _build_parser():
    ap = argparse.ArgumentParser(description="Preprocess an OBJ mesh")
    ap.add_argument("input", nargs="?", help="input OBJ path")
    ap.add_argument("--domain", type=float, default=DEFAULT_DOMAIN,
                    help="longest-axis size after scaling")
    ap.add_argument("--no-center", dest="center", action="store_false",
                    help="keep input coordinates (no center/scale)")
    ap.add_argument("--out", help="write the cleaned triangulated OBJ here")
    ap.add_argument("--json", action="store_true",
                    help="print stats and validation as JSON")
    ap.add_argument("--selftest", action="store_true",
                    help="run built-in tests and exit")
    return ap


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.selftest:
        return 1 if _selftest() else 0

    if not args.input:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "simulation"
        root = root / "assets" / "3d-files"
        bundled = ["ahmed_25deg_m.obj", "ahmed_35deg_m.obj", "car-model.obj"]
        for name in bundled:
            res = preprocess(root / name, args.domain, args.center)
            print(_format_report(name, res))
        return 0

    res = preprocess(args.input, args.domain, args.center)
    if args.json:
        print(json.dumps(
            {"ok": res["ok"], "transform": res["transform"],
             "validation": res["validation"], "stats": res["stats"],
             "warnings": res["warnings"]},
            indent=2,
        ))
    else:
        print(_format_report(args.input, res))
    if args.out:
        write_obj(args.out, res["vertices"], res["triangles"],
                  res["vertex_normals"])
        print(f"wrote {args.out}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
