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
from pathlib import Path

import numpy as np

DEFAULT_DOMAIN = 2.0  # longest axis scaled to this, matching [-1, 1]^3
EPS = 1e-12


# Mesh loading


def _read_text(source: str | bytes | Path) -> str:
    """Return OBJ text from a path, raw bytes, or inline OBJ text."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source).decode("utf-8", "replace")
    if isinstance(source, str) and "\n" in source and "v " in source:
        return source
    with Path(source).open() as f:
        return f.read()


def load_obj(source: str | bytes | Path) -> tuple[np.ndarray, list[list[int]]]:
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


def triangulate(vertices: np.ndarray, faces: list[list[int]]) -> tuple[np.ndarray, int, int]:
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
            tris.extend((face[0], face[i], face[i + 1]) for i in range(1, k - 1))
    triangles = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    return triangles, n_quads, n_ngons


# Center and scale


def center_and_scale(
    vertices: np.ndarray, domain: float = DEFAULT_DOMAIN
) -> tuple[np.ndarray, dict]:
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


def face_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Per-triangle area (float64, shape (M,))."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def signed_volume6(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Six times the signed volume (sign encodes winding orientation)."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum())


def _edge_counts(triangles: np.ndarray) -> dict[tuple[int, int], list[int]]:
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


def _count_components(triangles: np.ndarray, n_vertices: int) -> int:
    """Number of connected components over vertices sharing a triangle."""
    parent = list(range(n_vertices))

    def find(x: int) -> int:
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


def validate(vertices: np.ndarray, triangles: np.ndarray) -> dict:
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


def compute_normals(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def mesh_stats(
    vertices: np.ndarray, triangles: np.ndarray, n_quads: int = 0, n_ngons: int = 0
) -> dict:
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


def preprocess(
    source: str | bytes | Path, domain: float = DEFAULT_DOMAIN, recenter: bool = True
) -> dict:
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
    triangles, face_normals, vertex_normals = compute_normals(vertices, triangles)
    stats = mesh_stats(vertices, triangles, n_quads, n_ngons)

    if validation["degenerate_triangles"]:
        warnings.append(f"{validation['degenerate_triangles']} degenerate triangle(s)")
    if not validation["watertight"]:
        if validation["boundary_edges"]:
            warnings.append(f"{validation['boundary_edges']} boundary edge(s): open mesh")
        if validation["non_manifold_edges"]:
            warnings.append(f"{validation['non_manifold_edges']} non-manifold edge(s)")
        if validation["flipped_edges"]:
            warnings.append(f"{validation['flipped_edges']} inconsistently wound edge(s)")
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


def write_obj(
    path: str | Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    normals: np.ndarray | None = None,
) -> None:
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
    with Path(path).open("w") as f:
        f.write("\n".join(lines) + "\n")


def _selftest() -> int:
    try:
        import pytest
    except ImportError:
        sys.exit("pytest is required for --selftest, install it with `uv sync`")
    test_file = Path(__file__).resolve().parents[1] / "tests" / "test_preprocess.py"
    return pytest.main(["-q", str(test_file)])


def _format_report(name: object, result: dict) -> str:
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
    lines.extend(f"  WARN: {w}" for w in result["warnings"])
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Preprocess an OBJ mesh")
    ap.add_argument("input", nargs="?", help="input OBJ path")
    ap.add_argument(
        "--domain", type=float, default=DEFAULT_DOMAIN, help="longest-axis size after scaling"
    )
    ap.add_argument(
        "--no-center",
        dest="center",
        action="store_false",
        help="keep input coordinates (no center/scale)",
    )
    ap.add_argument("--out", help="write the cleaned triangulated OBJ here")
    ap.add_argument("--json", action="store_true", help="print stats and validation as JSON")
    ap.add_argument("--selftest", action="store_true", help="run built-in tests and exit")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.selftest:
        return int(_selftest())

    if not args.input:
        root = Path(__file__).resolve().parent.parent / "simulation"
        root = root / "assets" / "3d-files"
        bundled = ["ahmed_25deg_m.obj", "ahmed_35deg_m.obj", "car-model.obj"]
        for name in bundled:
            res = preprocess(root / name, args.domain, args.center)
            print(_format_report(name, res))
        return 0

    res = preprocess(args.input, args.domain, args.center)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": res["ok"],
                    "transform": res["transform"],
                    "validation": res["validation"],
                    "stats": res["stats"],
                    "warnings": res["warnings"],
                },
                indent=2,
            )
        )
    else:
        print(_format_report(args.input, res))
    if args.out:
        write_obj(args.out, res["vertices"], res["triangles"], res["vertex_normals"])
        print(f"wrote {args.out}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
