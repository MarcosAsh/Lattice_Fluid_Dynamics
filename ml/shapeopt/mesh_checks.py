"""
Geometry checks for a deformed mesh before it goes to the solver.

A shape fails when the mesh is not watertight, has degenerate triangles,
has triangles that turned over against the baseline, or when the FFD map
folds (its Jacobian determinant is not positive at some vertex). The
report also gives the change in frontal area, volume and length, since a
lower Cd from a smaller frontal area is not a better shape.

Usage:
    report = check(base_vertices, vertices, faces, lattice, displacements)
    if not report["ok"]:
        print(report["problems"])
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import signed_volume6
from preprocess import validate
from protocol import STREAMWISE
from protocol import extent


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = (vertices[faces[:, k]] for k in range(3))
    return np.cross(v1 - v0, v2 - v0)


def flipped_faces(base: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> int:
    """Triangles whose normal turned by more than 90 degrees"""
    dot = np.einsum("ij,ij->i", face_normals(base, faces), face_normals(vertices, faces))
    return int((dot <= 0).sum())


def frontal_area(vertices: np.ndarray, faces: np.ndarray, samples: int = 400) -> float:
    """Area of the silhouette seen along x, by rasterising the y-z projection"""
    v = np.asarray(vertices, dtype=np.float64)
    p = v[:, 1:]
    lo = p.min(axis=0)
    cell = float((p.max(axis=0) - lo).max()) / samples
    q = (p - lo) / cell
    size = np.ceil(q.max(axis=0)).astype(int) + 1
    mask = np.zeros(size, dtype=bool)
    # on a closed mesh the triangles facing one way along x cover the silhouette
    nx = face_normals(v, faces)[:, 0]
    half = faces[nx > 0] if (nx > 0).sum() <= (nx < 0).sum() else faces[nx < 0]
    for a, b, c in q[half]:
        det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(det) < 1e-12:
            continue
        y0, z0 = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        y1, z1 = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        ys, zs = np.meshgrid(
            np.arange(y0, y1 + 1) + 0.5, np.arange(z0, z1 + 1) + 0.5, indexing="ij"
        )
        # barycentric test on pixel centres
        w1 = ((ys - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (zs - a[1])) / det
        w2 = ((b[0] - a[0]) * (zs - a[1]) - (ys - a[0]) * (b[1] - a[1])) / det
        inside = (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
        mask[y0 : y1 + 1, z0 : z1 + 1] |= inside[: size[0] - y0, : size[1] - z0]
    return float(mask.sum()) * cell * cell


def min_jacobian(lattice, vertices: np.ndarray, displacements: np.ndarray) -> float:
    """Smallest determinant of the FFD map's Jacobian over the vertices"""
    h = 1e-4 * float(lattice.extent.min())
    cols = []
    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        hi = lattice.deform(vertices + step, displacements)
        lo = lattice.deform(vertices - step, displacements)
        cols.append((hi - lo) / (2 * h))
    jac = np.stack(cols, axis=-1)
    return float(np.linalg.det(jac).min())


def volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    return abs(signed_volume6(vertices, faces)) / 6.0


def check(
    base: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    lattice=None,
    displacements: np.ndarray | None = None,
    base_area: float | None = None,
) -> dict:
    """Validity and size change of a deformed mesh against its baseline"""
    base = np.asarray(base, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mesh = validate(vertices, faces)
    problems = []
    if not mesh["watertight"]:
        problems.append(
            f"not watertight ({mesh['boundary_edges']} boundary, "
            f"{mesh['non_manifold_edges']} non-manifold, {mesh['flipped_edges']} flipped edges)"
        )
    if mesh["degenerate_triangles"]:
        problems.append(f"{mesh['degenerate_triangles']} degenerate triangles")

    same_topology = base.shape == vertices.shape
    flipped = flipped_faces(base, vertices, faces) if same_topology else None
    if flipped:
        problems.append(f"{flipped} triangles turned over against the baseline")
    jac = None
    if lattice is not None and displacements is not None:
        jac = min_jacobian(lattice, base, displacements)
        if jac <= 0:
            problems.append(f"FFD map folds, min Jacobian determinant {jac:.3g}")

    size, size0 = extent(vertices), extent(base)
    if size[2] > size[STREAMWISE]:
        problems.append("longest side is no longer x, the solver would rotate the body")

    area = frontal_area(vertices, faces)
    area0 = base_area if base_area is not None else frontal_area(base, faces)
    vol, vol0 = volume(vertices, faces), volume(base, faces)
    return {
        "ok": not problems,
        "problems": problems,
        "watertight": mesh["watertight"],
        "degenerate_triangles": mesh["degenerate_triangles"],
        "flipped_faces": flipped,
        "min_jacobian": jac,
        "frontal_area": area,
        "frontal_area_ratio": area / area0,
        "volume": vol,
        "volume_ratio": vol / vol0,
        "length_ratio": float(size[STREAMWISE] / size0[STREAMWISE]),
    }
