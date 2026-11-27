"""
Convert STL or OBJ meshes to the OBJ the solver reads.

The automotive datasets (AhmedML, WindsorML, DrivAerML) ship STL. The
solver wants OBJ with triangle faces and positive indices. This script
reads binary or ASCII STL (or an OBJ to normalise), merges duplicate
vertices, drops degenerate triangles and writes a compact OBJ.

Usage:
    python ml/mesh_convert.py body.stl                 # writes body.obj
    python ml/mesh_convert.py body.stl -o out.obj --center --scale 0.001
    python ml/mesh_convert.py messy.obj -o clean.obj --merge-tol 1e-6 --json
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import TextIO

import numpy as np

import preprocess

STL_HEADER = 80
STL_RECORD = np.dtype(
    [("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")],
)
# height over longest edge below which a float32 normal is noise
FLAT_TOL = 1e-7
CHUNK_BYTES = 64 << 20
VERTEX_RE = re.compile(rb"^[ \t]*vertex[ \t]+(.*?)[ \t]*\r?$", re.MULTILINE)


class MeshError(ValueError):
    """Raised for unreadable or empty meshes"""


def stl_kind(path: Path) -> str:
    """Return "binary" or "ascii" for an STL file"""
    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(STL_HEADER + 4)
    if len(head) == STL_HEADER + 4:
        count = int(np.frombuffer(head, "<u4", 1, STL_HEADER)[0])
        if size == STL_HEADER + 4 + count * STL_RECORD.itemsize:
            return "binary"
    if head.lstrip().startswith(b"solid"):
        with path.open("rb") as f:
            sample = f.read(4096)
        if b"\0" not in sample and (b"facet" in sample or b"endsolid" in sample):
            return "ascii"
    if len(head) < STL_HEADER + 4:
        raise MeshError(f"{path}: {size} bytes is too short for an STL")
    return "binary"


def read_binary_stl(path: Path) -> tuple[np.ndarray, int]:
    """Read a binary STL into a (M, 3, 3) float32 array plus trailing byte count"""
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(STL_HEADER)
        count = int(np.fromfile(f, "<u4", 1)[0])
        need = STL_HEADER + 4 + count * STL_RECORD.itemsize
        if size < need:
            raise MeshError(
                f"{path}: binary STL truncated, header says {count} triangles "
                f"({need} bytes) but the file has {size}"
            )
        records = np.fromfile(f, STL_RECORD, count)
    return np.ascontiguousarray(records["v"]), size - need


def read_ascii_stl(path: Path) -> np.ndarray:
    """Read an ASCII STL into a (M, 3, 3) float64 array"""
    parts = []
    tail = b""
    saw_end = False
    with path.open("rb") as f:
        while True:
            block = f.read(CHUNK_BYTES)
            data = tail + block
            if block:
                cut = data.rfind(b"\n") + 1
                data, tail = data[:cut], data[cut:]
            lines = VERTEX_RE.findall(data)
            if lines:
                try:
                    xyz = np.loadtxt(io.BytesIO(b"\n".join(lines)), ndmin=2)
                except ValueError as e:
                    raise MeshError(f"{path}: bad vertex line: {e}") from e
                if xyz.shape[1] != 3:
                    raise MeshError(f"{path}: vertex lines need 3 numbers, got {xyz.shape[1]}")
                parts.append(xyz)
                saw_end = data.rfind(b"endsolid") > data.rfind(b"vertex")
            elif b"endsolid" in data:
                saw_end = True
            if not block:
                break
    corners = np.concatenate(parts) if parts else np.zeros((0, 3))
    if len(corners) % 3:
        raise MeshError(
            f"{path}: ASCII STL truncated, {len(corners)} vertex lines is not a multiple of 3"
        )
    if len(corners) and not saw_end:
        raise MeshError(f"{path}: ASCII STL truncated, no endsolid after the last facet")
    return corners.reshape(-1, 3, 3)


def read_obj(path: Path) -> tuple[np.ndarray, int, int]:
    """Read an OBJ into a (M, 3, 3) array plus split quad and n-gon counts"""
    try:
        verts, faces = preprocess.load_obj(path)
    except (ValueError, IndexError) as e:
        raise MeshError(f"{path}: cannot parse OBJ: {e}") from e
    bad = [f for f in faces if min(f) < 0 or max(f) >= len(verts)]
    if bad:
        raise MeshError(f"{path}: face index out of range: {[i + 1 for i in bad[0]]}")
    tris, quads, ngons = preprocess.triangulate(verts, faces)
    return verts[tris], quads, ngons


def read_mesh(path: Path) -> tuple[np.ndarray, dict]:
    """Read STL or OBJ triangles and describe the source"""
    if not path.is_file():
        raise MeshError(f"{path}: no such file")
    suffix = path.suffix.lower()
    if suffix == ".obj":
        corners, quads, ngons = read_obj(path)
        return corners, {"format": "obj", "quads_split": quads, "ngons_split": ngons}
    if suffix != ".stl":
        raise MeshError(f"{path}: unknown mesh type {suffix!r}, expected .stl or .obj")
    if stl_kind(path) == "ascii":
        return read_ascii_stl(path), {"format": "stl-ascii"}
    corners, extra = read_binary_stl(path)
    source = {"format": "stl-binary"}
    if extra:
        source["trailing_bytes"] = extra
    return corners, source


def merge_vertices(points: np.ndarray, tol: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Merge equal points, or points equal after rounding to tol, in first-seen order"""
    # adding 0.0 turns -0.0 into 0.0
    key = np.round(points.astype(np.float64) / tol).astype(np.int64) if tol else points + 0.0
    order = np.lexsort(key.T[::-1])
    ranked = key[order]
    new = np.ones(len(ranked), dtype=bool)
    np.any(ranked[1:] != ranked[:-1], axis=1, out=new[1:])
    group = np.cumsum(new) - 1
    # lexsort is stable, so each group starts at its first occurrence
    first = order[new]
    seen = np.argsort(first, kind="stable")
    rank = np.empty_like(seen)
    rank[seen] = np.arange(len(seen))
    inverse = np.empty(len(points), dtype=np.int64)
    inverse[order] = rank[group]
    return points[first[seen]], inverse


def degenerate_mask(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """True for triangles with a repeated vertex or no usable normal"""
    repeated = (tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) | (tris[:, 0] == tris[:, 2])
    p = verts[tris].astype(np.float64)
    e0 = p[:, 1] - p[:, 0]
    e1 = p[:, 2] - p[:, 1]
    e2 = p[:, 0] - p[:, 2]
    twice_area = np.linalg.norm(np.cross(e0, -e2), axis=1)
    longest = np.max([np.einsum("ij,ij->i", e, e) for e in (e0, e1, e2)], axis=0)
    flat = twice_area <= FLAT_TOL * longest
    return repeated | flat


def convert(
    corners: np.ndarray,
    merge_tol: float | None = None,
    scale: float | None = None,
    center: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Turn a triangle soup into indexed vertices and triangles with a report"""
    n_in = len(corners)
    if not n_in:
        raise MeshError("mesh has no triangles")
    finite = np.isfinite(corners).all(axis=(1, 2))
    corners = corners[finite]
    verts, idx = merge_vertices(corners.reshape(-1, 3), merge_tol)
    tris = idx.reshape(-1, 3)
    bad = degenerate_mask(verts, tris)
    tris = tris[~bad]
    if not len(tris):
        raise MeshError(f"all {n_in} triangles are degenerate or non-finite")
    used = np.zeros(len(verts), dtype=bool)
    used[tris] = True
    tris = (np.cumsum(used) - 1)[tris]
    verts = verts[used].astype(np.float64)

    if center:
        verts -= 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    if scale is not None:
        verts *= scale
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    report = {
        "triangles_in": int(n_in),
        "nonfinite_dropped": int(n_in - finite.sum()),
        "degenerate_dropped": int(bad.sum()),
        "triangles_out": len(tris),
        "corners_in": int(3 * finite.sum()),
        "vertices_out": len(verts),
        "merge_tol": merge_tol,
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "bbox_size": (hi - lo).tolist(),
    }
    return verts, tris, report


def _write_rows(f: TextIO, rows: np.ndarray, fmt: str, chunk: int = 65536) -> None:
    # one % per chunk, np.savetxt is too slow for millions of rows
    for start in range(0, len(rows), chunk):
        block = rows[start : start + chunk]
        f.write((fmt * len(block)) % tuple(block.ravel().tolist()))


def write_obj(path: Path, verts: np.ndarray, tris: np.ndarray, comment: str = "") -> None:
    """Write vertices and 1-based triangle faces"""
    with path.open("w") as f:
        if comment:
            f.write(f"# {comment}\n")
        _write_rows(f, verts, "v %.9g %.9g %.9g\n")
        _write_rows(f, tris + 1, "f %d %d %d\n")


def convert_file(
    src: Path,
    dst: Path | None,
    merge_tol: float | None = None,
    scale: float | None = None,
    center: bool = False,
    check: bool = False,
) -> dict:
    """Read, convert and optionally write a mesh, returning the report"""
    corners, source = read_mesh(src)
    verts, tris, report = convert(corners, merge_tol, scale, center)
    report = {"input": str(src), **source, **report}
    if check:
        report["validation"] = preprocess.validate(verts, tris)
    if dst is not None:
        write_obj(dst, verts, tris, f"converted from {src.name} by mesh_convert.py")
        report["output"] = str(dst)
    return report


def _vec(v: list[float]) -> str:
    return "(" + ", ".join(f"{x:.6g}" for x in v) + ")"


def format_report(r: dict) -> str:
    size = " x ".join(f"{s:.6g}" for s in r["bbox_size"])
    lines = [
        f"{r['input']} ({r['format']})",
        f"  triangles {r['triangles_in']} in, {r['triangles_out']} out "
        f"({r['degenerate_dropped']} degenerate, {r['nonfinite_dropped']} non-finite dropped)",
        f"  vertices {r['corners_in']} corners merged to {r['vertices_out']}",
        f"  bbox min {_vec(r['bbox_min'])} max {_vec(r['bbox_max'])} size {size}",
    ]
    if r.get("quads_split") or r.get("ngons_split"):
        lines.append(f"  split {r['quads_split']} quads and {r['ngons_split']} n-gons")
    if "validation" in r:
        v = r["validation"]
        lines.append(
            f"  watertight={v['watertight']} components={v['components']} "
            f"boundary={v['boundary_edges']} non_manifold={v['non_manifold_edges']} "
            f"flipped={v['flipped_edges']}"
        )
    if "output" in r:
        lines.append(f"wrote {r['output']}")
    return "\n".join(lines)


def _positive(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert STL or OBJ to a clean triangle OBJ")
    ap.add_argument("input", type=Path, help="STL (binary or ASCII) or OBJ")
    ap.add_argument("-o", "--output", type=Path, help="OBJ to write (default: input.obj)")
    ap.add_argument("--merge-tol", type=_positive, help="merge vertices equal after rounding")
    ap.add_argument("--scale", type=_positive, help="multiply coordinates after centering")
    ap.add_argument("--center", action="store_true", help="move the bbox center to the origin")
    ap.add_argument("--check", action="store_true", help="also check watertightness (slow)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)

    dst = None
    if not args.dry_run:
        dst = args.output or args.input.with_suffix(".obj")
        if dst.resolve() == args.input.resolve():
            ap.error(f"output would overwrite the input {args.input}, pass -o")
    try:
        report = convert_file(args.input, dst, args.merge_tol, args.scale, args.center, args.check)
    except (MeshError, OSError) as e:
        print(f"mesh_convert: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
