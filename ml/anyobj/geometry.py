"""
Geometry encoder for the any-OBJ Cd/Cl surrogate.

Turns any mesh into the two inputs the surrogate uses: a fixed descriptor
vector (Phase 1 MLP) and a solid voxel grid (Phase 2 CNN). Both are built in
the solver's frame: the mesh is centered, its longest axis is rotated to X (the
flow direction), and it is scaled into the [-1, 1]^3 cube. This matches
simulation/src/voxelize.c so the surrogate sees the body the way the GPU does.

Usage:
    from ml.anyobj.geometry import encode

    enc = encode("body.obj")
    enc["descriptors"]  # (N_DESCRIPTORS,) float32
    enc["voxels"]       # (res, res, res) uint8

    python geometry.py                    # report on the three bundled OBJs
    python geometry.py a.obj --voxel-res 48
"""

import numpy as np
import trimesh

PADDING = 0.05          # longest axis spans 2 - 2*PADDING inside [-1, 1]^3
DEFAULT_VOXEL_RES = 32
N_STATIONS = 8          # cross-section samples along the flow axis

# Descriptor order is the contract with the dataset and training scripts.
# Append only, never reorder. The trailing entries are the area profile.
DESCRIPTOR_NAMES = [
    "aspect_wl",        # width / length
    "aspect_hl",        # height / length
    "aspect_wh",        # width / height
    "bbox_solidity",    # volume / bbox volume
    "convexity",        # volume / convex-hull volume
    "slenderness",      # length / sqrt(width * height)
    "frontal_fill",     # frontal area / (width * height)
    "fineness",         # length / equivalent frontal diameter
    "wetted_ratio",     # surface area / frontal area
    "sphericity",       # 1.0 for a sphere, lower for elongated bodies
    "centroid_fore_aft",  # signed shift of the volume centroid along X
] + [f"area_{i}" for i in range(N_STATIONS)]

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)


# Mesh loading and canonicalization

def load_mesh(source, file_type=None):
    """
    Load a mesh from a path, raw bytes, inline OBJ text, or a Trimesh.

    Paths may be any format trimesh reads (OBJ, STL, PLY, glTF/GLB, OFF,
    3MF, ...); the format is inferred from the extension. For raw bytes pass
    file_type (e.g. "stl"); it defaults to "obj". Multi-object scenes are
    merged into one mesh, matching how the solver loads a model.

    Returns:
        A trimesh.Trimesh.
    """
    if isinstance(source, trimesh.Trimesh):
        return source
    if isinstance(source, (bytes, bytearray)):
        import io
        mesh = trimesh.load(io.BytesIO(bytes(source)), file_type=file_type or "obj")
    elif isinstance(source, str) and "\nv " in source:
        import io
        mesh = trimesh.load(io.StringIO(source), file_type="obj")
    else:
        mesh = trimesh.load(str(source))

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def canonicalize(mesh):
    """
    Center the mesh, rotate its longest axis to X, and scale into [-1, 1]^3.

    Mirrors copy_and_normalize_vertices in simulation/src/voxelize.c so X is
    the flow axis and the body fills the same cube the solver voxelizes into.

    Returns:
        A new trimesh.Trimesh in the canonical frame.
    """
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounding_box.centroid)

    # Swap the longest extent into X. The solver does the same single swap.
    longest = int(np.argmax(mesh.extents))
    if longest != 0:
        perm = [0, 1, 2]
        perm[0], perm[longest] = perm[longest], perm[0]
        mesh.vertices = mesh.vertices[:, perm]
        mesh.invert()  # the axis swap flips face winding

    max_dim = float(mesh.extents.max())
    if max_dim < 1e-8:
        raise ValueError("degenerate mesh: zero extent")
    mesh.apply_scale((2.0 - 2.0 * PADDING) / max_dim)
    return mesh


# Voxelization

def voxelize(mesh, res):
    """
    Solid occupancy grid over [-1, 1]^3.

    A cell is solid if its center is inside the mesh. X is the flow axis.

    Args:
        mesh: canonical trimesh.Trimesh.
        res: grid edge length.

    Returns:
        (res, res, res) uint8 array.
    """
    c = (np.arange(res) + 0.5) / res * 2.0 - 1.0  # cell centers in [-1, 1]
    gx, gy, gz = np.meshgrid(c, c, c, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    return mesh.contains(pts).reshape(res, res, res).astype(np.uint8)


# Descriptors

def _area_profile(voxels):
    """Cross-section occupancy at N_STATIONS stations along X, peak-normalized."""
    res = voxels.shape[0]
    per_slice = voxels.reshape(res, -1).sum(axis=1).astype(np.float64)
    edges = np.linspace(0, res, N_STATIONS + 1).astype(int)
    areas = np.array([per_slice[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    peak = areas.max()
    return areas / peak if peak > 0 else areas


def _descriptors(mesh, voxels):
    """
    Scale-invariant geometric descriptors on the canonical mesh.

    Returns:
        (N_DESCRIPTORS,) float32 array in DESCRIPTOR_NAMES order.
    """
    ex, ey, ez = mesh.extents  # ex is the flow-axis (longest) extent
    eps = 1e-9

    # Closed solids give a real volume; otherwise fall back to the convex hull.
    if mesh.is_watertight and mesh.volume > eps:
        vol = float(mesh.volume)
        centroid = mesh.center_mass
    else:
        hull = mesh.convex_hull
        vol = float(abs(hull.volume))
        centroid = hull.center_mass
    hull_vol = max(float(abs(mesh.convex_hull.volume)), eps)
    area = float(mesh.area)

    cell = 2.0 / voxels.shape[0]
    frontal_area = int(voxels.any(axis=0).sum()) * cell * cell
    equiv_diam = max((4.0 * frontal_area / np.pi) ** 0.5, eps)

    d = [
        ey / max(ex, eps),
        ez / max(ex, eps),
        ey / max(ez, eps),
        vol / max(ex * ey * ez, eps),
        vol / hull_vol,
        ex / max((ey * ez) ** 0.5, eps),
        frontal_area / max(ey * ez, eps),
        ex / equiv_diam,
        area / max(frontal_area, eps),
        (np.pi ** (1.0 / 3.0)) * (6.0 * vol) ** (2.0 / 3.0) / max(area, eps),
        float(centroid[0]) / max(ex, eps),
    ]
    d.extend(_area_profile(voxels).tolist())
    return np.array(d, dtype=np.float32)


def encode(source, voxel_res=DEFAULT_VOXEL_RES):
    """
    Encode a mesh into descriptors and a voxel grid.

    Args:
        source: OBJ path, bytes, inline text, or a Trimesh.
        voxel_res: voxel grid edge length.

    Returns:
        dict with keys: descriptors, voxels, voxel_res, watertight,
        fill_fraction.
    """
    mesh = canonicalize(load_mesh(source))
    voxels = voxelize(mesh, voxel_res)
    return {
        "descriptors": _descriptors(mesh, voxels),
        "voxels": voxels,
        "voxel_res": voxel_res,
        "watertight": bool(mesh.is_watertight),
        "fill_fraction": float(voxels.mean()),
    }


# Report

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "simulation" / "assets" / "3d-files"
    defaults = [str(root / f) for f in
                ("ahmed_25deg_m.obj", "ahmed_35deg_m.obj", "car-model.obj")]

    ap = argparse.ArgumentParser(description="Encode OBJ geometry for the surrogate")
    ap.add_argument("objs", nargs="*", default=defaults)
    ap.add_argument("--voxel-res", type=int, default=DEFAULT_VOXEL_RES)
    args = ap.parse_args()

    encs = {Path(p).stem: encode(p, args.voxel_res) for p in args.objs}

    print(f"voxel res {args.voxel_res}^3, {N_DESCRIPTORS} descriptors\n")
    head = f"{'descriptor':<20}" + "".join(f"{n:>14}" for n in encs)
    print(head)
    print("-" * len(head))
    for i, name in enumerate(DESCRIPTOR_NAMES):
        print(f"{name:<20}" + "".join(f"{e['descriptors'][i]:>14.4f}" for e in encs.values()))
    print("-" * len(head))
    print(f"{'watertight':<20}" + "".join(f"{str(e['watertight']):>14}" for e in encs.values()))
    print(f"{'fill_fraction':<20}" + "".join(f"{e['fill_fraction']:>14.4f}" for e in encs.values()))
