"""
Generate a family of parametric bluff bodies as OBJ files.

Three reference meshes can't teach the surrogate geometry: it only ever sees
three descriptor points. This writes a spread of primitives (boxes, ellipsoids,
capsules, cones, cylinders) across varied aspect ratios so the descriptor space
is sampled densely enough to generalize. Generation is deterministic, so the
meshes are reproducible from this script and are not committed.

Usage:
    python generate_meshes.py                 # writes meshes/*.obj
    python generate_meshes.py --report        # also print descriptor coverage
"""

import argparse
import sys
from pathlib import Path

import trimesh

OUT_DIR = Path(__file__).resolve().parent / "meshes"


def _ellipsoid(a, b, c):
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    m.apply_scale([a, b, c])
    return m


def build_meshes():
    """Return {name: Trimesh}. Names are filename-safe and deterministic."""
    meshes = {}

    # Boxes at varied length/width/height ratios.
    for L, W, H in [(2, 1, 1), (3, 1, 1), (4, 1, 1), (2, 2, 1)]:
        meshes[f"box_{L}{W}{H}"] = trimesh.creation.box(extents=(L, W, H))

    # Ellipsoids: smooth bodies from sphere to slender.
    for a, b, c in [(2, 1, 1), (3, 1, 1), (2, 1.5, 1)]:
        tag = f"{a}_{b}_{c}".replace(".", "p")
        meshes[f"ellip_{tag}"] = _ellipsoid(a, b, c)

    # Capsules: streamlined, high fineness ratio.
    for h, r in [(2, 0.5), (4, 0.4)]:
        meshes[f"caps_{h}_{int(r * 10)}"] = trimesh.creation.capsule(height=h, radius=r)

    # Cone (nose body) and cylinders (axisymmetric bluff bodies).
    meshes["cone_3_1"] = trimesh.creation.cone(radius=1.0, height=3.0)
    meshes["cyl_3_1"] = trimesh.creation.cylinder(radius=1.0, height=3.0)
    meshes["cyl_2_1"] = trimesh.creation.cylinder(radius=1.0, height=2.0)

    return meshes


def main():
    ap = argparse.ArgumentParser(description="Generate parametric meshes")
    ap.add_argument("--report", action="store_true",
                    help="print descriptor coverage after writing")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    meshes = build_meshes()
    for name, mesh in meshes.items():
        mesh.export(OUT_DIR / f"{name}.obj")
    print(f"Wrote {len(meshes)} meshes to {OUT_DIR}")

    if args.report:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from geometry import encode

        keys = ["aspect_wl", "slenderness", "fineness", "convexity", "sphericity"]
        print(f"\n{'mesh':<14}" + "".join(f"{k:>12}" for k in keys))
        for name in meshes:
            enc = encode(str(OUT_DIR / f"{name}.obj"))
            d = dict(zip(__import__("geometry").DESCRIPTOR_NAMES, enc["descriptors"]))
            print(f"{name:<14}" + "".join(f"{d[k]:>12.3f}" for k in keys))


if __name__ == "__main__":
    main()
