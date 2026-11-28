"""Write a unit icosphere (4 subdivisions, 5120 triangles) as OBJ."""

import sys
from pathlib import Path

import numpy as np


def icosphere(levels: int = 4) -> tuple[list, list]:
    t = (1 + 5**0.5) / 2
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]  # fmt: skip
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]  # fmt: skip
    verts = [list(np.asarray(v) / np.linalg.norm(v)) for v in verts]
    for _ in range(levels):
        cache: dict = {}

        def mid(a: int, b: int, cache: dict = cache) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = (np.asarray(verts[a]) + np.asarray(verts[b])) / 2
                verts.append(list(m / np.linalg.norm(m)))
                cache[key] = len(verts) - 1
            return cache[key]

        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new
    return verts, faces


if __name__ == "__main__":
    verts, faces = icosphere()
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    Path(sys.argv[1]).write_text("\n".join(lines) + "\n")
