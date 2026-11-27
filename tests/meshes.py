"""Tiny meshes shared by the tests"""

import numpy as np

CUBE_FACES = [
    [0, 1, 2],
    [0, 2, 3],
    [4, 6, 5],
    [4, 7, 6],
    [0, 4, 5],
    [0, 5, 1],
    [3, 2, 6],
    [3, 6, 7],
    [0, 3, 7],
    [0, 7, 4],
    [1, 5, 6],
    [1, 6, 2],
]


def cube(h=1.0, center=(0.0, 0.0, 0.0)):
    """Watertight cube with outward winding, as (vertices, faces)"""
    verts = np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float64,
    )
    return verts + np.asarray(center), [list(f) for f in CUBE_FACES]


def cube_obj_text():
    """Unit cube as OBJ text with quad faces"""
    return (
        "v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\n"
        "v -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1\n"
        "f 1 2 3 4\nf 5 8 7 6\nf 1 5 6 2\n"
        "f 3 7 8 4\nf 1 4 8 5\nf 2 6 7 3\n"
    )
