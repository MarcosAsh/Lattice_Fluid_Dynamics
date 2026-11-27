import numpy as np
import pytest

from meshes import cube


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def cube_mesh():
    verts, faces = cube()
    return verts, np.asarray(faces, dtype=np.int32)
