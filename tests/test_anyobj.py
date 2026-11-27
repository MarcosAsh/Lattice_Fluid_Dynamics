"""Read-only checks on the any-OBJ surrogate, skipped without the jax stack"""

import sys
from pathlib import Path

import numpy as np
import pytest

ANYOBJ = Path(__file__).resolve().parents[1] / "ml" / "anyobj"


@pytest.fixture(scope="module")
def anyobj():
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    trimesh = pytest.importorskip("trimesh")
    saved = list(sys.path)
    sys.path.insert(0, str(ANYOBJ))
    import geometry
    import train

    yield type("Mods", (), {"geometry": geometry, "train": train, "trimesh": trimesh})
    sys.path[:] = saved
    for name in ("geometry", "train"):
        sys.modules.pop(name, None)


def test_encode_box_is_finite_and_deterministic(anyobj):
    box = anyobj.trimesh.creation.box(extents=(3, 1, 1))
    a = anyobj.geometry.encode(box, voxel_res=16)
    b = anyobj.geometry.encode(box, voxel_res=16)
    assert a["descriptors"].shape == (anyobj.geometry.N_DESCRIPTORS,)
    assert np.all(np.isfinite(a["descriptors"]))
    assert np.array_equal(a["descriptors"], b["descriptors"])
    assert a["voxels"].shape == (16, 16, 16)
    assert a["watertight"]
    assert 0.0 < a["fill_fraction"] < 1.0


def test_encode_is_invariant_to_scale_and_translation(anyobj):
    small = anyobj.trimesh.creation.box(extents=(2, 1, 1))
    big = anyobj.trimesh.creation.box(extents=(20, 10, 10))
    big.apply_translation((5, 5, 5))
    a = anyobj.geometry.encode(small, voxel_res=16)["descriptors"]
    b = anyobj.geometry.encode(big, voxel_res=16)["descriptors"]
    assert np.allclose(a, b, atol=1e-4)


def test_zscore_handles_constant_columns(anyobj):
    arr = np.array([[1.0, 5.0], [3.0, 5.0]])
    mean, std = anyobj.train.zscore(arr)
    assert mean.tolist() == [2.0, 5.0]
    assert std.tolist() == [1.0, 1.0]


def test_params_round_trip_and_forward_shape(anyobj, tmp_path):
    import jax

    t = anyobj.train
    params = t.init_params(jax.random.PRNGKey(0), 8)
    path = tmp_path / "p.npz"
    t.save_params(path, params, 8)
    loaded = t.load_params(path)
    x = np.ones((3, t.N_INPUTS), dtype=np.float32)
    assert t.forward(loaded, x).shape == (3, t.N_TARGETS)
    assert np.allclose(t.forward(params, x), t.forward(loaded, x))
