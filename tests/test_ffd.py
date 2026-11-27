import numpy as np
import pytest

import ffd

CUBE_VERTS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.float64,
)
CUBE_FACES = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [0, 3, 7],
        [0, 7, 4],
        [1, 2, 6],
        [1, 6, 5],
    ],
    dtype=np.int64,
)


@pytest.fixture
def lattice():
    return ffd.FFDLattice([0, 0, 0], [1, 1, 1], n_control=(4, 3, 3))


def test_zero_displacement_is_identity(lattice):
    out = lattice.deform(CUBE_VERTS, np.zeros(lattice.get_num_params()))
    assert np.max(np.abs(out - CUBE_VERTS)) < 1e-12


def test_uniform_shift_translates_everything():
    lat = ffd.FFDLattice([0, 0, 0], [1, 1, 1], n_control=(2, 2, 2))
    shift = np.zeros((2, 2, 2, 3))
    shift[..., 0] = 0.5
    out = lat.deform(CUBE_VERTS, shift.ravel())
    expected = CUBE_VERTS.copy()
    expected[:, 0] += 0.5
    assert np.max(np.abs(out - expected)) < 1e-12


def test_parameter_counts(lattice):
    assert lattice.get_num_params() == 4 * 3 * 3 * 3
    # ny=3: one mirrored row (4*1*3*3) plus a midplane row without y (4*1*3*2)
    assert lattice.get_num_free_params() == 60


def test_symmetry_mirrors_displacements(lattice, rng):
    raw = rng.standard_normal(lattice.get_num_params()) * 0.01
    sym = lattice.enforce_symmetry(raw, axis=1).reshape(4, 3, 3, 3)
    assert np.allclose(sym[:, 0, :, 0], sym[:, 2, :, 0])
    assert np.allclose(sym[:, 0, :, 2], sym[:, 2, :, 2])
    assert np.allclose(sym[:, 0, :, 1], -sym[:, 2, :, 1])
    assert np.allclose(sym[:, 1, :, 1], 0.0)


def test_symmetry_is_idempotent(lattice, rng):
    once = lattice.enforce_symmetry(rng.standard_normal(lattice.get_num_params()))
    assert np.allclose(lattice.enforce_symmetry(once), once)


def test_clamp_limits_each_axis(lattice):
    clamped = lattice.clamp_displacements(np.full(lattice.get_num_params(), 10.0), 0.15)
    clamped = clamped.reshape(4, 3, 3, 3)
    for comp in range(3):
        assert np.all(np.abs(clamped[..., comp]) <= 0.15 * lattice.extent[comp] + 1e-14)


def test_obj_round_trip(tmp_path):
    path = tmp_path / "cube.obj"
    ffd.save_obj(path, CUBE_VERTS, CUBE_FACES)
    verts, faces = ffd.load_obj(path)
    assert np.allclose(verts, CUBE_VERTS, atol=1e-7)
    assert np.array_equal(faces, CUBE_FACES)


def test_load_obj_fans_quads(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1 2/2 3/3 4/4\n")
    _, faces = ffd.load_obj(path)
    assert faces.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_deform_obj_identity(tmp_path):
    src, dst = tmp_path / "a.obj", tmp_path / "b.obj"
    ffd.save_obj(src, CUBE_VERTS, CUBE_FACES)
    ffd.deform_obj(src, dst, np.zeros(3 * 3 * 3 * 3), n_control=(3, 3, 3))
    verts, _ = ffd.load_obj(dst)
    assert np.allclose(verts, CUBE_VERTS, atol=1e-10)


def test_compute_bbox_pads_each_side():
    lo, hi = ffd.compute_bbox(CUBE_VERTS, padding=0.1)
    assert np.allclose(lo, -0.1)
    assert np.allclose(hi, 1.1)


def test_degenerate_bbox_is_rejected():
    with pytest.raises(ValueError, match="strictly greater"):
        ffd.FFDLattice([0, 0, 0], [1, 0, 1])


def test_lhs_samples_are_symmetric_and_bounded(lattice):
    samples = lattice.sample_random(10, max_fraction=0.15, symmetric=True)
    assert samples.shape == (10, 108)
    for s in samples:
        assert np.allclose(lattice.enforce_symmetry(s), s, atol=1e-12)
    assert np.all(np.abs(samples.reshape(10, -1, 3)) <= 0.15 + 1e-12)


def test_lhs_unsymmetric_shape(lattice):
    assert lattice.sample_random(5, symmetric=False).shape == (5, lattice.get_num_params())


def test_lhs_is_reproducible_with_a_generator(lattice):
    a = lattice.sample_random(6, rng=np.random.default_rng(7))
    b = lattice.sample_random(6, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("degree", [2, 3, 5])
def test_bernstein_partition_of_unity(degree):
    basis = ffd._bernstein_basis(degree, np.linspace(0, 1, 50))
    assert np.allclose(basis.sum(axis=1), 1.0, atol=1e-14)
