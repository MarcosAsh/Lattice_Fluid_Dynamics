import struct

import numpy as np
import pytest

import gen_pairs as gp


def make_patches(rng, n):
    coarse = [rng.standard_normal((3, 3, 3, 4)).astype(np.float32) for _ in range(n)]
    fine = [rng.standard_normal((2, 2, 2, 4)).astype(np.float32) for _ in range(n)]
    return coarse, fine


def test_round_trip(tmp_path, rng):
    coarse, fine = make_patches(rng, 16)
    path = tmp_path / "p.srbin"
    gp.write_srbin(path, coarse, fine)
    coarse_out, fine_out = gp.read_srbin(path)
    assert coarse_out.shape == (16, 3, 3, 3, 4)
    assert fine_out.shape == (16, 2, 2, 2, 4)
    assert np.allclose(coarse_out, np.stack(coarse), atol=1e-7)
    assert np.allclose(fine_out, np.stack(fine), atol=1e-7)


def test_file_size_matches_header(tmp_path, rng):
    coarse, fine = make_patches(rng, 5)
    path = tmp_path / "p.srbin"
    gp.write_srbin(path, coarse, fine)
    magic, version, n, channels = struct.unpack("<IIII", path.read_bytes()[:16])
    assert (magic, version, n, channels) == (gp.SRBIN_MAGIC, 1, 5, 4)
    assert path.stat().st_size == 16 + 5 * (108 + 32) * 4


def test_mismatched_lengths_are_rejected(tmp_path, rng):
    coarse, fine = make_patches(rng, 3)
    with pytest.raises(ValueError, match="shorter"):
        gp.write_srbin(tmp_path / "p.srbin", coarse, fine[:2])


def test_read_rejects_bad_magic_and_version(tmp_path):
    bad = tmp_path / "bad.srbin"
    bad.write_bytes(struct.pack("<IIII", 0, 1, 0, 4))
    with pytest.raises(ValueError, match="Bad magic"):
        gp.read_srbin(bad)
    bad.write_bytes(struct.pack("<IIII", gp.SRBIN_MAGIC, 9, 0, 4))
    with pytest.raises(ValueError, match="version"):
        gp.read_srbin(bad)


def test_trainer_loader_agrees_with_writer(tmp_path, rng):
    pytest.importorskip("torch")
    import train_superres as ts

    coarse, fine = make_patches(rng, 8)
    path = tmp_path / "p.srbin"
    gp.write_srbin(path, coarse, fine)
    c, f = ts.load_srbin(str(path))
    assert c.shape == (8, ts.COARSE_DIM)
    assert f.shape == (8, ts.FINE_DIM)
    assert np.allclose(c, np.stack(coarse).reshape(8, -1))
    assert np.allclose(f, np.stack(fine).reshape(8, -1))


def test_downsample_averages_blocks_and_ors_solid():
    vel = np.zeros((2, 2, 2, 3), dtype=np.float32)
    vel[0, 0, 0] = [8, 0, 0]
    solid = np.zeros((2, 2, 2), dtype=np.int32)
    solid[1, 1, 1] = 1
    cvel, csolid = gp.downsample_2x(vel, solid)
    assert cvel.shape == (1, 1, 1, 3)
    assert cvel[0, 0, 0, 0] == pytest.approx(1.0)
    assert csolid.tolist() == [[[1]]]


def test_downsample_needs_even_dims():
    with pytest.raises(AssertionError, match="even"):
        gp.downsample_2x_scalar(np.zeros((3, 2, 2)))


def test_density_defaults_to_one():
    vel = np.zeros((2, 3, 4, 3), dtype=np.float32)
    rho = gp.compute_density(vel)
    assert rho.shape == (2, 3, 4)
    assert np.all(rho == 1.0)
    given = np.full((2, 3, 4), 0.9, dtype=np.float64)
    assert gp.compute_density(vel, given).dtype == np.float32


def test_extract_patches_counts_and_skips_solid(rng):
    fine_vel = rng.standard_normal((8, 8, 8, 3)).astype(np.float32)
    fine_solid = np.zeros((8, 8, 8), dtype=np.int32)
    coarse_vel, coarse_solid = gp.downsample_2x(fine_vel, fine_solid)
    c, f = gp.extract_patches(coarse_vel, coarse_solid, fine_vel, fine_solid)
    # 4x4x4 coarse grid leaves a 2x2x2 interior for 3x3x3 neighbourhoods
    assert len(c) == len(f) == 8
    assert c[0].shape == (3, 3, 3, 4)
    assert f[0].shape == (2, 2, 2, 4)

    fine_solid[0, 0, 0] = 1
    coarse_vel, coarse_solid = gp.downsample_2x(fine_vel, fine_solid)
    c2, _ = gp.extract_patches(coarse_vel, coarse_solid, fine_vel, fine_solid)
    assert len(c2) < 8


def test_extract_patches_wants_odd_neighbourhood():
    z = np.zeros((2, 2, 2, 3), dtype=np.float32)
    s = np.zeros((2, 2, 2), dtype=np.int32)
    with pytest.raises(AssertionError, match="odd"):
        gp.extract_patches(z, s, z, s, neighborhood=2)


def test_parse_grid_dims():
    assert gp.parse_grid_dims("0 128 0 96 0 64") == (128, 96, 64)
    with pytest.raises(ValueError, match="WholeExtent"):
        gp.parse_grid_dims("0 1 2")


def write_vti(path, nx, ny, nz, *, with_rho):
    total = nx * ny * nz
    rng = np.random.default_rng(1)
    vel = rng.standard_normal((nz, ny, nx, 3)).astype("<f4")
    rho = rng.random((nz, ny, nx)).astype("<f4")
    solid = (rng.random((nz, ny, nx)) > 0.5).astype("<i4")
    header = f'<VTKFile><ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}">'
    if with_rho:
        header += '<DataArray Name="rho"/>'
    blob = struct.pack("<Q", total * 12) + vel.tobytes()
    if with_rho:
        blob += struct.pack("<Q", total * 4) + rho.tobytes()
    blob += struct.pack("<Q", total * 4) + solid.tobytes()
    path.write_bytes(header.encode() + b'<AppendedData encoding="raw">\n_' + blob)
    return vel, rho, solid


@pytest.mark.parametrize("with_rho", [True, False])
def test_parse_vti(tmp_path, with_rho):
    path = tmp_path / "f.vti"
    vel, rho, solid = write_vti(path, 4, 3, 2, with_rho=with_rho)
    pv, prho, ps, dims = gp.parse_vti(path)
    assert dims == (4, 3, 2)
    assert np.array_equal(pv, vel)
    assert np.array_equal(ps, solid)
    if with_rho:
        assert np.array_equal(prho, rho)
    else:
        assert prho is None


def test_parse_vti_rejects_garbage(tmp_path):
    path = tmp_path / "f.vti"
    path.write_bytes(b"<VTKFile/>")
    with pytest.raises(ValueError, match="marker"):
        gp.parse_vti(path)
    write_vti(path, 2, 2, 2, with_rho=False)
    raw = bytearray(path.read_bytes())
    marker = raw.find(b"_") + 1
    raw[marker : marker + 8] = struct.pack("<Q", 5)
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="mismatch"):
        gp.parse_vti(path)


def test_pair_configs_are_a_cross_product():
    cfg = {
        "models": ["a", "b"],
        "wind_speeds": [1.0, 2.0],
        "reynolds_numbers": [0, 100, 200],
        "coarse_grid": "8x8x8",
        "fine_grid": "16x16x16",
        "duration": 5,
        "vtk_interval": 10,
    }
    pairs = gp.generate_pair_configs(cfg)
    assert len(pairs) == 12
    assert len({p.pair_id for p in pairs}) == 12
