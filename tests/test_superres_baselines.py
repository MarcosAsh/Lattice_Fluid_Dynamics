import json
import struct

import numpy as np
import pytest

import evaluate_baselines as eb

# coarse cell i sits at fine coordinate 2i + 0.5


def coarse_of(f, shape):
    nz, ny, nx = shape
    z, y, x = np.meshgrid(*(2 * np.arange(n) + 0.5 for n in (nz, ny, nx)), indexing="ij")
    return f(x, y, z)


def fine_of(f, shape):
    nz, ny, nx = (2 * n for n in shape)
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    return f(x.astype(float), y.astype(float), z.astype(float))


def inner(a, margin):
    return a[margin:-margin, margin:-margin, margin:-margin]


SHAPE = (8, 9, 10)


def linear(x, y, z):
    return 0.3 + 0.02 * x - 0.05 * y + 0.01 * z


def quadratic(x, y, z):
    return 0.1 * x**2 - 0.03 * x * y + 0.02 * z**2


def cubic(x, y, z):
    return 0.001 * x**3 - 0.002 * y**3 + 0.0005 * x * y * z + 0.01 * z**2


@pytest.mark.parametrize("name", ["trilinear", "tricubic", "catmull_rom"])
def test_interpolants_reproduce_a_linear_field(name):
    out = eb.upsample(coarse_of(linear, SHAPE), eb.INTERPOLANTS[name])
    assert np.allclose(inner(out, 4), inner(fine_of(linear, SHAPE), 4), atol=1e-12)


def test_tricubic_reproduces_a_cubic_field():
    out = eb.upsample(coarse_of(cubic, SHAPE), eb.CUBIC)
    assert np.allclose(inner(out, 4), inner(fine_of(cubic, SHAPE), 4), atol=1e-10)


def test_catmull_rom_reproduces_quadratics_but_not_cubics():
    out = eb.upsample(coarse_of(quadratic, SHAPE), eb.CATMULL_ROM)
    assert np.allclose(inner(out, 4), inner(fine_of(quadratic, SHAPE), 4), atol=1e-10)
    out = eb.upsample(coarse_of(cubic, SHAPE), eb.CATMULL_ROM)
    assert np.abs(inner(out, 4) - inner(fine_of(cubic, SHAPE), 4)).max() > 1e-4


def test_trilinear_does_not_reproduce_a_quadratic_field():
    out = eb.upsample(coarse_of(quadratic, SHAPE), eb.LINEAR)
    assert np.abs(inner(out, 4) - inner(fine_of(quadratic, SHAPE), 4)).max() > 1e-3


def test_nearest_copies_the_parent_cell():
    c = np.arange(24.0).reshape(2, 3, 4)
    out = eb.upsample(c, eb.NEAREST)
    assert out.shape == (4, 6, 8)
    assert np.array_equal(out, c.repeat(2, 0).repeat(2, 1).repeat(2, 2))


def test_taps_sum_to_one():
    for taps in [*eb.INTERPOLANTS.values(), eb.SHADER_SKIP, eb.TRAIN_SKIP]:
        for t in taps:
            assert sum(t.values()) == pytest.approx(1.0)


def zero_net(hidden=4):
    shapes = [(hidden,), (hidden, 108), (hidden,), (hidden, hidden)]
    shapes += [(hidden,), (hidden, hidden), (32,), (32, hidden)]
    return [np.zeros(s) for s in shapes], np.zeros(4), np.ones(4)


def skip_only(skip):
    params, mean, std = zero_net()
    c4 = np.repeat(coarse_of(linear, SHAPE)[..., None], 4, axis=-1)
    return eb.net_upscale(c4, params, mean, std, skip)[..., 0]


def test_shader_skip_samples_one_fine_cell_downstream():
    # children at +1/4 and +3/4 of a coarse cell, so fine j gets f(j + 1)
    out = skip_only(eb.SHADER_SKIP)
    shifted = fine_of(lambda x, y, z: linear(x + 1, y + 1, z + 1), SHAPE)
    assert np.allclose(inner(out, 4), inner(shifted, 4), atol=1e-12)


def test_training_skip_samples_three_quarters_out():
    # children at -3/4 and +3/4, so even j gets f(j - 1) and odd j f(j + 1)
    out = skip_only(eb.TRAIN_SKIP)

    def moved(v):
        return v + np.where(v % 2 == 0, -1.0, 1.0)

    expect = fine_of(lambda x, y, z: linear(moved(x), moved(y), moved(z)), SHAPE)
    assert np.allclose(inner(out, 4), inner(expect, 4), atol=1e-12)


def test_net_port_matches_the_torch_model():
    torch = pytest.importorskip("torch")
    import train_superres as ts

    torch.manual_seed(0)
    model = ts.SuperResNet(hidden=128).double().eval()
    params = [v.detach().numpy() for _, v in sorted(model.state_dict().items())]
    rng = np.random.default_rng(1)
    c4 = rng.normal(size=(4, 5, 6, 4))
    mean, std = np.array([0.03, 0.0, 0.0, 1.0]), np.array([0.02, 0.01, 0.01, 1.0])
    out = eb.net_upscale(c4, params, mean, std, eb.TRAIN_SKIP)
    z, y, x = 2, 2, 3
    patch = (c4[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] - mean) / std
    with torch.no_grad():
        ref = model(torch.from_numpy(patch.reshape(1, 108))).numpy()
    ref = ref.reshape(2, 2, 2, 4) * std + mean
    assert np.allclose(out[2 * z : 2 * z + 2, 2 * y : 2 * y + 2, 2 * x : 2 * x + 2], ref)


def grid(n=24):
    h = 2 * np.pi / n
    z, y, x = np.meshgrid(*(np.arange(n) * h,) * 3, indexing="ij")
    return x, y, z, h


def test_divergence_of_a_solenoidal_field_is_zero():
    x, y, _, _ = grid()
    u = np.stack([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y), np.zeros_like(x)], axis=-1)
    assert np.abs(inner(eb.divergence(u), 1)).max() < 1e-12


def test_divergence_of_a_compressing_field():
    n = 10
    z, y, x = np.meshgrid(*(np.arange(n, dtype=float),) * 3, indexing="ij")
    u = np.stack([2 * x, -y, 3 * z], axis=-1)
    assert np.allclose(eb.divergence(u), 4.0)


def test_vorticity_of_solid_body_rotation():
    n = 10
    _, y, x = np.meshgrid(*(np.arange(n, dtype=float),) * 3, indexing="ij")
    u = np.stack([-y, x, np.zeros_like(x)], axis=-1)
    w = eb.vorticity(u)
    assert np.allclose(w[..., :2], 0.0)
    assert np.allclose(w[..., 2], 2.0)


def test_block_sums_and_relative_l2_by_hand():
    e2 = np.array([1.0, 4.0, 9.0, 16.0]).reshape(1, 1, 4)
    r2 = np.array([4.0, 4.0, 4.0, 4.0]).reshape(1, 1, 4)
    mask = np.array([True, True, True, False]).reshape(1, 1, 4)
    ids = eb.block_ids((1, 1, 4), 2)
    s = eb.block_sums([e2, r2], mask, ids)
    assert s.tolist() == [[5.0, 8.0], [9.0, 4.0]]
    assert eb.rel_l2(s.sum(0)) == pytest.approx(np.sqrt(14 / 12))
    assert eb.ratio_of_sums(s.sum(0)) == pytest.approx(14 / 12)


def test_bootstrap_point_and_interval():
    s = np.array([[1.0, 1.0], [3.0, 1.0]])
    rng = np.random.default_rng(0)
    counts = [eb.resample_counts(2, 4000, rng)]
    p, lo, hi = eb.bootstrap([s], eb.ratio_of_sums, counts)
    assert p == pytest.approx(2.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(3.0)
    one = [np.array([[2.0, 4.0]])]
    assert eb.bootstrap(one, eb.ratio_of_sums, [eb.resample_counts(1, 10, rng)]) == (0.5,) * 3


def test_regions():
    solid = np.zeros((16, 16, 32), dtype=np.int32)
    solid[6:10, 6:10, 6:10] = 1
    m = eb.regions(solid, near=2)
    assert not (m["all"] & m["solid"]).any()
    assert not m["all"][:, :, 28:].any()
    assert m["near"][8, 8, 11] and not m["near"][8, 8, 12]
    assert m["wake"][8, 8, 10] and m["wake"][8, 8, 17] and not m["wake"][8, 8, 18]
    assert not m["wake"][8, 8, 5]
    assert not m["interior"][8, 8, 10] and m["interior"][8, 8, 11]


def test_train_like_matches_gen_pairs_rule():
    coarse_solid = np.zeros((6, 6, 6), dtype=np.int32)
    coarse_solid[3, 3, 3] = 1
    ok = eb.train_like(coarse_solid)
    assert ok.shape == (12, 12, 12)
    assert not ok[0].any()
    assert not ok[4, 4, 4] and ok[2, 2, 2]


def write_vti(path, arrays, nx, ny, nz):
    names = list(arrays)
    head, off = [], 0
    for n in names:
        a = arrays[n]
        comps = a.shape[-1] if a.ndim == 4 else 1
        kind = "Int32" if a.dtype == np.int32 else "Float32"
        head.append(
            f'<DataArray type="{kind}" Name="{n}" NumberOfComponents="{comps}" '
            f'format="appended" offset="{off}"/>'
        )
        off += 8 + a.nbytes
    xml = (
        f'<VTKFile><ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}"><Piece><PointData>'
        + "".join(head)
        + '</PointData></Piece></ImageData>\n  <AppendedData encoding="raw">\n_'
    )
    blob = b"".join(struct.pack("<Q", arrays[n].nbytes) + arrays[n].tobytes() for n in names)
    path.write_bytes(xml.encode() + blob)


def fields(rng, nz, ny, nx, body):
    vel = (0.05 + 0.005 * rng.standard_normal((nz, ny, nx, 3))).astype(np.float32)
    solid = np.zeros((nz, ny, nx), dtype=np.int32)
    solid[body] = 1
    vel[solid > 0] = 0
    rho = np.ones((nz, ny, nx), dtype=np.float32)
    return {"velocity": vel, "rho": rho, "solid": solid, "mean_velocity": vel.copy()}


def test_read_vti_round_trip(tmp_path, rng):
    arrays = fields(rng, 4, 5, 6, (slice(1, 2), slice(1, 2), slice(1, 2)))
    write_vti(tmp_path / "f.vti", arrays, 6, 5, 4)
    back = eb.read_vti(tmp_path / "f.vti")
    for k, v in arrays.items():
        assert np.array_equal(back[k], v)


def test_end_to_end_on_tiny_fields(tmp_path, rng):
    coarse = fields(rng, 8, 8, 16, (slice(3, 5), slice(3, 5), slice(4, 6)))
    fine = fields(rng, 16, 16, 32, (slice(6, 10), slice(6, 10), slice(8, 12)))
    write_vti(tmp_path / "c.vti", coarse, 16, 8, 8)
    write_vti(tmp_path / "f.vti", fine, 32, 16, 16)
    params, _, _ = zero_net(hidden=128)
    with (tmp_path / "w.bin").open("wb") as f:
        f.write(struct.pack("<III", eb.LTWS_MAGIC, 1, len(params)))
        for p in params:
            f.write(struct.pack(f"<I{p.ndim}I", p.ndim, *p.shape))
            f.write(p.astype("<f4").tobytes())
    norm = tmp_path / "n.bin"
    np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype="<f4").tofile(norm)
    out = tmp_path / "out"
    args = ["--case", "t", str(tmp_path / "c.vti"), str(tmp_path / "f.vti")]
    args += ["--weights", str(tmp_path / "w.bin"), "--norm", str(norm), "--boot", "50"]
    assert eb.main([*args, "--out", str(out)]) == 0
    res = json.loads((out / "results.json").read_text())
    row = res["real_mean"]["cases"]["t"]
    assert set(row) == {*eb.METHODS, "reference"}
    # the training skip alone is not trilinear, so a zero net does worse
    assert row["net_train_skip"]["vel_all"][0] > row["trilinear"]["vel_all"][0]
    assert (out / "tables.md").exists()
