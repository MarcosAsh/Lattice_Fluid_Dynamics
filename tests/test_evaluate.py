import struct

import numpy as np
import pytest

import evaluate as ev


def test_metrics_perfect_prediction():
    x = np.array([1.0, 2.0, 3.0])
    m = ev.compute_metrics(x, x)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0
    assert m["max_err"] == 0.0


def test_metrics_known_values():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    pred = true + np.array([0.5, -0.5, 0.5, -0.5])
    m = ev.compute_metrics(true, pred)
    assert m["mae"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(0.5)
    assert m["max_err"] == pytest.approx(0.5)
    # ss_res = 1.0, ss_tot = 5.0
    assert m["r2"] == pytest.approx(0.8)
    assert m["mean_true"] == pytest.approx(2.5)


def test_metrics_constant_target_gives_nan_r2():
    m = ev.compute_metrics(np.ones(4), np.zeros(4))
    assert np.isnan(m["r2"])


def test_swish_matches_definition_and_survives_extremes():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    out = ev.swish(x)
    assert np.all(np.isfinite(out))
    assert out[2] == 0.0
    assert out[3] == pytest.approx(1.0 / (1.0 + np.exp(-1.0)))
    assert out[4] == pytest.approx(1000.0)


def write_ltws(path, tensors, magic=0x4C545753, version=1):
    with path.open("wb") as f:
        f.write(struct.pack("<III", magic, version, len(tensors)))
        for t in tensors:
            f.write(struct.pack("<I", t.ndim))
            f.write(struct.pack(f"<{t.ndim}I", *t.shape))
            f.write(t.astype("<f4").tobytes())


def test_ltws_round_trip(tmp_path, rng):
    tensors = [rng.standard_normal((4, 3)), rng.standard_normal(4), rng.standard_normal((2, 2))]
    path = tmp_path / "m.bin"
    write_ltws(path, tensors)
    out = ev.load_weights(str(path))
    assert [o.shape for o in out] == [(4, 3), (4,), (2, 2)]
    for a, b in zip(out, tensors, strict=True):
        assert np.allclose(a, b, atol=1e-6)


def test_ltws_rejects_bad_header(tmp_path):
    path = tmp_path / "m.bin"
    write_ltws(path, [], magic=1)
    with pytest.raises(ValueError, match="magic"):
        ev.load_weights(str(path))
    write_ltws(path, [], version=2)
    with pytest.raises(ValueError, match="version"):
        ev.load_weights(str(path))


def test_normalizer_round_trip(tmp_path):
    path = tmp_path / "n.bin"
    path.write_bytes(np.array([1, 2, 3], "<f4").tobytes() + np.array([4, 5, 6], "<f4").tobytes())
    mean, std = ev.load_normalizer(str(path))
    assert mean.tolist() == [1, 2, 3]
    assert std.tolist() == [4, 5, 6]


def write_dataset(path, rows, magic=0x4C415454, features=5):
    rows = np.asarray(rows, dtype="<f4")
    with path.open("wb") as f:
        f.write(struct.pack("<IIII", magic, 1, len(rows), features))
        f.write(rows.tobytes())


def test_dataset_loader(tmp_path):
    rows = [[1.0, 144, 1, 3.3, 0.1], [2.0, 288, 2, 2.9, -0.2]]
    path = tmp_path / "d.bin"
    write_dataset(path, rows)
    data = ev.load_dataset(str(path))
    assert data.shape == (2, 5)
    assert np.allclose(data, rows)


def test_dataset_loader_rejects_bad_files(tmp_path):
    path = tmp_path / "d.bin"
    write_dataset(path, [[0] * 5], magic=7)
    with pytest.raises(ValueError, match="magic"):
        ev.load_dataset(str(path))
    write_dataset(path, [[0] * 4], features=4)
    with pytest.raises(ValueError, match="5 features"):
        ev.load_dataset(str(path))


def mlp_params(rng, scale=0.1):
    shapes = [
        (256, 3), (256, 1),
        (512, 256), (512, 256), (256, 512),
        (128, 256), (128, 1),
        (256, 128), (256, 128), (128, 256),
        (2, 128), (2, 1),
    ]  # fmt: skip
    return [(rng.standard_normal(s) * scale).astype(np.float32) for s in shapes]


def test_forward_shape_and_batching(rng):
    params = mlp_params(rng)
    x = rng.standard_normal((3, 7)).astype(np.float32)
    out = ev.forward(x, params)
    assert out.shape == (2, 7)
    single = ev.forward(x[:, :1], params)
    assert np.allclose(single[:, 0], out[:, 0], atol=1e-5)


def test_forward_with_zero_weights_returns_output_bias(rng):
    params = [np.zeros_like(p) for p in mlp_params(rng)]
    params[11] = np.array([[1.5], [-2.0]], dtype=np.float32)
    out = ev.forward(np.ones((3, 4), dtype=np.float32), params)
    assert np.allclose(out, np.array([[1.5], [-2.0]]) * np.ones((1, 4)))
