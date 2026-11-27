import argparse
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import evaluate as ev  # noqa: E402
import gen_shapes as gs  # noqa: E402
import optimize as opt  # noqa: E402
import train_shapeopt as ts  # noqa: E402


def test_optimize_model_matches_trainer_architecture():
    a = ts.ShapeOptSurrogate(110)
    b = opt.ShapeOptSurrogate(110)
    assert {k: v.shape for k, v in a.state_dict().items()} == {
        k: v.shape for k, v in b.state_dict().items()
    }
    b.load_state_dict(a.state_dict())
    x = torch.randn(5, 110)
    assert torch.allclose(a(x), b(x))


def test_output_shape():
    assert ts.ShapeOptSurrogate(110)(torch.randn(4, 110)).shape == (4, 2)
    assert ts.SwiGLU(8, 16)(torch.randn(3, 8)).shape == (3, 8)


def write_results_csv(path, rows):
    import csv
    import json

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["displacements", "cd_value", "cl_value"])
        w.writeheader()
        for disp, cd, cl in rows:
            w.writerow({"displacements": json.dumps(disp), "cd_value": cd, "cl_value": cl})


def test_sobin_round_trip(tmp_path):
    rows = [([0.1, 0.2, 0.3], "0.5", "0.01"), ([0.4, 0.5, 0.6], "0.7", "-0.02")]
    csv_path = tmp_path / "r.csv"
    write_results_csv(csv_path, rows)
    out = tmp_path / "d.sobin"
    assert gs.export_sobin(csv_path, out) == 2
    disp, outputs = ts.load_sobin(str(out))
    assert disp.shape == (2, 3)
    assert np.allclose(disp, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], atol=1e-6)
    assert np.allclose(outputs, [[0.5, 0.01], [0.7, -0.02]], atol=1e-6)


def test_sobin_skips_rows_with_wrong_param_count(tmp_path):
    rows = [([0.1, 0.2, 0.3], "0.5", "0.01"), ([0.4, 0.5], "0.7", "0.02")]
    csv_path = tmp_path / "r.csv"
    write_results_csv(csv_path, rows)
    out = tmp_path / "d.sobin"
    assert gs.export_sobin(csv_path, out) == 1
    assert out.stat().st_size == 20 + (3 + 2) * 4
    disp, _ = ts.load_sobin(str(out))
    assert disp.shape == (1, 3)


def test_sobin_skips_rows_without_results(tmp_path):
    csv_path = tmp_path / "r.csv"
    write_results_csv(csv_path, [([0.1], "", ""), ([0.2], "1.0", "0.0")])
    assert gs.export_sobin(csv_path, tmp_path / "d.sobin") == 1


def test_sobin_missing_file(tmp_path):
    assert gs.export_sobin(tmp_path / "none.csv", tmp_path / "d.sobin") == 0


def test_load_sobin_rejects_bad_header(tmp_path):
    path = tmp_path / "d.sobin"
    path.write_bytes(struct.pack("<IIIII", 0, 1, 0, 3, 2))
    with pytest.raises(ValueError, match="magic"):
        ts.load_sobin(str(path))
    path.write_bytes(struct.pack("<IIIII", ts.SOBIN_MAGIC, 1, 0, 3, 3))
    with pytest.raises(ValueError, match="2 outputs"):
        ts.load_sobin(str(path))


def test_sample_id_is_stable_and_distinct():
    assert gs.sample_id("ahmed25", 3) == gs.sample_id("ahmed25", 3)
    assert gs.sample_id("ahmed25", 3) != gs.sample_id("ahmed25", 4)
    assert len(gs.sample_id("car", 0)) == 12


def test_obj_to_base64_decodes_to_one_indexed_obj():
    import base64

    text = base64.b64decode(
        gs.obj_to_base64(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]))
    ).decode()
    assert "f 1 2 3" in text
    assert text.count("\nv ") == 3


def test_ltws_export_reads_back(tmp_path):
    model = ts.ShapeOptSurrogate(12)
    path = tmp_path / "m.bin"
    ts.export_ltws(model.state_dict(), str(path))
    params = ev.load_weights(str(path))
    state = model.state_dict()
    assert len(params) == len(state)
    # export sorts by name
    for arr, name in zip(params, sorted(state), strict=True):
        assert np.allclose(arr, state[name].numpy(), atol=1e-7)


def test_normalizer_file_layout(tmp_path):
    path = tmp_path / "n.bin"
    ts.save_normalizer(np.array([1.0, 2.0]), np.array([3.0, 4.0]), str(path))
    assert np.frombuffer(path.read_bytes(), dtype=np.float32).tolist() == [1, 2, 3, 4]


def test_checkpoint_with_numpy_arrays_loads_weights_only(tmp_path):
    path = tmp_path / "ck.pt"
    torch.save(
        {
            "model_state_dict": ts.ShapeOptSurrogate(6).state_dict(),
            "input_dim": 6,
            "feat_mean": np.zeros(6, dtype=np.float32),
            "feat_std": np.ones(6, dtype=np.float64),
            "best_val_cd_mae": 0.1,
            "epoch": 3,
        },
        path,
    )
    ck = ts.load_checkpoint(path)
    assert ck["input_dim"] == 6
    assert ck["feat_std"].dtype == np.float64


def test_checkpoint_loader_refuses_arbitrary_objects(tmp_path):
    path = tmp_path / "evil.pt"
    torch.save({"x": argparse.Namespace(a=1)}, path)
    with pytest.raises(Exception, match="Weights only"):
        ts.load_checkpoint(path)


def test_training_smoke(tmp_path, rng):
    n, p = 40, 6
    disp = rng.standard_normal((n, p)).astype(np.float32)
    outputs = np.column_stack([disp.sum(axis=1) * 0.1 + 1.0, disp[:, 0] * 0.05]).astype(np.float32)
    data = tmp_path / "d.sobin"
    with data.open("wb") as f:
        f.write(struct.pack("<IIIII", ts.SOBIN_MAGIC, 1, n, p, 2))
        for d, o in zip(disp, outputs, strict=True):
            f.write(d.tobytes() + o.tobytes())
    args = argparse.Namespace(
        data=str(data), epochs=2, batch_size=16, lr=1e-3, patience=5, wind_speed=1.0, reynolds=1e3
    )
    ts.train(args)
    assert (tmp_path / "shapeopt_model.bin").exists()
    assert (tmp_path / "shapeopt_normalizer.bin").stat().st_size == (p + 2) * 2 * 4
