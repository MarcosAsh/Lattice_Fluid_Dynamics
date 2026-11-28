import argparse
import json
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import ffd  # noqa: E402
import optimize as opt  # noqa: E402
import study  # noqa: E402
import train_shapeopt as ts  # noqa: E402
from test_verify import CannedBackend  # noqa: E402


class Linear(torch.nn.Module):
    """Cd = w . x, Cl = 0"""

    def __init__(self, w):
        super().__init__()
        self.w = torch.tensor(w, dtype=torch.float32)

    def forward(self, x):
        return torch.stack([x @ self.w, torch.zeros(len(x))], dim=1)


def member(w):
    n = len(w)
    return Linear(w), np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)


@pytest.fixture
def lattice():
    return ffd.FFDLattice([0, 0, 0], [1, 1, 1], n_control=(2, 2, 2))


def weights(n, **coef):
    w = np.zeros(n + 2)
    for k, v in coef.items():
        w[int(k[1:])] = v
    return w


@pytest.mark.parametrize("grid", [(4, 3, 3), (3, 2, 2), (2, 2, 2)])
def test_free_mask_counts_the_free_parameters(grid):
    lat = ffd.FFDLattice([0, 0, 0], [1, 1, 1], n_control=grid)
    assert opt.free_mask(lat).sum() == lat.get_num_free_params()
    assert opt.free_mask(lat, symmetric=False).all()


def test_bounds_come_from_training_inside_the_box(lattice):
    n = lattice.get_num_params()
    train = np.vstack([np.full(n, -0.3), np.full(n, 0.1)])
    lo, hi = opt.parameter_bounds(lattice, train, 0.15)
    assert np.allclose(lo, -0.15) and np.allclose(hi, 0.1)
    lo, hi = opt.parameter_bounds(lattice, None, 0.2)
    assert np.allclose(hi, 0.2) and np.allclose(lo, -0.2)
    with pytest.raises(ValueError, match="nothing bounds"):
        opt.parameter_bounds(lattice, None, None)


def test_projection_clips_and_stays_symmetric():
    lat = ffd.FFDLattice([0, 0, 0], [1, 1, 1], n_control=(3, 2, 2))
    rng = np.random.default_rng(0)
    train = lat.sample_random(20, 0.1, True, rng)
    lo, hi = opt.parameter_bounds(lat, train, 0.15)
    d = opt.project(rng.normal(0, 1, lat.get_num_params()), lo, hi, lat)
    assert np.all(d >= lo - 1e-12) and np.all(d <= hi + 1e-12)
    assert np.allclose(d, lat.enforce_symmetry(d))


def test_optimum_is_clipped_to_the_training_range(lattice):
    n = lattice.get_num_params()
    lo, hi = np.full(n, -0.1), np.full(n, 0.2)
    sur = opt.Surrogate([member(weights(n, w0=-1.0))])
    d, traj = opt.optimise(
        sur, lattice, lo, hi, np.zeros(2), 100, 0.05, symmetric=False, log=lambda _: None
    )
    assert d[0] == pytest.approx(0.2)
    assert np.allclose(d[1:], 0.0)
    assert traj[-1][2] == pytest.approx(-0.2, abs=1e-6)
    rep = opt.bound_report(d, lo, hi, opt.free_mask(lattice, symmetric=False))
    assert rep["at_bound"] == 1
    assert rep["indices"].tolist() == [0]


def test_ensemble_spread_holds_the_optimum_back(lattice):
    n = lattice.get_num_params()
    lo, hi = np.full(n, -0.2), np.full(n, 0.2)
    # members agree on x1, disagree on x0
    sur = opt.Surrogate([member(weights(n, w0=-1.0, w1=-1.0)), member(weights(n, w0=0.5, w1=-1.0))])
    x = torch.zeros(1, n + 2)
    x[0, 0] = 0.1
    mean, std, _ = sur.predict(x)
    assert mean.item() == pytest.approx(-0.025)
    assert std.item() == pytest.approx(0.15 / np.sqrt(2))
    quiet = {"symmetric": False, "log": lambda _: None}
    d_mean, _ = opt.optimise(sur, lattice, lo, hi, np.zeros(2), 300, 0.005, 0.0, **quiet)
    d_lcb, _ = opt.optimise(sur, lattice, lo, hi, np.zeros(2), 300, 0.005, 1.0, **quiet)
    assert d_mean[0] == pytest.approx(0.2)
    assert abs(d_lcb[0]) < 0.02
    assert d_mean[1] == pytest.approx(0.2) and d_lcb[1] == pytest.approx(0.2)


def test_distance_report_by_hand():
    train = np.array([[0.0, 0.0], [3.0, 4.0]])
    r = opt.distance_report(np.array([0.0, 1.0]), train, np.ones(2))
    assert r["nearest_index"] == 0
    assert r["distance"] == pytest.approx(1.0)
    assert r["median_train_nn"] == pytest.approx(5.0)
    assert r["ratio"] == pytest.approx(0.2)
    assert r["baseline_distance"] == pytest.approx(0.0)
    # distances are in the surrogate's normalised inputs
    r = opt.distance_report(np.array([0.0, 1.0]), train, np.array([1.0, 2.0]))
    assert r["distance"] == pytest.approx(0.5)
    assert r["median_train_nn"] == pytest.approx(np.hypot(3, 2))


def test_calibration_counts_coverage():
    sur = opt.Surrogate([member(np.array([1.0, 0, 0, 0, 0])), member(np.array([3.0, 0, 0, 0, 0]))])
    x = np.array([[1.0, 0, 0], [2.0, 0, 0]])
    # means 2 and 4, std sqrt(2) and 2 sqrt(2)
    y = np.array([[2.5, 0.0], [10.0, 0.0]])
    cal = opt.calibration(sur, np.zeros(2), x, y)
    assert cal["samples"] == 2
    assert cal["mae"] == pytest.approx((0.5 + 6.0) / 2, rel=1e-5)
    assert cal["coverage_2std"] == pytest.approx(0.5)


# Refusal without a solver check


PRED = {"cd": 3.7, "baseline_cd": 4.0}


def test_summary_refuses_without_verification():
    with pytest.raises(opt.Unverified, match="--no-verify"):
        opt.summary(PRED, None, no_verify=False)


def test_summary_labels_an_unverified_result():
    (line,) = opt.summary(PRED, None, no_verify=True)
    assert line.startswith("surrogate prediction, not verified: ")
    assert "-7.50%" in line


def test_summary_with_a_table():
    row = {
        "status": "ok",
        "grid": "64x32x32",
        "cd": 3.9,
        "delta": -0.1,
        "delta_ci95": 0.05,
        "delta_pct": -2.5,
        "verdict": "verified",
        "cda_delta": 0.02,
        "cda_delta_ci95": 0.05,
        "cda_verdict": "not distinguishable",
    }
    result = {"shapes": [{"rows": [row], "overall": "verified"}]}
    lines = opt.summary(PRED, result, no_verify=False)
    assert lines[0].startswith("surrogate prediction: ")
    assert "Cd 4.0000 -> 3.9000 (-0.1000 ± 0.0500, -2.50%), verified" in lines[1]
    assert lines[-1] == "verdict: verified"


# Command line, with a tiny checkpoint on the Ahmed mesh


@pytest.fixture
def checkpoint(tmp_path, rng):
    p, n = 24, 30
    disp = rng.uniform(-0.05, 0.05, (n, p)).astype(np.float32)
    out = np.column_stack([4.0 + disp.sum(axis=1), 0.1 + 0 * disp[:, 0]]).astype(np.float32)
    data = tmp_path / "d.sobin"
    with data.open("wb") as f:
        f.write(struct.pack("<IIIII", ts.SOBIN_MAGIC, 1, n, p, 2))
        for d, o in zip(disp, out, strict=True):
            f.write(d.tobytes() + o.tobytes())
    ts.train(
        argparse.Namespace(
            data=str(data),
            epochs=2,
            batch_size=16,
            lr=1e-3,
            patience=5,
            wind_speed=1.0,
            reynolds=50.0,
            seed=0,
        )
    )
    ck = ts.load_checkpoint(tmp_path / "shapeopt_model_best.pt")
    assert ck["train_x"].shape == (24, p) and ck["val_x"].shape == (6, p)
    return tmp_path / "shapeopt_model_best.pt"


def cli(checkpoint, tmp_path, *extra):
    return opt.main(
        [
            "--weights",
            str(checkpoint),
            "--base-model",
            "ahmed25",
            "--output",
            str(tmp_path / "opt.obj"),
            "--steps",
            "3",
            "--ffd-grid",
            "2",
            "2",
            "2",
            *extra,
        ]
    )


def test_cli_refuses_without_a_solver(checkpoint, tmp_path, capsys):
    assert cli(checkpoint, tmp_path, "--binary", str(tmp_path / "missing")) == 2
    cap = capsys.readouterr()
    assert "no solver binary" in cap.err
    assert " -> " not in cap.out
    cand = json.loads((tmp_path / "opt.json").read_text())
    assert cand["kind"] == "shapeopt-candidate"
    assert cand["guard"]["distance"]["samples"] == 24


def test_cli_no_verify_is_labelled(checkpoint, tmp_path, capsys):
    assert cli(checkpoint, tmp_path, "--no-verify") == 0
    assert "surrogate prediction, not verified: Cd" in capsys.readouterr().out


def test_cli_verifies_by_default(checkpoint, tmp_path, capsys, monkeypatch):
    cds = {
        ("ahmed_25deg_m", "16x8x8"): (4.0, 0.03, 40.0),
        ("opt", "16x8x8"): (4.0, 0.03, 40.0),
    }
    monkeypatch.setattr(study, "LocalBackend", lambda: CannedBackend(cds))
    solver = tmp_path / "solver"
    solver.write_text("")
    args = ["--binary", str(solver), "--grid", "16x8x8", "--reynolds", "50", "--duration", "5"]
    assert cli(checkpoint, tmp_path, *args) == 0
    out = capsys.readouterr().out
    assert "solver on 16x8x8: Cd 4.0000 -> 4.0000" in out
    assert "verdict: not distinguishable" in out
    assert "surrogate trained" not in out
