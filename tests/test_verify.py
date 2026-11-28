import json
import math

import numpy as np
import pytest
import yaml

import ffd
import mesh_checks
import protocol
import study
import verify
from meshes import cube

BASE = {"cd": 4.0, "ci95": 0.03, "area": 40.0}


def shape(cd, ci95=0.04, area=38.0, pred=3.7, pred_base=4.1, pred_area_ratio=0.96):
    return {
        "cd": cd,
        "ci95": ci95,
        "area": area,
        "pred": pred,
        "pred_base": pred_base,
        "pred_area_ratio": pred_area_ratio,
    }


# Verdicts and interval arithmetic


@pytest.mark.parametrize(
    ("delta", "half", "predicted", "expected"),
    [
        (-0.2, 0.05, -0.4, "verified"),
        (0.2, 0.05, 0.1, "verified"),
        (-0.04, 0.05, -0.4, "not distinguishable"),
        (0.05, 0.05, -0.4, "not distinguishable"),
        (0.2, 0.05, -0.4, "wrong direction"),
        (-0.2, 0.05, 0.3, "wrong direction"),
        (-0.2, 0.05, 0.0, "verified"),
        (-0.2, None, -0.4, "no interval"),
    ],
)
def test_verdict(delta, half, predicted, expected):
    assert verify.verdict(delta, half, predicted) == expected


def test_combine_adds_half_widths_in_quadrature():
    assert verify.combine(0.03, 0.04) == pytest.approx(0.05)
    assert verify.combine(0.03, None) is None
    assert verify.combine(0.03, math.nan) is None


def test_compare_by_hand():
    c = verify.compare(BASE, shape(3.8))
    # 3.8 - 4.0, half-width sqrt(0.04^2 + 0.03^2)
    assert c["delta"] == pytest.approx(-0.2)
    assert c["delta_ci95"] == pytest.approx(0.05)
    assert c["delta_pct"] == pytest.approx(-5.0)
    assert c["delta_pct_ci95"] == pytest.approx(1.25)
    assert c["surrogate_error"] == pytest.approx(-0.1)
    assert c["pred_delta"] == pytest.approx(-0.4)
    assert c["verdict"] == "verified"
    # CdA over the baseline area: 3.8 * 38 / 40
    assert c["area_ratio"] == pytest.approx(0.95)
    assert c["cda"] == pytest.approx(3.61)
    assert c["cda_ci95"] == pytest.approx(0.038)
    assert c["cda_delta"] == pytest.approx(-0.39)
    assert c["cda_delta_ci95"] == pytest.approx(math.hypot(0.038, 0.03))
    assert c["pred_cda_delta"] == pytest.approx(3.7 * 0.96 - 4.1)
    assert c["cda_verdict"] == "verified"


def test_lower_cd_from_a_larger_frontal_area_is_caught():
    c = verify.compare(BASE, shape(3.8, area=44.0, pred_area_ratio=1.1))
    assert c["verdict"] == "verified"
    # 3.8 * 1.1 = 4.18 against 4.0
    assert c["cda_delta"] == pytest.approx(0.18)
    assert c["cda_delta_ci95"] == pytest.approx(math.hypot(0.044, 0.03))
    assert c["pred_cda_delta"] == pytest.approx(3.7 * 1.1 - 4.1)
    assert c["cda_verdict"] == "wrong direction"


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        (["verified", "verified"], "verified"),
        (["verified", "not distinguishable"], "grid dependent"),
        (["wrong direction"], "wrong direction"),
        ([], "not run"),
    ],
)
def test_overall(verdicts, expected):
    assert verify.overall(verdicts) == expected


# Protocol


def test_reynolds_keeps_the_baseline_viscosity():
    base = np.array([[0.0, 0, 0], [2.0, 1, 0.5]])
    ref = protocol.reference(base, 100.0)
    assert ref.scale == pytest.approx(protocol.SOLVER_SPAN / 2.0)
    longer = base * [1.2, 1, 1]
    assert protocol.case_reynolds(longer, ref) == pytest.approx(120.0)
    params = protocol.case_params("s.obj", longer, ref, grid="64x32x32", duration=30)
    assert params["scale"] == ref.scale
    assert params["args"] == ["--strict-reynolds"]


def test_scale_run_keeps_flow_throughs():
    assert protocol.scale_run({"duration": 30}, "128x64x64", "64x32x32") == {"duration": 60}
    run = protocol.run_settings(converge=0.01, max_duration=25)
    assert protocol.scale_run(run, "96x48x48", "64x32x32")["max_duration"] == 38
    with pytest.raises(ValueError, match="max-duration"):
        protocol.run_settings(converge=0.01)
    with pytest.raises(ValueError, match="duration"):
        protocol.run_settings()


# Geometry


def box(lx=2.0):
    v, f = cube(0.5)
    return v * [lx, 1, 1], np.asarray(f)


def test_frontal_area_of_a_box():
    v, f = box()
    assert mesh_checks.frontal_area(v, f) == pytest.approx(1.0, rel=0.01)
    assert mesh_checks.frontal_area(v * [1, 2, 1], f) == pytest.approx(2.0, rel=0.01)


def test_check_reports_size_changes():
    v, f = box()
    lat = ffd.FFDLattice(*ffd.compute_bbox(v), n_control=(2, 2, 2))
    d = np.zeros((2, 2, 2, 3))
    d[:, 1, :, 1] = 0.1 * lat.extent[1]
    d[:, 0, :, 1] = -0.1 * lat.extent[1]
    d = d.ravel()
    r = mesh_checks.check(v, lat.deform(v, d), f, lat, d)
    assert r["ok"], r["problems"]
    assert r["min_jacobian"] > 0
    # vertices sit inside the 5% padding, so they move 0.1 each way
    assert r["frontal_area_ratio"] == pytest.approx(1.2, rel=0.01)
    assert r["volume_ratio"] == pytest.approx(1.2, rel=1e-6)
    assert r["length_ratio"] == pytest.approx(1.0)


def test_check_rejects_a_fold():
    v, f = box()
    lat = ffd.FFDLattice(*ffd.compute_bbox(v), n_control=(2, 2, 2))
    d = np.zeros((2, 2, 2, 3))
    # push the front face behind the back face
    d[0, :, :, 0] = 1.5 * lat.extent[0]
    d = d.ravel()
    r = mesh_checks.check(v, lat.deform(v, d), f, lat, d)
    assert not r["ok"]
    assert r["min_jacobian"] < 0
    assert r["flipped_faces"] > 0


def test_check_rejects_an_open_mesh():
    v, f = box()
    r = mesh_checks.check(v, v, f[:-1])
    assert not r["ok"]
    assert "not watertight" in r["problems"][0]


# Running through the study runner with canned manifests


class CannedBackend(study.Backend):
    """Writes a manifest per case, Cd picked by mesh name and grid"""

    name = "canned"

    def __init__(self, cds):
        self.cds = cds
        self.cases = []

    def run(self, case):
        self.cases.append(case)
        p = case.params
        name = p["model"].rsplit("/", 1)[-1].removesuffix(".obj")
        cd, ci, area = self.cds[(name, p["grid"])]
        grid = [int(n) for n in p["grid"].split("x")]
        case.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "git": {"commit": "abc", "dirty": False},
                    "gl": {"renderer": "test"},
                    "lattice": {"grid": grid, "tau": 0.53, "reynolds_effective": p["reynolds"]},
                    "model": {"ref_area": area},
                    "results": {"cd_mean": cd, "cl_mean": 0.1},
                    "convergence": {
                        "method": "test",
                        "enabled": False,
                        "stationary": True,
                        "cd_mean": cd,
                        "cd_sem": ci / 2,
                        "cd_ci95_half_width": ci,
                    },
                }
            )
        )
        return study.Outcome("ok", 0, 0.1)


GRID = "16x8x8"
FINE = "32x16x16"


def candidate(tmp_path, name="opt", pred=(4.1, 3.7)):
    """Baseline box, a slightly narrower FFD variant and its candidate file"""
    v, f = box()
    base = tmp_path / "base.obj"
    ffd.save_obj(base, v, f)
    lat = ffd.FFDLattice(*ffd.compute_bbox(v), n_control=(2, 2, 2))
    d = np.zeros((2, 2, 2, 3))
    d[:, 1, :, 1] = -0.02 * lat.extent[1]
    d[:, 0, :, 1] = 0.02 * lat.extent[1]
    obj = tmp_path / f"{name}.obj"
    ffd.save_obj(obj, lat.deform(v, d.ravel()), f)
    data = {
        "kind": verify.CANDIDATE_KIND,
        "name": name,
        "obj": str(obj),
        "base_obj": str(base),
        "ffd_grid": [2, 2, 2],
        "displacements": d.ravel().tolist(),
        "predicted": {"cd": pred[1], "baseline_cd": pred[0], "cl": 0.0},
        "protocol": {
            "grid": GRID,
            "run": {"duration": 10},
            "reference": {"reynolds": 50.0},
        },
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(data))
    return path


def run_verify(tmp_path, cds, grids=None, **over):
    args = verify.argparse.Namespace(
        grid=grids,
        reynolds=None,
        duration=None,
        converge=None,
        max_duration=None,
        converge_window=None,
        binary=tmp_path / "solver",
        workdir=tmp_path,
        timeout=None,
        force=False,
        **over,
    )
    backend = CannedBackend(cds)
    result = verify.verify_files([candidate(tmp_path)], args, tmp_path / "out", backend)
    return result, backend


@pytest.mark.parametrize(
    ("cd", "expected"),
    [(3.8, "verified"), (3.98, "not distinguishable"), (4.2, "wrong direction")],
)
def test_verify_end_to_end(tmp_path, cd, expected, capsys):
    result, backend = run_verify(
        tmp_path, {("base", GRID): (4.0, 0.03, 40.0), ("opt", GRID): (cd, 0.04, 39.0)}
    )
    row = result["shapes"][0]["rows"][0]
    assert row["verdict"] == expected
    assert result["shapes"][0]["overall"] == expected
    assert row["surrogate_error"] == pytest.approx(3.7 - cd)
    assert row["delta_ci95"] == pytest.approx(0.05)
    out = capsys.readouterr().out
    assert expected in out
    assert "± 0.0500" in out
    assert (tmp_path / "out" / "verify.md").read_text().startswith("| shape |")
    # same scale and strict Re for both, duration from the training protocol
    params = [c.params for c in backend.cases]
    assert {p["scale"] for p in params} == {protocol.SOLVER_SPAN / 2.0}
    assert all(p["args"] == ["--strict-reynolds"] and p["duration"] == 10 for p in params)
    assert params[0]["reynolds"] == pytest.approx(50.0)


def test_verify_two_grids(tmp_path):
    cds = {
        ("base", GRID): (4.0, 0.03, 40.0),
        ("opt", GRID): (3.8, 0.04, 39.0),
        ("base", FINE): (3.5, 0.03, 160.0),
        ("opt", FINE): (3.49, 0.04, 156.0),
    }
    result, backend = run_verify(tmp_path, cds, grids=[GRID, FINE])
    rows = result["shapes"][0]["rows"]
    # finest grid first, as gci orders them
    assert [r["grid"] for r in rows] == [FINE, GRID]
    assert [r["verdict"] for r in rows] == ["not distinguishable", "verified"]
    assert result["shapes"][0]["overall"] == "grid dependent"
    # the surrogate error only means something on its training grid
    assert rows[0]["surrogate_error"] is None
    assert rows[1]["surrogate_error"] == pytest.approx(-0.1)
    durations = {c.params["grid"]: c.params["duration"] for c in backend.cases}
    assert durations == {GRID: 10, FINE: 20}
    groups = {c.group for c in study.load_study(tmp_path / "out" / "study.yaml").cases}
    assert len(groups) == 2


def test_verify_skips_a_rejected_shape(tmp_path):
    path = candidate(tmp_path)
    data = json.loads(path.read_text())
    v, f = box()
    lat = ffd.FFDLattice(*ffd.compute_bbox(v), n_control=(2, 2, 2))
    d = np.zeros((2, 2, 2, 3))
    d[0, :, :, 0] = 1.5 * lat.extent[0]
    ffd.save_obj(data["obj"], lat.deform(v, d.ravel()), f)
    data["displacements"] = d.ravel().tolist()
    path.write_text(json.dumps(data))
    settings, _ = verify.settings_from(
        verify.argparse.Namespace(
            grid=None, reynolds=None, duration=None, converge=None, max_duration=None
        ),
        data["protocol"],
    )
    backend = CannedBackend({("base", GRID): (4.0, 0.03, 40.0)})
    result = verify.run(
        [verify.load_candidate(path)],
        settings,
        tmp_path / "solver",
        tmp_path / "o",
        tmp_path,
        backend=backend,
    )
    assert result["shapes"][0]["overall"] == "geometry rejected"
    assert len(backend.cases) == 1


def test_settings_need_a_grid_without_protocol():
    ns = verify.argparse.Namespace(
        grid=None, reynolds=None, duration=None, converge=None, max_duration=None
    )
    with pytest.raises(ValueError, match="--grid"):
        verify.settings_from(ns, None)


def test_settings_flag_a_different_reynolds():
    ns = verify.argparse.Namespace(
        grid=None, reynolds=80.0, duration=None, converge=None, max_duration=None
    )
    proto = {"grid": GRID, "run": {"duration": 10}, "reference": {"reynolds": 50.0}}
    settings, notes = verify.settings_from(ns, proto)
    assert settings["reynolds"] == 80.0
    assert any("Re 50" in n for n in notes)


def test_study_file_round_trips(tmp_path):
    path = protocol.write_study(
        tmp_path / "s.yaml",
        tmp_path / "bin",
        tmp_path,
        tmp_path / "out",
        [{"model": "a.obj", "grid": GRID, "duration": 5, "scale": 1.0, "args": ["--x"]}],
    )
    data = yaml.safe_load(path.read_text())
    assert data["vary_with_grid"] == ["duration", "max_duration"]
    assert study.load_study(path).cases[0].cmd[-2] == "--x"


def test_verify_reports_a_failed_grid(tmp_path):
    cds = {("base", GRID): (4.0, 0.03, 40.0), ("opt", GRID): (3.8, 0.04, 39.0)}

    class HalfBackend(CannedBackend):
        def run(self, case):
            if case.params["grid"] == FINE:
                return study.Outcome("failed", 3, 0.1, "exit code 3")
            return super().run(case)

    args = verify.argparse.Namespace(
        grid=[GRID, FINE],
        reynolds=None,
        duration=None,
        converge=None,
        max_duration=None,
        converge_window=None,
        binary=tmp_path / "solver",
        workdir=tmp_path,
        timeout=None,
        force=False,
    )
    result = verify.verify_files([candidate(tmp_path)], args, tmp_path / "out", HalfBackend(cds))
    assert result["shapes"][0]["overall"] == "baseline failed"
    assert any("failed" in w for w in result["warnings"])
