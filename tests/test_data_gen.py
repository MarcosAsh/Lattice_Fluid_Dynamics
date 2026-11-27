import csv
import struct
from pathlib import Path

import pytest
import yaml

import data_gen as dg

REPO = Path(__file__).resolve().parents[1]


def sweep_cfg(**over):
    cfg = {
        "models": ["car", "ahmed25"],
        "wind_speeds": {"min": 1.0, "max": 2.0, "step": 0.5},
        "reynolds_numbers": [72, 144],
        "duration": 10,
    }
    cfg.update(over)
    return cfg


def test_run_configs_are_a_cross_product():
    runs = dg.generate_run_configs(sweep_cfg())
    assert len(runs) == 2 * 3 * 2
    assert {r.wind_speed for r in runs} == {1.0, 1.5, 2.0}
    assert all(r.grid == "256x128x128" for r in runs)
    assert all(isinstance(r.reynolds, float) for r in runs)


def test_run_id_depends_on_the_physics_only():
    a = dg.RunConfig("car", 1.0, 72.0, 10, 1, 2, "8x8x8")
    b = dg.RunConfig("car", 1.0, 72.0, 10, 3, 1, "8x8x8")
    c = dg.RunConfig("car", 1.0, 144.0, 10, 1, 2, "8x8x8")
    assert a.run_id == b.run_id
    assert a.run_id != c.run_id
    assert len(a.run_id) == 12


def test_meshes_dir_adds_custom_bodies(tmp_path, monkeypatch):
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "blob.obj").write_text("v 0 0 0\n")
    monkeypatch.setattr(dg, "__file__", str(tmp_path / "data_gen.py"))
    runs = dg.generate_run_configs(sweep_cfg(models=[], meshes_dir="meshes"))
    assert {r.model for r in runs} == {"blob"}
    assert all(r.obj_path == str(mesh_dir / "blob.obj") for r in runs)


def test_tail_stats():
    assert dg.tail_stats([], 0.25) is None
    mean, std, rel = dg.tail_stats([9.0, 9.0, 2.0, 4.0], 0.5)
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(1.0)
    assert rel == pytest.approx(1.0 / 3.0)
    assert dg.tail_stats([0.0, 0.0], 1.0)[2] == float("inf")


QUALITY = {
    "cd_max": 15.0,
    "cd_min": 0.0,
    "min_cd_samples": 5,
    "tail_fraction": 0.5,
    "convergence_threshold": 0.05,
}


def result(cd_series, effective_re=None, reynolds=144.0):
    cfg = dg.RunConfig("car", 1.0, reynolds, 10, 1, 2, "8x8x8")
    return dg.RunResult(
        cfg, None, None, cd_series, [0.1] * len(cd_series), "complete", effective_re=effective_re
    )


def test_quality_accepts_a_converged_run():
    r = dg.check_quality(result([3.3] * 10), QUALITY)
    assert r.status == "complete"
    assert r.cd_value == pytest.approx(3.3)
    assert r.converged


def test_quality_flags_noisy_run_as_unconverged():
    r = dg.check_quality(result([1.0, 5.0] * 5), QUALITY)
    assert r.status == "complete"
    assert not r.converged


@pytest.mark.parametrize(
    ("series", "kw", "text"),
    [
        ([3.3] * 3, {}, "Cd samples"),
        ([20.0] * 10, {}, "outside valid range"),
        ([-1.0] * 10, {}, "outside valid range"),
        ([3.3] * 10, {"effective_re": 100.0}, "Re clamped"),
    ],
)
def test_quality_flags_bad_runs(series, kw, text):
    r = dg.check_quality(result(series, **kw), QUALITY)
    assert r.status == "flagged"
    assert text in r.error


def test_quality_ignores_failed_runs():
    r = result([])
    r.status = "error"
    assert dg.check_quality(r, QUALITY) is r
    assert r.cd_value is None


class FakeCall:
    def __init__(self, data=None, exc=None):
        self.data, self.exc = data, exc

    def get(self, timeout):
        if self.exc:
            raise self.exc
        return self.data


class FakeFn:
    def __init__(self, call):
        self.call, self.kwargs = call, None

    def spawn(self, **kwargs):
        self.kwargs = kwargs
        return self.call


def test_submit_run_success_and_failures():
    run = dg.RunConfig("car", 1.0, 72.0, 10, 1, 2, "8x8x8")
    ok = FakeFn(FakeCall({"status": "complete", "cd_series": [1.0], "cl_series": [0.0]}))
    res = dg.submit_run(run, ok)
    assert res.status == "complete"
    assert ok.kwargs["model"] == "car"
    assert ok.kwargs["job_id"] == f"datagen_{run.run_id}"

    res = dg.submit_run(run, FakeFn(FakeCall({"status": "error", "error": "boom"})))
    assert (res.status, res.error) == ("error", "boom")
    res = dg.submit_run(run, FakeFn(FakeCall(exc=TimeoutError())))
    assert "No result" in res.error


def test_submit_run_uploads_custom_mesh(tmp_path):
    obj = tmp_path / "b.obj"
    obj.write_text("v 0 0 0\n")
    run = dg.RunConfig("blob", 1.0, 72.0, 10, 1, 2, "8x8x8", obj_path=str(obj))
    fn = FakeFn(FakeCall({"status": "complete"}))
    dg.submit_run(run, fn)
    assert fn.kwargs["model"] == "custom"
    assert fn.kwargs["obj_data"]


@pytest.fixture
def dataset_dir(tmp_path):
    saved = (dg.DATASET_DIR, dg.MANIFEST_FILE, dg.RESULTS_FILE, dg.BINARY_FILE, dg.STATS_FILE)
    dg.set_dataset_dir(tmp_path / "ds")
    yield tmp_path / "ds"
    dg.DATASET_DIR, dg.MANIFEST_FILE, dg.RESULTS_FILE, dg.BINARY_FILE, dg.STATS_FILE = saved


def test_manifest_results_binary_and_stats(dataset_dir):
    for model, cd in [("car", 3.0), ("ahmed25", 2.0), ("custom_blob", 1.0)]:
        cfg = dg.RunConfig(model, 1.0, 144.0, 10, 1, 2, "8x8x8")
        r = dg.RunResult(cfg, cd, 0.5, [cd] * 4, [0.5] * 4, "complete", cd_relstd=0.01)
        r.converged = True
        dg.save_manifest_row(r)
        dg.save_results_csv(r)

    assert len(dg.load_completed_runs()) == 3
    with dg.MANIFEST_FILE.open() as f:
        assert len(list(csv.DictReader(f))) == 3

    dg.export_binary_dataset()
    raw = dg.BINARY_FILE.read_bytes()
    magic, version, n, feats = struct.unpack("<IIII", raw[:16])
    assert (magic, version, n, feats) == (0x4C415454, 1, 2, 5)
    first = struct.unpack("<5f", raw[16:36])
    assert first == pytest.approx((1.0, 144.0, 0.0, 3.0, 0.5))

    stats = dg.compute_stats()
    assert stats["total"] == 3
    assert stats["converged"] == 3
    assert stats["by_model"]["car"]["cd_mean"] == pytest.approx(3.0)
    assert dg.STATS_FILE.exists()


def test_shipped_sweep_config_loads():
    cfg = dg.load_sweep_config(str(REPO / "ml" / "sweep_config.yaml"))
    assert len(dg.generate_run_configs(cfg)) == 3 * 1 * 5
    assert yaml.safe_load(Path(REPO / "ml" / "sweep_config.yaml").read_text())
