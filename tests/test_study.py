import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

import study as st
from pipeline_config import ConfigError

MODEL = "assets/3d-files/ahmed_25deg_m.obj"
GRIDS = ["32x16x16", "48x24x24", "64x32x32"]
COMMIT = "b43eb39dfb97ab88f5c07fd8231c49d46028f21a"


def manifest(params, cd=0.4, sem=0.01, commit=COMMIT, dirty=False, renderer="Mesa Intel"):
    """Schema 1 manifest shaped like the solver's"""
    grid = [int(n) for n in params["grid"].split("x")]
    return {
        "schema_version": 1,
        "program": "3d_fluid_simulation_car",
        "git": {"describe": commit[:7], "commit": commit, "dirty": dirty},
        "wall_seconds": 5.4,
        "gl": {"vendor": "Intel", "renderer": renderer, "version": "4.6"},
        "parameters": {"grid": grid, "model": params["model"]},
        "lattice": {
            "grid": grid,
            "tau": 0.52,
            "mach": 0.0866,
            "reynolds": 144.0,
            "reynolds_effective": 144.0,
            "char_length": 0.3 * grid[0],
        },
        "model": {"solid_cells": 730, "blockage_factor": 0.92},
        "run": {"frames": 300},
        "results": {"cd_mean": cd + 1, "cd_corr_mean": cd + 0.9, "flow_throughs": 2.3},
        "convergence": {
            "method": "mser5+lugsail_obm+hw_cvm+trend",
            "enabled": True,
            "converged": True,
            "stationary": True,
            "cd_mean": cd,
            "cd_sem": sem,
            "cd_ci95_half_width": 2.1 * sem,
            "effective_sample_size": 120.0,
        },
    }


class FakeBackend(st.Backend):
    name = "fake"

    def __init__(self, fail=(), silent=(), **per_grid):
        self.calls = []
        self.fail = set(fail)
        self.silent = set(silent)
        self.per_grid = per_grid

    def run(self, case):
        self.calls.append(case.name)
        grid = case.params["grid"]
        if grid in self.fail:
            return st.Outcome("failed", 3, 0.1, "exit code 3")
        if grid not in self.silent:
            kw = self.per_grid.get(grid.replace("x", "_"), {})
            case.manifest.write_text(json.dumps(manifest(case.params, **kw)))
        return st.Outcome("ok", 0, 0.1)


def study_data(**extra):
    data = {
        "binary": "bin/solver",
        "workdir": ".",
        "output": "out",
        "base": {"model": MODEL, "duration": 3},
        "sweep": {"grid": list(GRIDS)},
    }
    data.update(extra)
    return data


def write_study(tmp_path, data):
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def study(tmp_path):
    return st.load_study(write_study(tmp_path, study_data()))


def quiet(_msg):
    pass


def test_names_are_stable():
    p = st._normalise({"model": MODEL, "grid": "32x16x16", "duration": 3, "wind": 1})
    assert st.case_name(p) == "ahmed_25deg_m-32x16x16-cdea6f43"
    q = st._normalise({"wind": 1.0, "duration": 3, "grid": "32x16x16", "model": MODEL})
    assert st.case_name(q) == st.case_name(p)
    assert st.case_name({**p, "reynolds": 500.0}) != st.case_name(p)


def test_groups_ignore_grid_and_listed_keys():
    a = {"model": MODEL, "grid": "32x16x16", "duration": 3}
    b = {"model": MODEL, "grid": "64x32x32", "duration": 6}
    assert st.group_name(a) != st.group_name(b)
    assert st.group_name(a, ("duration",)) == st.group_name(b, ("duration",))
    assert st.group_name({**a, "duration": 6}) == st.group_name(b)


def test_command_line_and_case_files(tmp_path):
    data = study_data(base={"model": MODEL, "duration": 3, "wind": 1, "mrt": True})
    data["base"]["args"] = ["--sponge=4"]
    s = st.load_study(write_study(tmp_path, data))
    case = s.cases[0]
    assert case.cmd == [
        str(tmp_path / "bin" / "solver"),
        f"--model={MODEL}",
        "--grid=32x16x16",
        "--wind=1",
        "--duration=3",
        "--mrt",
        "--sponge=4",
        f"--manifest={case.manifest}",
    ]
    st.run_study(s, FakeBackend(), log=quiet)
    cmd = (case.dir / "cmd.txt").read_text().splitlines()
    assert cmd[0] == f"cd {tmp_path}"
    assert cmd[1].endswith(f"--manifest={case.manifest}")
    assert json.loads((case.dir / "params.json").read_text())["params"] == case.params


def test_run_collects_and_resumes(study):
    fake = FakeBackend()
    result, counts = st.run_study(study, fake, log=quiet)
    assert counts["ok"] == 3
    assert len(fake.calls) == 3
    assert [r["status"] for r in result["rows"]] == ["ok"] * 3
    assert result["warnings"] == []

    again = FakeBackend()
    _, counts = st.run_study(study, again, log=quiet)
    assert again.calls == []
    assert counts["resumed"] == 3

    forced = FakeBackend()
    st.run_study(study, forced, force=True, log=quiet)
    assert len(forced.calls) == 3


def test_changed_parameters_make_a_new_case(tmp_path):
    path = write_study(tmp_path, study_data())
    st.run_study(st.load_study(path), FakeBackend(), log=quiet)
    data = study_data(base={"model": MODEL, "duration": 4})
    path.write_text(yaml.safe_dump(data))
    fake = FakeBackend()
    st.run_study(st.load_study(path), fake, log=quiet)
    assert len(fake.calls) == 3


def test_failure_is_recorded_and_does_not_stop(study):
    fake = FakeBackend(fail={"48x24x24"})
    result, counts = st.run_study(study, fake, log=quiet)
    assert len(fake.calls) == 3
    assert counts == {"ok": 2, "failed": 1}
    failed = [r for r in result["rows"] if r["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["returncode"] == 3
    assert failed[0]["error"] == "exit code 3"
    assert result["failures"][0]["name"] == failed[0]["name"]
    assert any(w.startswith("1 case(s) failed") for w in result["warnings"])

    retry = FakeBackend()
    st.run_study(study, retry, log=quiet)
    assert retry.calls == [failed[0]["name"]]


def test_missing_manifest_is_a_failure(study):
    result, _ = st.run_study(study, FakeBackend(silent={"32x16x16"}), log=quiet)
    row = result["rows"][0]
    assert row["status"] == "failed"
    assert row["error"] == "no manifest written"


def test_mixed_commits_and_dirty_tree_warn(study):
    fake = FakeBackend(
        **{"48_24_24": {"commit": "1" * 40}, "64_32_32": {"dirty": True}},
    )
    result, counts = st.run_study(study, fake, log=quiet)
    text = st.summary(study, result, counts)
    assert "WARNING: mixed commits: b43eb39 (2), 1111111 (1)" in text
    assert "WARNING: 1 case(s) built from a dirty tree: ahmed_25deg_m-64x32x32" in text


def test_capped_reynolds_warns(tmp_path):
    data = study_data(base={"model": MODEL, "duration": 3, "reynolds": 500})
    s = st.load_study(write_study(tmp_path, data))
    result, _ = st.run_study(s, FakeBackend(), log=quiet)
    assert any(
        w.startswith("3 case(s) ran below the requested Re: ahmed_25deg_m-32x16x16")
        and "(500 to 144)" in w
        for w in result["warnings"]
    )


def test_table_columns_and_values(study):
    fake = FakeBackend(**{"64_32_32": {"cd": 0.35, "sem": 0.004}})
    st.run_study(study, fake, log=quiet)
    with (study.output / "results.csv").open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == st.COLUMNS
    row = dict(zip(rows[0], rows[3], strict=True))
    assert row["grid"] == "64x32x32"
    assert row["cells"] == str(64 * 32 * 32)
    assert float(row["cd"]) == pytest.approx(0.35)
    assert float(row["cd_sem"]) == pytest.approx(0.004)
    assert float(row["cd_ci95"]) == pytest.approx(0.0084)
    assert float(row["cd_ess"]) == pytest.approx(120.0)
    assert row["cd_stationary"] == "true"
    assert row["cd_source"] == "convergence"
    assert float(row["cd_mean"]) == pytest.approx(1.35)
    assert row["converged"] == "true"
    assert row["dirty"] == "false"
    assert row["commit"] == COMMIT
    assert row["renderer"] == "Mesa Intel"
    assert row["mach"] == "0.0866"
    assert row["tau"] == "0.52"
    assert row["error"] == ""

    data = json.loads((study.output / "results.json").read_text())
    assert data["columns"] == st.COLUMNS
    assert data["rows"][2]["params"]["grid"] == "64x32x32"


def test_results_without_statistics_have_no_sem():
    m = manifest({"grid": "32x16x16", "model": MODEL})
    m["convergence"].update(enabled=False, converged=False, cd_mean=None, cd_sem=None)
    row = st.manifest_row(m)
    assert row["cd"] == m["results"]["cd_mean"]
    assert row["cd_sem"] is None
    assert row["cd_stationary"] is None
    assert row["cd_source"] == "results"
    assert row["converged"] is None


def test_statistics_without_converge():
    m = manifest({"grid": "32x16x16", "model": MODEL}, cd=0.4, sem=0.01)
    m["convergence"].update(enabled=False, converged=False, stationary=False)
    row = st.manifest_row(m)
    assert row["cd"] == 0.4
    assert row["cd_sem"] == 0.01
    assert row["cd_source"] == "convergence"
    assert row["converged"] is None
    row.update(name="a", status="ok", params={}, commit="c", dirty=False, renderer="r")
    assert any("trend in Cd" in w for w in st.check_rows([row]))


def test_old_manifest_sem_is_dropped():
    m = manifest({"grid": "32x16x16", "model": MODEL}, cd=0.4, sem=0.01)
    m["convergence"] = {"enabled": True, "converged": True, "cd_mean": 0.4, "cd_sem": 0.01}
    row = st.manifest_row(m)
    assert row["cd"] == 0.4
    assert row["cd_sem"] is None
    assert row["cd_source"] == "convergence"


FAKE_SOLVER = """#!{python}
import json, os, sys, time
args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if "=" in a)
if "sleep" in args:
    time.sleep(float(args["sleep"]))
if "fail" in args:
    print("boom")
    sys.exit(int(args["fail"]))
grid = [int(n) for n in args["grid"].split("x")]
m = {{"schema_version": 1, "lattice": {{"grid": grid}}, "results": {{"cd_mean": 0.4}},
     "git": {{"commit": "abc", "dirty": False}}, "gl": {{"renderer": "fake"}}}}
with open(args["manifest"], "w") as f:
    json.dump(m, f)
print("ran in", os.getcwd())
"""


def test_local_backend_with_a_fake_solver(tmp_path, capsys):
    solver = tmp_path / "bin" / "solver"
    solver.parent.mkdir()
    solver.write_text(FAKE_SOLVER.format(python=sys.executable))
    solver.chmod(0o755)
    data = study_data(
        timeout=1,
        sweep={"grid": ["32x16x16"]},
        cases=[{}, {"args": ["--fail=4"]}, {"args": ["--sleep=10"]}],
    )
    path = write_study(tmp_path, data)
    assert st.main([str(path)]) == 1
    out = capsys.readouterr().out
    assert "1 ok" in out
    assert "2 failed" in out

    rows = json.loads((tmp_path / "out" / "results.json").read_text())["rows"]
    assert [r["status"] for r in rows] == ["ok", "failed", "failed"]
    assert rows[1]["returncode"] == 4
    assert rows[1]["error"] == "exit code 4"
    assert rows[2]["error"] == "timeout after 1 s"
    log = (tmp_path / "out" / "cases" / rows[0]["name"] / "log.txt").read_text()
    assert f"ran in {tmp_path}" in log


def test_missing_binary_is_reported(tmp_path, capsys):
    assert st.main([str(write_study(tmp_path, study_data()))]) == 2
    assert "no solver binary" in capsys.readouterr().err


def test_dry_run_writes_nothing(tmp_path, capsys):
    path = write_study(tmp_path, study_data())
    assert st.main([str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("--manifest=") == 3
    assert not (tmp_path / "out").exists()


def test_slurm_writes_scripts_without_submitting(tmp_path):
    data = study_data(slurm={"time": "00:20:00", "setup": ["module load mesa"]})
    s = st.load_study(write_study(tmp_path, data))
    result, counts = st.run_study(s, st.SlurmBackend(s.slurm), log=quiet)
    assert counts["pending"] == 3
    assert [r["status"] for r in result["rows"]] == ["pending"] * 3

    script = (s.cases[0].dir / "job.sbatch").read_text()
    assert "#SBATCH --gres=shard:1" in script
    assert "#SBATCH --time=00:20:00" in script
    assert "module load mesa" in script
    assert f"cd {tmp_path} || exit 1" in script
    submit = (s.output / "submit.sh").read_text().splitlines()
    assert sum(line.startswith("sbatch ") for line in submit) == 3

    case = s.cases[0]
    case.manifest.write_text(json.dumps(manifest(case.params)))
    (case.dir / "status.json").write_text('{"backend": "slurm", "returncode": 0, "job_id": "7"}')
    rows = st.collect(s)["rows"]
    assert [r["status"] for r in rows] == ["ok", "pending", "pending"]
    _, counts = st.run_study(s, st.SlurmBackend(s.slurm), log=quiet)
    assert counts == {"resumed": 1, "pending": 2}


def test_slurm_submit_records_the_job(tmp_path):
    sbatch = tmp_path / "sbatch"
    sbatch.write_text("#!/bin/sh\necho Submitted batch job 42\n")
    sbatch.chmod(0o755)
    s = st.load_study(write_study(tmp_path, study_data()))
    backend = st.SlurmBackend({}, submit=True, sbatch=str(sbatch))
    result, _ = st.run_study(s, backend, log=quiet)
    assert {r["job_id"] for r in result["rows"]} == {"42"}
    assert not (s.output / "submit.sh").exists()

    again = st.SlurmBackend({}, submit=True, sbatch=str(tmp_path / "missing"))
    _, counts = st.run_study(s, again, log=quiet)
    assert counts == {"pending": 3}


def problems(data):
    return st.validate(data)


def test_valid_study_has_no_problems():
    assert problems(study_data()) == []


def test_shipped_example_is_valid():
    example = Path(__file__).resolve().parents[1] / "ml" / "examples" / "ahmed25_grid.yaml"
    s = st.load_study(example)
    assert [c.params["grid"] for c in s.cases] == GRIDS
    assert len({c.group for c in s.cases}) == 1
    assert s.workdir.name == "simulation"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"binray": "x"}, "binray: unknown key (did you mean 'binary'?)"),
        ({"sweep": {"grid": ["32x16x16", "big"]}}, "sweep.grid[1]: expected a grid"),
        ({"sweep": {"gird": ["32x16x16"]}}, "sweep.gird: cannot sweep this key (did you mean"),
        ({"sweep": {"grid": "32x16x16"}}, "sweep.grid: expected a non-empty list"),
        ({"cases": [{"grid": "32x16x16"}]}, "cases[0].grid: also set in sweep.grid"),
        ({"cases": [{"wind": 9}]}, "cases[0].wind: must be <= 5, got 9"),
        ({"base": {"duration": 3}}, "needs a model"),
        ({"base": {"model": MODEL}}, "needs duration or max_duration"),
        ({"base": {"model": MODEL, "max_duration": 9}}, "max_duration needs converge"),
        ({"base": {"model": MODEL, "duration": 3, "converge": 1.5}}, "must be < 1, got 1.5"),
        ({"cases": [{}, {}]}, "same parameters as case 1"),
        ({"base": {"model": MODEL, "duration": 3, "args": ["--manifest=x"]}}, "sets --manifest"),
        ({"backend": "pbs"}, "backend: must be one of local, slurm"),
        ({"vary_with_grid": ["grid"]}, "vary_with_grid: unknown key 'grid'"),
    ],
)
def test_validation_problems(change, expected):
    found = problems(study_data(**change))
    assert any(expected in p for p in found), found


def test_load_error_names_the_file(tmp_path):
    path = write_study(tmp_path, study_data(output=3))
    with pytest.raises(ConfigError, match=r"study\.yaml") as err:
        st.load_study(path)
    assert "output: expected a string, got 3" in str(err.value)
