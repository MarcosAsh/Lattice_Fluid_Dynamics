"""
Run a list of solver cases and collect their manifests into one table.

A study file (YAML or JSON) names the solver binary, the directory to run
it from, an output directory and the cases. Each case runs in its own
directory with its command line, log and manifest. A case whose manifest
already exists for the same parameters is skipped, so a study resumes.

Study file (paths are relative to the study file):
    binary: ../../build/3d_fluid_simulation_car
    workdir: ../../simulation          # the solver loads shaders from here
    output: ../../build/studies/ahmed25
    timeout: 600                       # seconds per case
    base: {model: assets/3d-files/ahmed_25deg_m.obj, duration: 3}
    sweep: {grid: [32x16x16, 48x24x24, 64x32x32]}
    cases: [{reynolds: 0}, {reynolds: 500}]   # crossed with sweep
    vary_with_grid: [duration]         # keys that do not split a GCI group
    slurm: {gres: "shard:1", time: "00:30:00"}

Usage:
    python ml/study.py study.yaml                  # run locally, then collect
    python ml/study.py study.yaml --dry-run        # print the commands only
    python ml/study.py study.yaml --backend slurm  # write sbatch scripts
    python ml/study.py study.yaml --backend slurm --submit
    python ml/study.py study.yaml --collect        # rebuild the results only
"""

import argparse
import csv
import difflib
import hashlib
import itertools
import json
import math
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml

from pipeline_config import ConfigError
from pipeline_config import Field
from pipeline_config import _validate_mapping
from pipeline_config import _validate_value

MANIFEST_SCHEMA = 1
BACKENDS = ("local", "slurm")


def _below_one(v: float) -> str | None:
    return None if v < 1 else f"must be < 1, got {v!r}"


def _no_manifest_flag(v: list) -> str | None:
    bad = [a for a in v if str(a).startswith("--manifest")]
    return f"the study sets --manifest itself, drop {bad[0]!r}" if bad else None


def _backend(v: str) -> str | None:
    return None if v in BACKENDS else f"must be one of {', '.join(BACKENDS)}, got {v!r}"


# Solver options a case can set, in command line order
PARAM_FIELDS = (
    Field("model", "str", required=False),
    Field("grid", "grid", required=False),
    Field("reynolds", "number", required=False, min=0),
    Field("wind", "number", required=False, min=0, max=5),
    Field("collision", "int", required=False, min=0, max=3),
    Field("viz", "int", required=False, min=0),
    Field("duration", "int", required=False, min=1),
    Field("converge", "number", required=False, min=0, exclusive_min=True, check=_below_one),
    Field("converge_window", "int", required=False, min=4, max=100000),
    Field("max_duration", "int", required=False, min=1),
    Field("mrt", "bool", required=False),
    Field("entropic", "bool", required=False),
    Field("smagorinsky", "number", required=False, min=0, max=0.5),
    Field("turbulence", "number", required=False, min=0, max=0.3),
    Field("sponge", "int", required=False, min=0),
    Field("scale", "number", required=False, min=0, exclusive_min=True),
    Field("angle", "int", required=False),
    Field("args", "list", required=False, item="str", check=_no_manifest_flag),
)
PARAMS = {f.key: f for f in PARAM_FIELDS}

SLURM_FIELDS = (
    Field("gres", "str", required=False),
    Field("time", "str", required=False),
    Field("partition", "str", required=False),
    Field("cpus", "int", required=False, min=1),
    Field("setup", "list", required=False, item="str"),
    Field("extra", "list", required=False, item="str"),
)

STUDY_FIELDS = (
    Field("name", "str", required=False),
    Field("binary", "str"),
    Field("workdir", "str", required=False),
    Field("output", "str"),
    Field("backend", "str", required=False, check=_backend),
    Field("timeout", "number", required=False, min=0, exclusive_min=True),
    Field("base", "dict", required=False, fields=PARAM_FIELDS),
    Field("sweep", "dict", required=False),
    Field("cases", "list", required=False, item="dict", nonempty=True),
    Field("vary_with_grid", "list", required=False, item="str"),
    Field("slurm", "dict", required=False, fields=SLURM_FIELDS),
)

COLUMNS = [
    "name",
    "group",
    "status",
    "model",
    "grid",
    "cells",
    "reynolds",
    "mach",
    "tau",
    "char_length",
    "solid_cells",
    "blockage_factor",
    "cd",
    "cd_sem",
    "cd_source",
    "cd_mean",
    "cd_corr_mean",
    "converged",
    "frames",
    "flow_throughs",
    "wall_seconds",
    "commit",
    "dirty",
    "renderer",
    "returncode",
    "error",
    "job_id",
    "manifest",
]


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    params: dict
    cmd: list[str]
    workdir: Path
    dir: Path
    timeout: float | None

    @property
    def manifest(self) -> Path:
        return self.dir / "manifest.json"


@dataclass
class Study:
    path: Path
    name: str
    binary: Path
    workdir: Path
    output: Path
    timeout: float | None
    backend: str
    slurm: dict
    cases: list[Case]


@dataclass
class Outcome:
    status: str  # ok, failed or pending
    returncode: int | None = None
    seconds: float | None = None
    error: str | None = None
    job_id: str | None = None
    extra: dict = field(default_factory=dict)


# Study file


def _normalise(params: dict) -> dict:
    """Coerce numbers so 1 and 1.0 give the same case"""
    out = {}
    for key, value in params.items():
        kind = PARAMS[key].kind
        if kind == "number":
            value = float(value)
        elif kind == "int":
            value = int(value)
        out[key] = value
    return out


def expand(data: dict) -> list[dict]:
    """Cases crossed with the sweep, each over base"""
    base = data.get("base") or {}
    sweep = data.get("sweep") or {}
    keys = list(sweep)
    combos = [
        dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(sweep[k] for k in keys))
    ]
    cases = data.get("cases") or [{}]
    return [_normalise({**base, **case, **combo}) for case in cases for combo in combos]


def _case_problems(i: int, params: dict) -> list[str]:
    where = f"case {i + 1} ({params.get('grid', 'no grid')})"
    problems = []
    if "model" not in params:
        problems.append(f"{where}: needs a model")
    if "grid" not in params:
        problems.append(f"{where}: needs a grid")
    if "duration" not in params and "max_duration" not in params:
        problems.append(f"{where}: needs duration or max_duration to run headless")
    if "max_duration" in params and "converge" not in params:
        problems.append(f"{where}: max_duration needs converge")
    return problems


def validate(data: Any) -> list[str]:
    """Return every problem with a study file, empty when it is valid"""
    if not isinstance(data, dict):
        return ["top level must be a mapping of keys to values"]
    problems: list[str] = []
    _validate_mapping(data, STUDY_FIELDS, "", problems)

    sweep = data.get("sweep")
    swept = set()
    if isinstance(sweep, dict):
        for key, values in sweep.items():
            where = f"sweep.{key}"
            if key not in PARAMS or key == "args":
                near = difflib.get_close_matches(str(key), sorted(PARAMS), n=1)
                hint = f" (did you mean {near[0]!r}?)" if near else ""
                problems.append(f"{where}: cannot sweep this key{hint}")
            elif not isinstance(values, list) or not values:
                problems.append(f"{where}: expected a non-empty list, got {values!r}")
            else:
                swept.add(key)
                for i, v in enumerate(values):
                    _validate_value(PARAMS[key], v, f"{where}[{i}]", problems)

    cases = data.get("cases")
    if isinstance(cases, list):
        for i, case in enumerate(cases):
            if not isinstance(case, dict):
                continue
            _validate_mapping(case, PARAM_FIELDS, f"cases[{i}].", problems)
            problems.extend(
                f"cases[{i}].{k}: also set in sweep.{k}" for k in sorted(swept & set(case))
            )

    vary = data.get("vary_with_grid")
    if isinstance(vary, list):
        problems.extend(
            f"vary_with_grid: unknown key {k!r}" for k in vary if k not in PARAMS or k == "grid"
        )

    if problems:
        return problems
    names: dict[str, int] = {}
    for i, params in enumerate(expand(data)):
        problems += _case_problems(i, params)
        if "model" in params and "grid" in params:
            name = case_name(params)
            if name in names:
                problems.append(f"case {i + 1}: same parameters as case {names[name] + 1}")
            names.setdefault(name, i)
    return problems


def _digest(d: dict) -> str:
    text = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "case"


def case_name(params: dict) -> str:
    """Readable name plus a hash of every parameter"""
    return f"{_slug(Path(params['model']).stem)}-{params['grid']}-{_digest(params)}"


def group_name(params: dict, vary_with_grid: tuple[str, ...] = ()) -> str:
    """Name shared by cases that differ only in grid"""
    rest = {k: v for k, v in params.items() if k != "grid" and k not in vary_with_grid}
    return f"{_slug(Path(params['model']).stem)}-{_digest(rest)}"


def _flag_value(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def build_command(binary: Path, params: dict, manifest: Path) -> list[str]:
    cmd = [str(binary)]
    for f in PARAM_FIELDS:
        if f.key not in params or f.key == "args":
            continue
        flag = "--" + f.key.replace("_", "-")
        value = params[f.key]
        if f.kind == "bool":
            if value:
                cmd.append(flag)
        else:
            cmd.append(f"{flag}={_flag_value(value)}")
    cmd += params.get("args", [])
    cmd.append(f"--manifest={manifest}")
    return cmd


def load_study(path: str | Path, timeout: float | None = None) -> Study:
    """Read and validate a study file and build its cases"""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(path, [f"not valid YAML or JSON: {e}"]) from e
    problems = validate(data)
    if problems:
        raise ConfigError(path, problems)

    here = path.resolve().parent
    binary = (here / data["binary"]).resolve()
    workdir = (here / data.get("workdir", ".")).resolve()
    output = (here / data["output"]).resolve()
    timeout = timeout or data.get("timeout")
    vary = tuple(data.get("vary_with_grid", ()))
    cases = []
    for params in expand(data):
        name = case_name(params)
        case_dir = output / "cases" / name
        cases.append(
            Case(
                name=name,
                group=group_name(params, vary),
                params=params,
                cmd=build_command(binary, params, case_dir / "manifest.json"),
                workdir=workdir,
                dir=case_dir,
                timeout=timeout,
            )
        )
    return Study(
        path=path,
        name=data.get("name", path.stem),
        binary=binary,
        workdir=workdir,
        output=output,
        timeout=timeout,
        backend=data.get("backend", "local"),
        slurm=data.get("slurm", {}),
        cases=cases,
    )


# Backends


class Backend:
    """Runs one case"""

    name = "base"

    def run(self, case: Case) -> Outcome:
        raise NotImplementedError

    def finish(self, study: Study, ran: list[Case]) -> None:
        pass


class LocalBackend(Backend):
    """One case at a time on this machine"""

    name = "local"

    def run(self, case: Case) -> Outcome:
        start = time.monotonic()
        with (case.dir / "log.txt").open("wb") as log:
            try:
                proc = subprocess.run(
                    case.cmd,
                    cwd=case.workdir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=case.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return Outcome(
                    "failed", None, time.monotonic() - start, f"timeout after {case.timeout:g} s"
                )
            except OSError as e:
                return Outcome("failed", None, 0.0, f"cannot start {case.cmd[0]}: {e}")
        seconds = time.monotonic() - start
        if proc.returncode != 0:
            return Outcome("failed", proc.returncode, seconds, f"exit code {proc.returncode}")
        return Outcome("ok", 0, seconds)


class SlurmBackend(Backend):
    """Writes one sbatch script per case, submits only when asked"""

    name = "slurm"

    def __init__(self, options: dict, submit: bool = False, sbatch: str = "sbatch") -> None:
        self.options = options
        self.submit = submit
        self.sbatch = sbatch

    def script(self, case: Case) -> str:
        q = shlex.quote
        o = self.options
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={case.name}",
            f"#SBATCH --output={case.dir / 'slurm-%j.out'}",
            f"#SBATCH --gres={o.get('gres', 'shard:1')}",
        ]
        if o.get("time"):
            lines.append(f"#SBATCH --time={o['time']}")
        elif case.timeout:
            lines.append(f"#SBATCH --time={math.ceil(case.timeout / 60)}")
        if o.get("partition"):
            lines.append(f"#SBATCH --partition={o['partition']}")
        if o.get("cpus"):
            lines.append(f"#SBATCH --cpus-per-task={o['cpus']}")
        lines += [f"#SBATCH {x}" for x in o.get("extra", [])]
        lines += o.get("setup", [])
        status = q(str(case.dir / "status.json"))
        lines += [
            f"cd {q(str(case.workdir))} || exit 1",
            f"{shlex.join(case.cmd)} > {q(str(case.dir / 'log.txt'))} 2>&1",
            "rc=$?",
            'printf \'{"backend": "slurm", "returncode": %d, "job_id": "%s"}\\n\' '
            f'"$rc" "$SLURM_JOB_ID" > {status}',
            "exit $rc",
        ]
        return "\n".join(lines) + "\n"

    def run(self, case: Case) -> Outcome:
        path = case.dir / "job.sbatch"
        path.write_text(self.script(case))
        if not self.submit:
            return Outcome("pending")
        try:
            proc = subprocess.run(
                [self.sbatch, str(path)], capture_output=True, text=True, check=False
            )
        except OSError as e:
            return Outcome("failed", None, None, f"cannot run {self.sbatch}: {e}")
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip()
            return Outcome("failed", proc.returncode, None, f"sbatch failed: {err}")
        words = proc.stdout.split()
        return Outcome("pending", job_id=words[-1] if words else None)

    def finish(self, study: Study, ran: list[Case]) -> None:
        if self.submit or not ran:
            return
        path = study.output / "submit.sh"
        lines = ["#!/bin/sh", "set -e"]
        lines += [f"{self.sbatch} {shlex.quote(str(c.dir / 'job.sbatch'))}" for c in ran]
        path.write_text("\n".join(lines) + "\n")
        path.chmod(0o755)


# Running and collecting


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_done(case: Case) -> bool:
    """True when the case has a manifest for the same parameters and exit code 0"""
    saved = _read_json(case.dir / "params.json")
    status = _read_json(case.dir / "status.json") or {}
    return (
        saved is not None
        and saved.get("params") == case.params
        and _read_json(case.manifest) is not None
        and status.get("returncode", 0) == 0
        and status.get("status") != "failed"
    )


def _pending_job(case: Case) -> str | None:
    status = _read_json(case.dir / "status.json") or {}
    if status.get("status") == "pending" and "returncode" not in status:
        return status.get("job_id")
    return None


def _prepare(case: Case) -> None:
    case.dir.mkdir(parents=True, exist_ok=True)
    for stale in ("manifest.json", "status.json"):
        (case.dir / stale).unlink(missing_ok=True)
    meta = {"name": case.name, "group": case.group, "params": case.params}
    (case.dir / "params.json").write_text(json.dumps(meta, indent=2) + "\n")
    (case.dir / "cmd.txt").write_text(
        f"cd {shlex.quote(str(case.workdir))}\n{shlex.join(case.cmd)}\n"
    )


def manifest_row(m: dict) -> dict:
    """Table columns taken from a schema 1 manifest"""
    lat = m.get("lattice") or {}
    model = m.get("model") or {}
    res = m.get("results") or {}
    conv = m.get("convergence") or {}
    git = m.get("git") or {}
    gl = m.get("gl") or {}
    grid = lat.get("grid") or (m.get("parameters") or {}).get("grid")
    use_conv = bool(conv.get("enabled")) and conv.get("cd_mean") is not None
    return {
        "grid": "x".join(str(n) for n in grid) if grid else None,
        "cells": math.prod(grid) if grid else None,
        "reynolds": lat.get("reynolds_effective"),
        "mach": lat.get("mach"),
        "tau": lat.get("tau"),
        "char_length": lat.get("char_length"),
        "solid_cells": model.get("solid_cells"),
        "blockage_factor": model.get("blockage_factor"),
        "cd": conv.get("cd_mean") if use_conv else res.get("cd_mean"),
        "cd_sem": conv.get("cd_sem") if use_conv else None,
        "cd_source": "convergence" if use_conv else "results",
        "cd_mean": res.get("cd_mean"),
        "cd_corr_mean": res.get("cd_corr_mean"),
        "converged": conv.get("converged") if conv.get("enabled") else None,
        "frames": (m.get("run") or {}).get("frames"),
        "flow_throughs": res.get("flow_throughs"),
        "wall_seconds": m.get("wall_seconds"),
        "commit": git.get("commit"),
        "dirty": git.get("dirty"),
        "renderer": gl.get("renderer"),
    }


def case_row(case: Case) -> dict:
    """One results row from whatever the case directory holds"""
    row: dict[str, Any] = dict.fromkeys(COLUMNS)
    row.update(
        name=case.name, group=case.group, model=case.params["model"], grid=case.params["grid"]
    )
    status = _read_json(case.dir / "status.json")
    manifest = _read_json(case.manifest)
    rc = (status or {}).get("returncode")
    row["returncode"] = rc
    row["job_id"] = (status or {}).get("job_id")
    if status is None and manifest is None:
        row["status"] = "not run"
    elif (status or {}).get("status") == "pending" and rc is None:
        row["status"] = "pending"
    elif manifest is not None and rc in (0, None) and (status or {}).get("status") != "failed":
        row.update(manifest_row(manifest))
        row.update(status="ok", manifest=str(case.manifest))
        row["schema_version"] = manifest.get("schema_version")
    else:
        row["status"] = "failed"
        row["error"] = (status or {}).get("error") or (
            f"exit code {rc}" if rc else "no manifest written"
        )
    row["params"] = case.params
    return row


def _names(rows: list[dict], limit: int = 3) -> str:
    names = [r["name"] for r in rows]
    more = f" and {len(names) - limit} more" if len(names) > limit else ""
    return ", ".join(names[:limit]) + more


def check_rows(rows: list[dict]) -> list[str]:
    """Warnings for the summary"""
    ok = [r for r in rows if r["status"] == "ok"]
    warnings = []
    commits = Counter((r["commit"] or "unknown")[:7] for r in ok)
    if len(commits) > 1:
        warnings.append(
            "mixed commits: " + ", ".join(f"{c} ({n})" for c, n in commits.most_common())
        )
    dirty = [r for r in ok if r["dirty"]]
    if dirty:
        warnings.append(f"{len(dirty)} case(s) built from a dirty tree: {_names(dirty)}")
    renderers = Counter(r["renderer"] or "unknown" for r in ok)
    if len(renderers) > 1:
        warnings.append(
            "mixed renderers: " + ", ".join(f"{g} ({n})" for g, n in renderers.most_common())
        )
    schemas = {r.get("schema_version") for r in ok} - {MANIFEST_SCHEMA}
    if schemas:
        warnings.append(f"manifest schema_version {sorted(schemas)}, this tool reads 1")
    capped = [
        r
        for r in ok
        if r["params"].get("reynolds") and (r["reynolds"] or 0) < 0.99 * r["params"]["reynolds"]
    ]
    if capped:
        detail = ", ".join(
            f"{r['name']} ({r['params']['reynolds']:g} to {r['reynolds']:.0f})" for r in capped[:3]
        )
        more = f" and {len(capped) - 3} more" if len(capped) > 3 else ""
        warnings.append(f"{len(capped)} case(s) ran below the requested Re: {detail}{more}")
    unconverged = [r for r in ok if r["converged"] is False]
    if unconverged:
        warnings.append(f"{len(unconverged)} case(s) did not converge: {_names(unconverged)}")
    failed = [r for r in rows if r["status"] == "failed"]
    if failed:
        detail = ", ".join(f"{r['name']} ({r['error']})" for r in failed[:3])
        more = f" and {len(failed) - 3} more" if len(failed) > 3 else ""
        warnings.append(f"{len(failed)} case(s) failed: {detail}{more}")
    return warnings


def _csv_value(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return v


def collect(study: Study) -> dict:
    """Write results.csv and results.json for every case of the study"""
    rows = [case_row(c) for c in study.cases]
    warnings = check_rows(rows)
    failures = [
        {
            "name": r["name"],
            "returncode": r["returncode"],
            "error": r["error"],
            "log": str(study.output / "cases" / r["name"] / "log.txt"),
        }
        for r in rows
        if r["status"] == "failed"
    ]
    result = {
        "study": study.name,
        "file": str(study.path),
        "columns": COLUMNS,
        "rows": rows,
        "failures": failures,
        "warnings": warnings,
    }
    study.output.mkdir(parents=True, exist_ok=True)
    (study.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (study.output / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows([_csv_value(r[c]) for c in COLUMNS] for r in rows)
    return result


def run_study(
    study: Study, backend: Backend, force: bool = False, log: Callable[[str], None] = print
) -> tuple[dict, Counter]:
    """Run every case that is not done yet, then collect"""
    counts: Counter = Counter()
    ran = []
    for case in study.cases:
        if not force and is_done(case):
            counts["resumed"] += 1
            log(f"skip {case.name}: manifest exists")
            continue
        job = _pending_job(case)
        if not force and job:
            counts["pending"] += 1
            log(f"skip {case.name}: submitted as job {job}")
            continue
        _prepare(case)
        log(f"run  {case.name}")
        outcome = backend.run(case)
        if outcome.status == "ok" and _read_json(case.manifest) is None:
            outcome = Outcome("failed", outcome.returncode, outcome.seconds, "no manifest written")
        status = {
            "backend": backend.name,
            "status": outcome.status,
            "seconds": outcome.seconds,
            "error": outcome.error,
            "job_id": outcome.job_id,
            **outcome.extra,
        }
        if outcome.status != "pending":
            status["returncode"] = outcome.returncode
        # a fast Slurm job may have written its exit code already
        done = _read_json(case.dir / "status.json") or {}
        if outcome.status != "pending" or "returncode" not in done:
            (case.dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        if outcome.status == "failed":
            log(f"     failed: {outcome.error}, see {case.dir / 'log.txt'}")
        counts[outcome.status] += 1
        ran.append(case)
    backend.finish(study, ran)
    return collect(study), counts


def _fmt(v: Any, spec: str) -> str:
    return "-" if v is None else format(v, spec)


def summary(study: Study, result: dict, counts: Counter | None = None) -> str:
    rows = result["rows"]
    states = Counter(r["status"] for r in rows)
    head = f"study {study.name}: {len(rows)} cases, " + ", ".join(
        f"{n} {s}" for s, n in sorted(states.items())
    )
    if counts:
        head += " (this run: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) + ")"
    lines = [head]
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        width = max(len(r["name"]) for r in ok)
        lines.append(f"  {'case':<{width}}  {'Re':>8}  {'Cd':>8}  {'sem':>7}  converged")
        lines += [
            f"  {r['name']:<{width}}  {_fmt(r['reynolds'], '8.1f')}  {_fmt(r['cd'], '8.4f')}  "
            f"{_fmt(r['cd_sem'], '7.4f')}  {_fmt(r['converged'], '')}"
            for r in ok
        ]
    lines += [f"WARNING: {w}" for w in result["warnings"]]
    lines.append(f"wrote {study.output / 'results.csv'} and results.json")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run solver cases and collect their manifests")
    ap.add_argument("study", type=Path, help="study file (YAML or JSON)")
    ap.add_argument("--backend", choices=BACKENDS, help="override the study file's backend")
    ap.add_argument("--submit", action="store_true", help="slurm: run sbatch for each case")
    ap.add_argument("--sbatch", default="sbatch", help="sbatch command (default: sbatch)")
    ap.add_argument("--timeout", type=float, help="seconds per case, overrides the file")
    ap.add_argument("--force", action="store_true", help="rerun cases that are done")
    ap.add_argument("--collect", action="store_true", help="only rebuild the results files")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    args = ap.parse_args(argv)

    try:
        study = load_study(args.study, args.timeout)
    except (ConfigError, OSError) as e:
        print(e, file=sys.stderr)
        return 2

    if args.dry_run:
        for case in study.cases:
            done = " (done)" if is_done(case) else ""
            print(f"# {case.name}{done}")
            print(f"cd {shlex.quote(str(case.workdir))} && {shlex.join(case.cmd)}")
        return 0

    counts = None
    if args.collect:
        result = collect(study)
    else:
        kind = args.backend or study.backend
        if kind == "local" and not study.binary.is_file():
            print(f"study: no solver binary at {study.binary}", file=sys.stderr)
            return 2
        backend: Backend = (
            SlurmBackend(study.slurm, args.submit, args.sbatch)
            if kind == "slurm"
            else LocalBackend()
        )
        result, counts = run_study(study, backend, args.force)
        if kind == "slurm" and not args.submit and counts["pending"]:
            print(f"wrote sbatch scripts, submit with {study.output / 'submit.sh'}")
    print(summary(study, result, counts))
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
