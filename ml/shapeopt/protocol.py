"""
Solver settings shared by the local shape data and the verify step.

Without --scale the solver stretches every mesh so its longest side spans
2.4 world units, which would undo a deformation that changes the length.
Every shape here runs at the baseline's scale instead. Re is set on the
baseline length, so all shapes see the same inflow speed and viscosity,
and Cd times frontal area compares drag forces.

Usage:
    ref = reference(base_vertices, reynolds=100)
    params = case_params("shape.obj", shape_vertices, ref, grid="64x32x32", duration=30)
    write_study(path, binary, workdir, output, [params])
"""

import math
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# main.c scales the longest side of a mesh to this many world units
SOLVER_SPAN = 2.4
STREAMWISE = 0


@dataclass(frozen=True)
class Reference:
    scale: float
    length: float  # streamwise extent of the baseline in OBJ units
    reynolds: float  # on the baseline length

    def to_dict(self) -> dict:
        return asdict(self)


def extent(vertices: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    return v.max(axis=0) - v.min(axis=0)


def reference(base_vertices: np.ndarray, reynolds: float) -> Reference:
    """Scale and length of the baseline, as the solver's autoscale would set them"""
    size = extent(base_vertices)
    return Reference(
        scale=SOLVER_SPAN / float(size.max()), length=float(size[STREAMWISE]), reynolds=reynolds
    )


def case_reynolds(vertices: np.ndarray, ref: Reference) -> float:
    """Re on the shape's own length that keeps the baseline viscosity"""
    return ref.reynolds * float(extent(vertices)[STREAMWISE]) / ref.length


def run_settings(
    duration: int | None = None,
    converge: float | None = None,
    max_duration: int | None = None,
    converge_window: int | None = None,
) -> dict:
    """Run length keys for a study case, a fixed duration or --converge"""
    if converge is not None:
        if max_duration is None:
            raise ValueError("--converge needs --max-duration")
        run = {"converge": float(converge), "max_duration": int(max_duration)}
        if converge_window is not None:
            run["converge_window"] = int(converge_window)
        return run
    if duration is None:
        raise ValueError("give --duration or --converge with --max-duration")
    return {"duration": int(duration)}


def nx(grid: str) -> int:
    return int(grid.split("x")[0])


def scale_run(run: dict, grid: str, first_grid: str) -> dict:
    """Same flow-throughs on another grid, durations grow with Nx"""
    factor = nx(grid) / nx(first_grid)
    out = dict(run)
    for key in ("duration", "max_duration"):
        if key in out:
            out[key] = math.ceil(out[key] * factor)
    return out


def case_params(model: str | Path, vertices: np.ndarray, ref: Reference, **run) -> dict:
    """Study case for one shape, with --strict-reynolds so a capped Re fails"""
    return {
        "model": str(model),
        "scale": ref.scale,
        "reynolds": case_reynolds(vertices, ref),
        **run,
        "args": ["--strict-reynolds"],
    }


def write_study(
    path: Path,
    binary: Path,
    workdir: Path,
    output: Path,
    cases: list[dict],
    name: str | None = None,
    timeout: float | None = None,
) -> Path:
    """Study file with absolute paths, for study.load_study"""
    data = {
        "name": name or path.stem,
        "binary": str(Path(binary).resolve()),
        "workdir": str(Path(workdir).resolve()),
        "output": str(Path(output).resolve()),
        "cases": cases,
        "vary_with_grid": ["duration", "max_duration"],
    }
    if timeout:
        data["timeout"] = timeout
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
