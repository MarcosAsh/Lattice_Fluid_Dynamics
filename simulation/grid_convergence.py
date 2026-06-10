"""
Grid convergence study for LBM drag prediction (#155).

Runs the Ahmed 25 deg body at several grid resolutions on Modal,
collects Cd statistics from the converged tail of each run, and
Richardson-extrapolates the grid-independent Cd.

Usage:
    modal run simulation/grid_convergence.py
    modal run simulation/grid_convergence.py --duration=30 --model=ahmed25

Results land in docs/grid_convergence/ (results.csv + raw JSON).
"""

import csv
import json
import math
import time
from pathlib import Path

from modal_worker import app, render_simulation

# grid -> simulated seconds. Finer grids need more wall-clock seconds per
# flow-through, so duration scales up to keep the averaging window (in
# flow-throughs) roughly comparable across resolutions.
DEFAULT_GRIDS = "32x16x16:30,64x32x32:30,128x64x64:30,256x128x128:30"

OUT_DIR = Path(__file__).parent.parent / "docs" / "grid_convergence"


def _cd_stats(cd_series: list[float]) -> tuple[float, float, float]:
    """Mean, std and relative std over the last half of the Cd series,
    which discards what's left of the startup transient."""
    tail = cd_series[len(cd_series) // 2 :]
    if not tail:
        return float("nan"), float("nan"), float("nan")
    mean = sum(tail) / len(tail)
    var = sum((x - mean) ** 2 for x in tail) / len(tail)
    std = math.sqrt(var)
    rel = std / abs(mean) if mean else float("inf")
    return mean, std, rel


def _richardson(rows: list[dict]) -> dict | None:
    """Richardson extrapolation from the three finest grids
    (refinement ratio r=2 in every direction)."""
    if len(rows) < 3:
        return None
    c, m, f = (r["cd_mean"] for r in rows[-3:])
    if (m - f) == 0 or (c - m) / (m - f) <= 0:
        return None
    p = math.log((c - m) / (m - f)) / math.log(2.0)
    cd_inf = f + (f - m) / (2.0**p - 1.0)
    return {"order": p, "cd_extrapolated": cd_inf}


@app.local_entrypoint()
def sweep(
    model: str = "ahmed25",
    wind: float = 1.0,
    grids: str = DEFAULT_GRIDS,
    collision: int = 2,
    reynolds: float = 0,
    entropic: bool = False,
    tag: str = "",
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{model}_{tag}" if tag else model

    grid_durations = {}
    for spec in grids.split(","):
        grid, _, dur = spec.partition(":")
        grid_durations[grid.strip()] = int(dur) if dur else 30

    print(
        f"Grid convergence: {model}, wind={wind}, re={reynolds or 'auto'}, "
        f"grids={grid_durations}"
    )
    handles = {}
    t0 = time.monotonic()
    for grid, duration in grid_durations.items():
        handles[grid] = render_simulation.spawn(
            job_id=f"gridconv-{suffix}-{grid}",
            wind_speed=wind,
            viz_mode=1,
            collision_mode=collision,
            duration=duration,
            model=model,
            reynolds=reynolds,
            grid=grid,
            entropic=entropic,
        )
        print(f"  spawned {grid} ({duration}s)")

    rows = []
    for grid in grid_durations:
        result = handles[grid].get()
        elapsed = time.monotonic() - t0
        if result.get("status") != "complete":
            print(f"  {grid}: FAILED -- {result.get('error')}")
            continue

        (OUT_DIR / f"raw_{suffix}_{grid}.json").write_text(
            json.dumps(result, indent=2)
        )

        nx, ny, nz = (int(v) for v in grid.split("x"))
        cd_mean, cd_std, rel = _cd_stats(result.get("cd_series") or [])
        rows.append(
            {
                "grid": grid,
                "cells": nx * ny * nz,
                "nx": nx,
                "cd_mean": cd_mean,
                "cd_std": cd_std,
                "cd_relstd": rel,
                "cd_final": result.get("cd_value"),
                "cl_final": result.get("cl_value"),
                "re_eff": result.get("effective_re"),
                "sim_seconds": (result.get("timings") or {}).get("simulation"),
                "video_url": result.get("video_url"),
            }
        )
        print(
            f"  {grid}: Cd={cd_mean:.4f} +/- {cd_std:.4f} "
            f"(relStd {rel * 100:.2f}%), Re_eff={result.get('effective_re')}, "
            f"[{elapsed:.0f}s elapsed]"
        )

    if not rows:
        print("No successful runs.")
        return

    csv_path = OUT_DIR / f"results_{suffix}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    rich = _richardson(rows)
    if rich:
        print(
            f"Richardson: observed order p={rich['order']:.2f}, "
            f"grid-independent Cd={rich['cd_extrapolated']:.4f}"
        )
        (OUT_DIR / f"richardson_{suffix}.json").write_text(json.dumps(rich))
    else:
        print("Richardson extrapolation not possible (non-monotone Cd).")
