"""
Training data generation pipeline for the Cd surrogate model.

Spawns simulation runs on the deployed Modal GPU worker (`fluid-sim`
app), collects force-averaged Cd/Cl series, and saves them as CSV +
binary dataset files.

Protocol follows the grid convergence study (#155,
docs/grid_convergence/REPORT.md): 256x128x128 grid, 120 s per run,
sponge layer on (solver default), Cd taken as the mean over the last
quarter of the series.

Usage:
    python data_gen.py --config sweep_config.yaml
    python data_gen.py --config sweep_config.yaml --resume
    python data_gen.py --pilot    # anchor run, checks Cd against 3.333
    python data_gen.py --status
"""

import argparse
import csv
import hashlib
import json
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path

import modal
import yaml

_csv_lock = threading.Lock()

MODAL_APP = "fluid-sim"
MODAL_FUNCTION = "render_simulation"

# Reference point from the grid convergence study (#155): ahmed25 at
# Re 144 on the 256 grid with the equilibrium sponge converged to
# Cd = 3.334 +/- 0.009 across five independent runs.
ANCHOR_MODEL = "ahmed25"
ANCHOR_WIND = 1.0
ANCHOR_RE = 144.0
ANCHOR_CD = 3.333
ANCHOR_TOL = 0.15

DATASET_DIR = Path(__file__).parent / "dataset" / "v2"
MANIFEST_FILE = DATASET_DIR / "manifest.csv"
RESULTS_FILE = DATASET_DIR / "results.csv"
BINARY_FILE = DATASET_DIR / "training_data.bin"
STATS_FILE = DATASET_DIR / "stats.json"


def set_dataset_dir(path: Path):
    """Point all output files at the configured dataset directory."""
    global DATASET_DIR, MANIFEST_FILE, RESULTS_FILE, BINARY_FILE, STATS_FILE
    DATASET_DIR = path
    MANIFEST_FILE = DATASET_DIR / "manifest.csv"
    RESULTS_FILE = DATASET_DIR / "results.csv"
    BINARY_FILE = DATASET_DIR / "training_data.bin"
    STATS_FILE = DATASET_DIR / "stats.json"


@dataclass
class RunConfig:
    model: str
    wind_speed: float
    reynolds: float
    duration: int
    viz_mode: int
    collision_mode: int
    grid: str
    # When set, the OBJ is uploaded to the worker as a custom mesh instead of
    # selecting a baked-in model by name. `model` then holds its display name.
    obj_path: str | None = None

    @property
    def run_id(self) -> str:
        key = (
            f"{self.model}_{self.wind_speed}_{self.reynolds}_"
            f"{self.grid}_{self.duration}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]


@dataclass
class RunResult:
    config: RunConfig
    cd_value: float | None
    cl_value: float | None
    cd_series: list[float]
    cl_series: list[float]
    status: str
    error: str | None = None
    converged: bool = False
    cd_relstd: float | None = None
    effective_re: float | None = None
    flow_throughs: float | None = None
    cfl: float | None = None


def load_sweep_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_run_configs(config: dict) -> list[RunConfig]:
    """Build the full list of parameter combinations.

    `models` are baked-in model names. `meshes_dir` (optional, relative to ml/)
    adds every *.obj it contains as a custom mesh uploaded to the worker, named
    by file stem.
    """
    runs = []
    ws = config["wind_speeds"]
    wind_speeds = []
    v = ws["min"]
    while v <= ws["max"] + 1e-9:
        wind_speeds.append(round(v, 2))
        v += ws["step"]

    # (display_name, obj_path) pairs; obj_path None means a baked-in model.
    bodies = [(m, None) for m in config.get("models", [])]
    if config.get("meshes_dir"):
        mesh_dir = Path(__file__).parent / config["meshes_dir"]
        bodies += [(p.stem, str(p)) for p in sorted(mesh_dir.glob("*.obj"))]

    for name, obj_path in bodies:
        for wind_speed in wind_speeds:
            for reynolds in config["reynolds_numbers"]:
                runs.append(
                    RunConfig(
                        model=name,
                        obj_path=obj_path,
                        wind_speed=wind_speed,
                        reynolds=float(reynolds),
                        duration=config["duration"],
                        viz_mode=config.get("viz_mode", 1),
                        collision_mode=config.get("collision_mode", 2),
                        grid=config.get("grid", "256x128x128"),
                    )
                )
    return runs


def load_completed_runs() -> set[str]:
    """Load run IDs that have already completed."""
    if not MANIFEST_FILE.exists():
        return set()
    completed = set()
    with open(MANIFEST_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "complete":
                completed.add(row["run_id"])
    return completed


def tail_stats(series: list[float], fraction: float):
    """Mean, std and relative std over the trailing fraction of a series."""
    if not series:
        return None
    n = max(1, int(len(series) * fraction))
    tail = series[-n:]
    mean = sum(tail) / len(tail)
    std = (sum((x - mean) ** 2 for x in tail) / len(tail)) ** 0.5
    rel = std / abs(mean) if abs(mean) > 1e-9 else float("inf")
    return mean, std, rel


def submit_run(
    run: RunConfig, render_fn, timeout: int = 2400
) -> RunResult:
    """Spawn a single simulation run on Modal and wait for the result."""

    def _error(msg: str) -> RunResult:
        return RunResult(
            config=run,
            cd_value=None,
            cl_value=None,
            cd_series=[],
            cl_series=[],
            status="error",
            error=msg,
        )

    kwargs = dict(
        job_id=f"datagen_{run.run_id}",
        wind_speed=run.wind_speed,
        viz_mode=run.viz_mode,
        collision_mode=run.collision_mode,
        duration=run.duration,
        reynolds=run.reynolds,
        grid=run.grid,
    )
    if run.obj_path:
        import base64

        kwargs["model"] = "custom"
        kwargs["obj_data"] = base64.b64encode(
            Path(run.obj_path).read_bytes()
        ).decode()
    else:
        kwargs["model"] = run.model

    try:
        call = render_fn.spawn(**kwargs)
        data = call.get(timeout=timeout)
    except TimeoutError:
        return _error(f"No result after {timeout}s")
    except Exception as e:
        return _error(str(e))

    if data.get("status") != "complete":
        return _error(data.get("error", "unknown"))

    return RunResult(
        config=run,
        cd_value=None,  # filled in by check_quality from the series tail
        cl_value=None,
        cd_series=data.get("cd_series", []),
        cl_series=data.get("cl_series", []),
        status="complete",
        effective_re=data.get("effective_re"),
        flow_throughs=data.get("flow_throughs"),
        cfl=data.get("cfl"),
    )


def check_quality(result: RunResult, quality_config: dict) -> RunResult:
    """Compute tail-averaged Cd/Cl and flag bad data points."""
    if result.status != "complete":
        return result

    if len(result.cd_series) < quality_config["min_cd_samples"]:
        result.status = "flagged"
        result.error = f"Only {len(result.cd_series)} Cd samples"
        return result

    # The worker's own cd_value is a last-5-sample mean; recompute over
    # the last quarter of the series to match the convergence protocol.
    fraction = quality_config.get("tail_fraction", 0.25)
    cd_stats = tail_stats(result.cd_series, fraction)
    cl_stats = tail_stats(result.cl_series, fraction)
    result.cd_value, _, result.cd_relstd = cd_stats
    if cl_stats:
        result.cl_value = cl_stats[0]

    cd = result.cd_value
    if cd < quality_config["cd_min"] or cd > quality_config["cd_max"]:
        result.status = "flagged"
        result.error = f"Cd={cd:.4f} outside valid range"
        return result

    # A requested Re above the grid's cap clamps to the cap, so the run
    # duplicates another parameter point under a different label.
    req = result.config.reynolds
    eff = result.effective_re
    if req > 0 and eff is not None and abs(eff - req) > 0.01 * req:
        result.status = "flagged"
        result.error = f"Re clamped: requested {req:g}, effective {eff:g}"
        return result

    threshold = quality_config["convergence_threshold"]
    result.converged = result.cd_relstd < threshold

    return result


def save_manifest_row(result: RunResult):
    """Append one row to the manifest CSV."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    with _csv_lock:
        write_header = not MANIFEST_FILE.exists()

        with open(MANIFEST_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "run_id",
                        "model",
                        "wind_speed",
                        "reynolds",
                        "grid",
                        "duration",
                        "status",
                        "cd_value",
                        "cd_relstd",
                        "cl_value",
                        "effective_re",
                        "flow_throughs",
                        "cfl",
                        "converged",
                        "error",
                    ]
                )
            writer.writerow(
                [
                    result.config.run_id,
                    result.config.model,
                    result.config.wind_speed,
                    result.config.reynolds,
                    result.config.grid,
                    result.config.duration,
                    result.status,
                    f"{result.cd_value:.6f}" if result.cd_value is not None else "",
                    f"{result.cd_relstd:.4f}" if result.cd_relstd is not None else "",
                    f"{result.cl_value:.6f}" if result.cl_value is not None else "",
                    result.effective_re if result.effective_re is not None else "",
                    result.flow_throughs if result.flow_throughs is not None else "",
                    result.cfl if result.cfl is not None else "",
                    result.converged,
                    result.error or "",
                ]
            )


def save_results_csv(result: RunResult):
    """Append the full time series to the results file."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    with _csv_lock:
        write_header = not RESULTS_FILE.exists()

        with open(RESULTS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "run_id",
                        "model",
                        "wind_speed",
                        "reynolds",
                        "cd_value",
                        "cl_value",
                        "cd_series",
                        "cl_series",
                    ]
                )
            writer.writerow(
                [
                    result.config.run_id,
                    result.config.model,
                    result.config.wind_speed,
                    result.config.reynolds,
                    f"{result.cd_value:.6f}" if result.cd_value is not None else "",
                    f"{result.cl_value:.6f}" if result.cl_value is not None else "",
                    json.dumps(result.cd_series),
                    json.dumps(result.cl_series),
                ]
            )


def export_binary_dataset():
    """
    Convert the results CSV into a binary file for the C++ training framework.

    Format: header (16 bytes) + N records.

    Header:
        uint32 magic (0x4C415454 = "LATT")
        uint32 version (1)
        uint32 num_records
        uint32 features_per_record

    Each record:
        float32 wind_speed
        float32 reynolds
        float32 model_id (0=car, 1=ahmed25, 2=ahmed35)
        float32 cd_value
        float32 cl_value
    """
    if not RESULTS_FILE.exists():
        print("No results file found, skipping binary export.")
        return

    model_ids = {"car": 0.0, "ahmed25": 1.0, "ahmed35": 2.0}
    records = []

    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["cd_value"] or not row["cl_value"]:
                continue
            # The legacy binary only encodes the three baked-in models by id;
            # custom meshes have no id, so leave them out (they feed the
            # descriptor-based any-OBJ trainer via results.csv instead).
            if row["model"] not in model_ids:
                continue
            records.append(
                (
                    float(row["wind_speed"]),
                    float(row["reynolds"]),
                    model_ids[row["model"]],
                    float(row["cd_value"]),
                    float(row["cl_value"]),
                )
            )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(BINARY_FILE, "wb") as f:
        # Header
        f.write(struct.pack("<I", 0x4C415454))  # magic
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", len(records)))  # num_records
        f.write(struct.pack("<I", 5))  # features_per_record

        # Records
        for rec in records:
            f.write(struct.pack("<5f", *rec))

    print(f"Exported {len(records)} records to {BINARY_FILE}")


def compute_stats():
    """Generate summary statistics and save as JSON."""
    if not MANIFEST_FILE.exists():
        return

    stats = {
        "total": 0,
        "complete": 0,
        "failed": 0,
        "flagged": 0,
        "converged": 0,
        "by_model": {},
    }

    with open(MANIFEST_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total"] += 1
            status = row["status"]
            if status == "complete":
                stats["complete"] += 1
            elif status == "error":
                stats["failed"] += 1
            elif status == "flagged":
                stats["flagged"] += 1

            if row.get("converged") == "True":
                stats["converged"] += 1

            model = row["model"]
            if model not in stats["by_model"]:
                stats["by_model"][model] = {"count": 0, "cd_values": []}
            stats["by_model"][model]["count"] += 1
            if row["cd_value"]:
                stats["by_model"][model]["cd_values"].append(
                    float(row["cd_value"])
                )

    # Compute per-model stats
    for model_stats in stats["by_model"].values():
        cd_vals = model_stats.pop("cd_values")
        if cd_vals:
            model_stats["cd_mean"] = sum(cd_vals) / len(cd_vals)
            model_stats["cd_min"] = min(cd_vals)
            model_stats["cd_max"] = max(cd_vals)
            mean = model_stats["cd_mean"]
            model_stats["cd_std"] = (
                sum((x - mean) ** 2 for x in cd_vals) / len(cd_vals)
            ) ** 0.5

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def get_render_function():
    """Look up the deployed render function on Modal."""
    return modal.Function.from_name(MODAL_APP, MODAL_FUNCTION)


def run_sweep(config: dict, resume: bool = False, limit: int = 0):
    """Run the full parameter sweep."""
    runs = generate_run_configs(config)
    completed = load_completed_runs() if resume else set()
    quality = config.get(
        "quality",
        {
            "cd_max": 15.0,
            "cd_min": 0.0,
            "min_cd_samples": 20,
            "tail_fraction": 0.25,
            "convergence_threshold": 0.05,
        },
    )

    pending = [r for r in runs if r.run_id not in completed]
    if limit:
        pending = pending[:limit]
    done = len(completed)
    todo = len(pending)
    print(f"Total: {len(runs)}, done: {done}, pending: {todo}")

    if not pending:
        print("Nothing to do.")
        return

    render_fn = get_render_function()
    max_workers = config.get("max_concurrent", 4)
    retry_limit = config.get("retry_limit", 3)
    delay = config.get("delay_between_batches", 2.0)

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for run in pending:
            fut = pool.submit(submit_run, run, render_fn)
            futures[fut] = (run, 0)  # (config, attempt)

        while futures:
            done_batch = []
            for fut in as_completed(futures):
                run_cfg, attempt = futures[fut]
                result = fut.result()
                result = check_quality(result, quality)

                if result.status == "error" and attempt < retry_limit:
                    print(f"  Retry {run_cfg.run_id} (attempt {attempt + 1})")
                    time.sleep(delay)
                    new_fut = pool.submit(submit_run, run_cfg, render_fn)
                    futures[new_fut] = (run_cfg, attempt + 1)
                else:
                    save_manifest_row(result)
                    if result.status == "complete":
                        save_results_csv(result)
                        success += 1
                        conv = (
                            "converged" if result.converged else "not converged"
                        )
                        print(
                            f"  [{success + failed}/{len(pending)}] "
                            f"{run_cfg.model} ws={run_cfg.wind_speed} "
                            f"re={run_cfg.reynolds:g} "
                            f"-> Cd={result.cd_value:.4f} "
                            f"relStd={result.cd_relstd * 100:.1f}% ({conv})"
                        )
                    else:
                        failed += 1
                        print(
                            f"  [{success + failed}/{len(pending)}] "
                            f"{run_cfg.model} ws={run_cfg.wind_speed} "
                            f"re={run_cfg.reynolds:g} "
                            f"-> {result.status.upper()}: {result.error}"
                        )

                done_batch.append(fut)

            for fut in done_batch:
                del futures[fut]

    print(f"\nDone: {success} complete, {failed} failed")
    export_binary_dataset()
    stats = compute_stats()
    if stats:
        print(f"Stats: {json.dumps(stats, indent=2)}")


def run_pilot(config: dict):
    """Run the convergence study's anchor point end to end and compare
    against the reference Cd. Validates the whole pipeline (Modal call,
    series collection, tail averaging) for the cost of one run."""
    run = RunConfig(
        model=ANCHOR_MODEL,
        wind_speed=ANCHOR_WIND,
        reynolds=ANCHOR_RE,
        duration=config["duration"],
        viz_mode=config.get("viz_mode", 1),
        collision_mode=config.get("collision_mode", 2),
        grid=config.get("grid", "256x128x128"),
    )
    quality = config["quality"]

    print(
        f"Pilot: {run.model} ws={run.wind_speed} re={run.reynolds:g} "
        f"grid={run.grid} {run.duration}s"
    )
    print(f"Expecting Cd = {ANCHOR_CD} +/- {ANCHOR_TOL} (#155 reference)")

    result = submit_run(run, get_render_function())
    result = check_quality(result, quality)

    if result.status != "complete":
        print(f"FAILED: {result.error}")
        sys.exit(1)

    save_manifest_row(result)
    save_results_csv(result)

    diff = abs(result.cd_value - ANCHOR_CD)
    verdict = "OK" if diff <= ANCHOR_TOL else "MISMATCH"
    print(
        f"Cd={result.cd_value:.4f} (ref {ANCHOR_CD}, diff {diff:.4f}) "
        f"relStd={result.cd_relstd * 100:.2f}% "
        f"Re_eff={result.effective_re} -> {verdict}"
    )
    if verdict == "MISMATCH":
        sys.exit(1)


def show_status():
    """Show current progress of data generation."""
    stats = compute_stats()
    if not stats:
        print("No data generated yet.")
        return

    print(f"Total runs: {stats['total']}")
    print(f"  Complete: {stats['complete']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Flagged: {stats['flagged']}")
    print(f"  Converged: {stats['converged']}")
    print()
    for model, ms in stats["by_model"].items():
        print(f"  {model}: {ms['count']} runs", end="")
        if "cd_mean" in ms:
            print(f", Cd mean={ms['cd_mean']:.4f} std={ms['cd_std']:.4f}")
        else:
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for Cd surrogate model"
    )
    parser.add_argument(
        "--config", default="sweep_config.yaml", help="Sweep config YAML"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip already-completed runs"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run at most N pending runs (0 = all)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run the #155 anchor point and check Cd against the reference",
    )
    parser.add_argument("--status", action="store_true", help="Show progress")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Just re-export binary from CSV",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Error: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    config = load_sweep_config(str(config_path))

    if config.get("dataset_dir"):
        set_dataset_dir(Path(__file__).parent / config["dataset_dir"])

    if args.status:
        show_status()
        return

    if args.export_only:
        export_binary_dataset()
        compute_stats()
        return

    if args.pilot:
        run_pilot(config)
        return

    run_sweep(config, resume=args.resume, limit=args.limit)


if __name__ == "__main__":
    main()
