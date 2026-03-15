"""
Training data generation pipeline for the Cd surrogate model.

Submits parameter sweeps to the Modal render endpoint, collects results,
and saves them as CSV + binary dataset files.

Usage:
    python data_gen.py --config sweep_config.yaml
    python data_gen.py --config sweep_config.yaml --resume  # skip completed runs
    python data_gen.py --status                              # show progress
"""

import argparse
import csv
import hashlib
import json
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
import yaml

DATASET_DIR = Path(__file__).parent / "dataset"
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

    @property
    def run_id(self) -> str:
        key = f"{self.model}_{self.wind_speed}_{self.reynolds}"
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


def load_sweep_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_run_configs(config: dict) -> list[RunConfig]:
    """Build the full list of parameter combinations."""
    runs = []
    ws = config["wind_speeds"]
    wind_speeds = []
    v = ws["min"]
    while v <= ws["max"] + 1e-9:
        wind_speeds.append(round(v, 2))
        v += ws["step"]

    for model in config["models"]:
        for wind_speed in wind_speeds:
            for reynolds in config["reynolds_numbers"]:
                runs.append(RunConfig(
                    model=model,
                    wind_speed=wind_speed,
                    reynolds=float(reynolds),
                    duration=config["duration"],
                    viz_mode=config.get("viz_mode", 1),
                    collision_mode=config.get("collision_mode", 1),
                ))
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


def submit_run(run: RunConfig, endpoint: str, timeout: int = 600) -> RunResult:
    """Submit a single simulation run to Modal."""
    payload = {
        "job_id": f"datagen_{run.run_id}",
        "wind_speed": run.wind_speed,
        "viz_mode": run.viz_mode,
        "collision_mode": run.collision_mode,
        "duration": run.duration,
        "model": run.model,
        "reynolds": run.reynolds,
    }

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "complete":
            return RunResult(
                config=run,
                cd_value=data.get("cd_value"),
                cl_value=data.get("cl_value"),
                cd_series=data.get("cd_series", []),
                cl_series=data.get("cl_series", []),
                status="complete",
            )
        else:
            return RunResult(
                config=run,
                cd_value=None,
                cl_value=None,
                cd_series=[],
                cl_series=[],
                status="error",
                error=data.get("error", "unknown"),
            )
    except Exception as e:
        return RunResult(
            config=run,
            cd_value=None,
            cl_value=None,
            cd_series=[],
            cl_series=[],
            status="error",
            error=str(e),
        )


def check_quality(result: RunResult, quality_config: dict) -> RunResult:
    """Flag bad data points."""
    if result.status != "complete" or result.cd_value is None:
        return result

    cd = result.cd_value
    if cd < quality_config["cd_min"] or cd > quality_config["cd_max"]:
        result.status = "flagged"
        result.error = f"Cd={cd:.4f} outside valid range"
        return result

    if len(result.cd_series) < quality_config["min_cd_samples"]:
        result.status = "flagged"
        result.error = f"Only {len(result.cd_series)} Cd samples"
        return result

    # Check convergence: std dev of last N values relative to mean
    window = quality_config["convergence_window"]
    threshold = quality_config["convergence_threshold"]
    if len(result.cd_series) >= window:
        tail = result.cd_series[-window:]
        mean = sum(tail) / len(tail)
        if mean > 1e-6:
            std = (sum((x - mean) ** 2 for x in tail) / len(tail)) ** 0.5
            result.converged = (std / mean) < threshold

    return result


def save_manifest_row(result: RunResult):
    """Append one row to the manifest CSV."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not MANIFEST_FILE.exists()

    with open(MANIFEST_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_id", "model", "wind_speed", "reynolds", "duration",
                "status", "cd_value", "cl_value", "converged", "error",
            ])
        writer.writerow([
            result.config.run_id,
            result.config.model,
            result.config.wind_speed,
            result.config.reynolds,
            result.config.duration,
            result.status,
            f"{result.cd_value:.6f}" if result.cd_value is not None else "",
            f"{result.cl_value:.6f}" if result.cl_value is not None else "",
            result.converged,
            result.error or "",
        ])


def save_results_csv(result: RunResult):
    """Append the full time series to the results file."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_FILE.exists()

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_id", "model", "wind_speed", "reynolds",
                "cd_value", "cl_value", "cd_series", "cl_series",
            ])
        writer.writerow([
            result.config.run_id,
            result.config.model,
            result.config.wind_speed,
            result.config.reynolds,
            f"{result.cd_value:.6f}" if result.cd_value is not None else "",
            f"{result.cl_value:.6f}" if result.cl_value is not None else "",
            json.dumps(result.cd_series),
            json.dumps(result.cl_series),
        ])


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
            records.append((
                float(row["wind_speed"]),
                float(row["reynolds"]),
                model_ids.get(row["model"], -1.0),
                float(row["cd_value"]),
                float(row["cl_value"]),
            ))

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(BINARY_FILE, "wb") as f:
        # Header
        f.write(struct.pack("<I", 0x4C415454))  # magic
        f.write(struct.pack("<I", 1))             # version
        f.write(struct.pack("<I", len(records)))   # num_records
        f.write(struct.pack("<I", 5))              # features_per_record

        # Records
        for rec in records:
            f.write(struct.pack("<5f", *rec))

    print(f"Exported {len(records)} records to {BINARY_FILE}")


def compute_stats():
    """Generate summary statistics and save as JSON."""
    if not MANIFEST_FILE.exists():
        return

    stats = {
        "total": 0, "complete": 0, "failed": 0, "flagged": 0,
        "converged": 0, "by_model": {},
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
                stats["by_model"][model]["cd_values"].append(float(row["cd_value"]))

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


def run_sweep(config_path: str, endpoint: str, resume: bool = False):
    """Run the full parameter sweep."""
    config = load_sweep_config(config_path)
    runs = generate_run_configs(config)
    completed = load_completed_runs() if resume else set()
    quality = config.get("quality", {
        "cd_max": 10.0, "cd_min": 0.0, "min_cd_samples": 5,
        "convergence_window": 5, "convergence_threshold": 0.05,
    })

    pending = [r for r in runs if r.run_id not in completed]
    print(f"Total runs: {len(runs)}, already done: {len(completed)}, pending: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    max_workers = config.get("max_concurrent", 4)
    retry_limit = config.get("retry_limit", 3)
    delay = config.get("delay_between_batches", 2.0)

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for run in pending:
            fut = pool.submit(submit_run, run, endpoint)
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
                    new_fut = pool.submit(submit_run, run_cfg, endpoint)
                    futures[new_fut] = (run_cfg, attempt + 1)
                else:
                    save_manifest_row(result)
                    if result.status == "complete":
                        save_results_csv(result)
                        success += 1
                        conv = "converged" if result.converged else "not converged"
                        print(
                            f"  [{success + failed}/{len(pending)}] "
                            f"{run_cfg.model} ws={run_cfg.wind_speed} re={run_cfg.reynolds} "
                            f"-> Cd={result.cd_value:.4f} ({conv})"
                        )
                    else:
                        failed += 1
                        print(
                            f"  [{success + failed}/{len(pending)}] "
                            f"{run_cfg.model} ws={run_cfg.wind_speed} re={run_cfg.reynolds} "
                            f"-> FAILED: {result.error}"
                        )

                done_batch.append(fut)

            for fut in done_batch:
                del futures[fut]

    print(f"\nDone: {success} complete, {failed} failed")
    export_binary_dataset()
    stats = compute_stats()
    if stats:
        print(f"Stats: {json.dumps(stats, indent=2)}")


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
    parser = argparse.ArgumentParser(description="Generate training data for Cd surrogate model")
    parser.add_argument("--config", default="sweep_config.yaml", help="Sweep config YAML")
    parser.add_argument("--endpoint", default=None, help="Modal render endpoint URL")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed runs")
    parser.add_argument("--status", action="store_true", help="Show progress")
    parser.add_argument("--export-only", action="store_true", help="Just re-export binary from CSV")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.export_only:
        export_binary_dataset()
        compute_stats()
        return

    endpoint = args.endpoint
    if not endpoint:
        import os
        endpoint = os.environ.get("MODAL_RENDER_ENDPOINT")
    if not endpoint:
        print("Error: provide --endpoint or set MODAL_RENDER_ENDPOINT", file=sys.stderr)
        sys.exit(1)

    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Error: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    run_sweep(str(config_path), endpoint, resume=args.resume)


if __name__ == "__main__":
    main()
