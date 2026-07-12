"""
Fetch public CFD datasets into the anyobj training layout.

Downloads only STL geometry and force coefficient CSVs, never the TB-scale
flow fields. Meshes go to meshes/<dataset>_<run>.stl, labels to
ml/dataset/<dataset>/results.csv with columns model,reynolds,cd_value,cl_value.
All labels use each dataset's per-geometry reference-area coefficients.

Usage:
    python fetch.py ahmedml windsorml
    python fetch.py drivaerml --workers 4     # 135 MB per STL, ~67 GB
    python fetch.py combine                   # merge into all/results.csv
"""

import argparse
import csv
import io
import multiprocessing
import os
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HF = "https://huggingface.co/datasets"
DATA_ROOT = Path(__file__).resolve().parents[1] / "dataset"
MESH_DIR = Path(__file__).resolve().parent / "meshes"

# re_per_m = U/nu so reynolds = longest mesh extent * re_per_m
SPECS = {
    "ahmedml": {
        "repo": "neashton/ahmedml",
        "stl": "run_{i}/ahmed_{i}.stl",
        "labels_all": "force_mom_varref_all.csv",
        "runs": range(1, 501),
        "re_per_m": 1.0 / 3.75e-7,
    },
    "windsorml": {
        "repo": "neashton/windsorml",
        "stl": "run_{i}/windsor_{i}.stl",
        "labels_run": "run_{i}/force_mom_varref_{i}.csv",
        "runs": range(1, 356),
        "re_per_m": 40.0 / 1.44e-5,
    },
    "drivaerml": {
        "repo": "neashton/drivaerml",
        "stl": "run_{i}/drivaer_{i}.stl",
        "labels_run": "run_{i}/force_mom_{i}.csv",
        "runs": range(1, 501),
        "re_per_m": 38.889 / 1.507e-5,
    },
}


def get(url, tries=3):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if t == tries - 1:
                raise
        except Exception:
            if t == tries - 1:
                raise


def fetch_file(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    data = get(url)
    if data is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.rename(dest)
    return True


def mesh_length(path):
    import trimesh
    return float(trimesh.load(str(path)).extents.max())


def _length_one(job):
    i, path = job
    return i, mesh_length(path)


def mesh_lengths(jobs, workers):
    """lengths in parallel, one process per mesh so big STLs cant pile up"""
    workers = min(workers, os.cpu_count() or 1, max(len(jobs), 1))
    ctx = multiprocessing.get_context("spawn")
    out = {}
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1,
                             mp_context=ctx) as pool:
        for i, length in pool.map(_length_one, jobs):
            out[i] = length
    return out


def parse_cd_cl(text):
    """last data row of a force CSV, columns matched case insensitively"""
    rows = list(csv.DictReader(io.StringIO(text)))
    row = {k.strip().lower(): v for k, v in rows[-1].items()}
    return float(row["cd"]), float(row["cl"])


def run_dataset(name, mesh_dir=MESH_DIR, data_root=DATA_ROOT,
                workers=8, stl_only=False, runs=None):
    spec = SPECS[name]
    base = f"{HF}/{spec['repo']}/resolve/main"
    runs = list(runs or spec["runs"])

    def grab(i):
        dest = mesh_dir / f"{name}_{i}.stl"
        ok = fetch_file(f"{base}/{spec['stl'].format(i=i)}", dest)
        return i if ok else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        got = [i for i in pool.map(grab, runs) if i is not None]
    print(f"{name}: {len(got)}/{len(runs)} meshes in {mesh_dir}")
    if stl_only:
        return

    labels = {}
    if "labels_all" in spec:
        text = get(f"{base}/{spec['labels_all']}").decode()
        for row in csv.DictReader(io.StringIO(text)):
            row = {k.strip().lower(): v.strip() for k, v in row.items()}
            labels[int(row["run"])] = (float(row["cd"]), float(row["cl"]))
    else:
        raw = data_root / name / "raw"

        def grab_label(i):
            dest = raw / f"force_{i}.csv"
            if fetch_file(f"{base}/{spec['labels_run'].format(i=i)}", dest):
                return i, parse_cd_cl(dest.read_text())
            return None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for hit in pool.map(grab_label, got):
                if hit:
                    labels[hit[0]] = hit[1]

    keep = sorted(set(got) & set(labels))
    lengths = mesh_lengths(
        [(i, str(mesh_dir / f"{name}_{i}.stl")) for i in keep], workers)
    out = data_root / name / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "reynolds", "cd_value", "cl_value"])
        for i in keep:
            re = lengths[i] * spec["re_per_m"]
            w.writerow([f"{name}_{i}", f"{re:.6g}", labels[i][0], labels[i][1]])
    print(f"{name}: {len(keep)} labeled rows -> {out}")


def combine(data_root=DATA_ROOT):
    rows = []
    for f in sorted(data_root.glob("*/results.csv")):
        if f.parent.name == "all":
            continue
        with open(f) as fh:
            rows.extend(list(csv.DictReader(fh)))
    out = data_root / "all" / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["model", "reynolds", "cd_value", "cl_value"])
        w.writeheader()
        w.writerows(rows)
    print(f"combined: {len(rows)} rows -> {out}")


def main():
    p = argparse.ArgumentParser(description="Fetch CFD datasets")
    p.add_argument("datasets", nargs="+",
                   help=f"{', '.join(SPECS)} or combine")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stl-only", action="store_true")
    args = p.parse_args()
    for name in args.datasets:
        if name == "combine":
            combine()
        else:
            run_dataset(name, workers=args.workers, stl_only=args.stl_only)


if __name__ == "__main__":
    main()
