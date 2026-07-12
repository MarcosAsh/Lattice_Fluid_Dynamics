"""
Stream the SHIFT-SUV dataset (CC-BY-NC, gated) into training caches.

Per run: download the watertight STL plus DRAG.csv/LIFT.csv force histories,
window-average the forces, recompute Cd/Cl against the geometry's own
frontal area (the dataset's reference-area convention is unpublished, so
recomputing from raw forces matches the varref convention of the other
datasets), save the descriptor+voxel cache, delete the 252 MB STL.

Needs an HF read token with access granted on the dataset page.
Runs are resumable: already ingested runs are skipped via rows_<key>.csv.
"""

import csv
import io
import json
import multiprocessing
import os
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from geometry import N_DESCRIPTORS
from geometry import PADDING
from geometry import encode

HF_API = "https://huggingface.co/api/datasets"
HF_RES = "https://huggingface.co/datasets"

RHO = 1.225      # kg/m3
NU = 1.461e-5    # m2/s standard air

# key -> repo, run path template
CONFIGS = {
    "fse": ("luminary-shift/SUV",
            "AeroSUV_full_scale_estate_transient/run_{i:05d}"),
    "fsf": ("luminary-shift/SUV",
            "AeroSUV_full_scale_fastback_transient/run_{i:05d}"),
    "qse": ("luminary-shift/SUV",
            "AeroSUV_qrtr_scale_estate_transient/run_{i:05d}"),
    "qsf": ("luminary-shift/SUV",
            "AeroSUV_qrtr_scale_fastback_transient/run_{i:05d}"),
    "sample": ("luminary-shift/SUV-sample", "RUN_{i}"),
}


def _flow(length):
    """U and force averaging window from scale, full ~4.62 m qrtr ~1.155 m"""
    return (30.0, 15000) if length > 2.5 else (50.0, 6000)


def _get(url, token, tries=3):
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"})
    for t in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if t == tries - 1:
                raise
        except Exception:
            if t == tries - 1:
                raise


def _mean_force(text, window):
    rows = list(csv.reader(io.StringIO(text)))
    vals = [float(r[-1]) for r in rows[1:] if r]
    return float(np.mean(vals[-window:]))


def list_runs(key, token):
    repo, tmpl = CONFIGS[key]
    subdir = tmpl.split("/")[0] if "/" in tmpl else ""
    url = f"{HF_API}/{repo}/tree/main" + (f"/{subdir}" if subdir else "")
    entries = json.loads(_get(url, token).decode())
    idx = []
    for e in entries:
        tail = e["path"].rsplit("_", 1)[-1]
        if e["type"] == "directory" and tail.isdigit():
            idx.append(int(tail))
    return sorted(idx)


def _ingest_one(job):
    key, i, cache_dir, voxel_res, token = job
    import trimesh
    repo, tmpl = CONFIGS[key]
    base = f"{HF_RES}/{repo}/resolve/main/{tmpl.format(i=i)}"

    drag = _get(f"{base}/DRAG.csv", token)
    lift = _get(f"{base}/LIFT.csv", token)
    # older sample release has no filled variant
    stl = (_get(f"{base}/merged_surfaces_filled.stl", token)
           or _get(f"{base}/merged_surfaces.stl", token))
    if drag is None or lift is None or stl is None:
        return None

    tmp = Path(f"/tmp/shift_{key}_{i}.stl")
    tmp.write_bytes(stl)
    try:
        mesh = trimesh.load(str(tmp))
        length = float(mesh.extents.max())
        enc = encode(mesh, voxel_res)
    finally:
        tmp.unlink(missing_ok=True)

    U, window = _flow(length)
    f_drag = _mean_force(drag.decode(), window)
    f_lift = _mean_force(lift.decode(), window)

    # frontal area in real units from the canonical voxel grid
    vox = enc["voxels"]
    res = vox.shape[0]
    canon = int(vox.any(axis=0).sum()) * (2.0 / res) ** 2
    scale = (2.0 - 2.0 * PADDING) / length
    area = canon / scale ** 2

    q = 0.5 * RHO * U * U * area
    model = f"shiftsuv_{key}_{i:05d}"
    np.savez_compressed(
        Path(cache_dir) / f"{model}.r{voxel_res}d{N_DESCRIPTORS}.npz",
        descriptors=enc["descriptors"], voxels=vox)
    return [model, f"{length * U / NU:.6g}",
            f"{f_drag / q:.6g}", f"{f_lift / q:.6g}"]


def ingest(key, data_root, token, limit=0, workers=8, voxel_res=32):
    data_root = Path(data_root)
    cache_dir = data_root / "out" / ".desc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows_path = data_root / "dataset" / "shiftsuv" / f"rows_{key}.csv"
    rows_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if rows_path.exists():
        with open(rows_path) as f:
            done = {r[0] for r in csv.reader(f) if r}

    idx = list_runs(key, token)
    if limit:
        idx = idx[:limit]
    todo = [i for i in idx
            if f"shiftsuv_{key}_{i:05d}" not in done]
    print(f"{key}: {len(idx)} runs, {len(todo)} to ingest")
    if not todo:
        return

    jobs = [(key, i, str(cache_dir), voxel_res, token) for i in todo]
    workers = min(workers, os.cpu_count() or 1, len(jobs))
    ctx = multiprocessing.get_context("spawn")
    n = 0
    with open(rows_path, "a", newline="") as f, \
            ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1,
                                mp_context=ctx) as pool:
        w = csv.writer(f)
        for row in pool.map(_ingest_one, jobs):
            if row:
                w.writerow(row)
                f.flush()
                n += 1
                if n % 25 == 0:
                    print(f"  {key}: {n}/{len(todo)}")
    print(f"{key}: ingested {n}")


def write_results(data_root):
    root = Path(data_root) / "dataset" / "shiftsuv"
    rows = []
    for f in sorted(root.glob("rows_*.csv")):
        with open(f) as fh:
            rows.extend(r for r in csv.reader(fh) if r)
    with open(root / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "reynolds", "cd_value", "cl_value"])
        w.writerows(rows)
    print(f"shiftsuv: {len(rows)} rows -> {root / 'results.csv'}")
