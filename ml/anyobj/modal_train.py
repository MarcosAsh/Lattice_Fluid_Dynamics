"""
Fetch datasets and train the any-mesh Cd/Cl surrogate on Modal.

Meshes and labels live in the persistent anyobj-data volume, fetched straight
from HuggingFace inside Modal so big datasets never touch the local machine.
The descriptor cache also persists there, so re-training after the first run
skips encoding entirely. Trained weights are written back next to this file.

Usage:
    modal run ml/anyobj/modal_train.py --fetch ahmedml,windsorml,drivaerml
    modal run ml/anyobj/modal_train.py --epochs 1000 --hidden 128

One-time setup: pip install modal && modal setup
"""

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REMOTE_CODE = "/root/anyobj"
DATA = "/data"

app = modal.App("anyobj-train")
vol = modal.Volume.from_name("anyobj-data", create_if_missing=True)

IGNORE = ["**/__pycache__", "*.npz", ".desc_cache", ".venv", "meshes"]
deps = modal.Image.debian_slim("3.12").pip_install(
    "numpy>=2.5.0", "scipy>=1.17", "trimesh>=4.12", "rtree>=1.4.1",
    "embreex", "jax>=0.4.30", "optax>=0.2")
image = deps.add_local_dir(HERE, REMOTE_CODE, ignore=IGNORE)
gpu_image = deps.pip_install("jax[cuda12]").add_local_dir(
    HERE, REMOTE_CODE, ignore=IGNORE)


def _bootstrap():
    import os
    import sys
    os.environ["ANYOBJ_MESHES"] = f"{DATA}/meshes"
    os.environ["ANYOBJ_OUT"] = f"{DATA}/out"
    sys.path.insert(0, REMOTE_CODE)


@app.function(image=image, cpu=8, memory=32768, timeout=14400,
              volumes={DATA: vol})
def fetch_remote(datasets, workers=8):
    _bootstrap()
    import fetch
    for name in datasets:
        fetch.run_dataset(name, mesh_dir=Path(f"{DATA}/meshes"),
                          data_root=Path(f"{DATA}/dataset"), workers=workers)
    fetch.combine(Path(f"{DATA}/dataset"))
    vol.commit()


@app.function(image=image, cpu=16, memory=65536, timeout=7200,
              volumes={DATA: vol})
def train_remote(epochs, batch_size, hidden, lr, patience, voxel_res, workers):
    import argparse
    _bootstrap()
    import train as anyobj_train

    anyobj_train.train(argparse.Namespace(
        data=f"{DATA}/dataset/all/results.csv", epochs=epochs,
        batch_size=batch_size, hidden=hidden, lr=lr, patience=patience,
        voxel_res=voxel_res, workers=workers))

    vol.commit()  # persist the descriptor cache
    return {p.name: p.read_bytes() for p in Path(f"{DATA}/out").glob("*.npz")}


@app.function(image=image, cpu=16, memory=65536, timeout=14400,
              volumes={DATA: vol})
def ingest_shift(token, configs, limit, voxel_res, workers):
    _bootstrap()
    import fetch
    import shift
    for key in [c.strip() for c in configs.split(",") if c.strip()]:
        shift.ingest(key, DATA, token, limit, workers, voxel_res)
    shift.write_results(DATA)
    fetch.combine(Path(f"{DATA}/dataset"))
    vol.commit()


@app.function(image=image, cpu=16, memory=32768, timeout=7200,
              volumes={DATA: vol})
def logo_remote(epochs, hidden, lr, voxel_res):
    import argparse
    import contextlib
    import io as _io
    _bootstrap()
    import evaluate
    from train import load_dataset

    X, Y, names = load_dataset(f"{DATA}/dataset/all/results.csv", voxel_res)
    args = argparse.Namespace(hidden=hidden, epochs=epochs, lr=lr, plot=False)
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluate.run_logo(X, Y, names, args)
    return buf.getvalue()


@app.function(image=gpu_image, gpu="T4", cpu=8, memory=32768, timeout=7200,
              volumes={DATA: vol})
def train_cnn_remote(epochs, batch_size, width, lr, patience, voxel_res,
                     workers):
    import argparse
    _bootstrap()
    import train_cnn

    train_cnn.train(argparse.Namespace(
        data=f"{DATA}/dataset/all/results.csv", epochs=epochs,
        batch_size=batch_size, width=width, lr=lr, patience=patience,
        voxel_res=voxel_res, workers=workers))

    vol.commit()
    return {p.name: p.read_bytes()
            for p in Path(f"{DATA}/out").glob("anyobj_cnn*.npz")}


@app.local_entrypoint()
def main(fetch: str = "", shift: str = "", shift_limit: int = 0,
         hf_token: str = "", cnn: bool = False, train: bool = True,
         logo: bool = False, epochs: int = 0, batch_size: int = 0,
         hidden: int = 64, width: int = 16, lr: float = 1e-3,
         patience: int = 0, voxel_res: int = 32, workers: int = 12):
    if fetch:
        fetch_remote.remote([d.strip() for d in fetch.split(",")], 8)
    if shift:
        import os
        token = hf_token or os.environ.get("HF_TOKEN", "")
        if not token:
            raise SystemExit("--shift needs --hf-token or HF_TOKEN env")
        ingest_shift.remote(token, shift, shift_limit, voxel_res, 8)
    if logo:
        print(logo_remote.remote(epochs or 250, hidden, lr, voxel_res))
        return
    if not train:
        return
    if cnn:
        blobs = train_cnn_remote.remote(epochs or 200, batch_size or 32,
                                        width, lr, patience or 40,
                                        voxel_res, workers)
    else:
        blobs = train_remote.remote(epochs or 500, batch_size or 64, hidden,
                                    lr, patience or 80, voxel_res, workers)
    for name, blob in sorted(blobs.items()):
        (HERE / name).write_bytes(blob)
        print(f"saved {HERE / name} ({len(blob)} bytes)")
