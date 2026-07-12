"""
Train the any-mesh Cd/Cl surrogate (Phase 1: descriptor MLP), in JAX.

Each training row becomes [geometry descriptors, reynolds] -> [Cd, Cl], so the
model reads geometry directly instead of a model-id label. Reynolds is the only
flow input: Cd at fixed Re is independent of the lattice velocity, so wind_speed
carries no signal. Descriptors come from geometry.encode; inputs and targets are
z-score normalized and the normalizer is saved next to the weights.

Meshes may be any format trimesh reads (OBJ, STL, PLY, glTF/GLB, OFF, 3MF, ...):
a dataset row's model name resolves to a bundled reference OBJ or to any
matching mesh file under meshes/.

Usage:
    python train.py                                  # dataset/all/results.csv
    python train.py --data ../dataset/ahmedml/results.csv --hidden 128
"""

import argparse
import csv
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import DESCRIPTOR_NAMES  # noqa: E402
from geometry import N_DESCRIPTORS  # noqa: E402
from geometry import encode  # noqa: E402

# Dataset model name -> mesh, mirroring simulation/modal_worker.py.
ASSETS = (Path(__file__).resolve().parents[2]
          / "simulation" / "assets" / "3d-files")
MESH_DIR = Path(os.environ.get("ANYOBJ_MESHES")
                or Path(__file__).resolve().parent / "meshes")
MODEL_OBJS = {
    "car": ASSETS / "car-model.obj",
    "ahmed25": ASSETS / "ahmed_25deg_m.obj",
    "ahmed35": ASSETS / "ahmed_35deg_m.obj",
}

# Formats geometry.load_mesh (trimesh) can ingest, in lookup order.
MESH_EXTS = (".obj", ".stl", ".ply", ".glb", ".gltf", ".off", ".3mf")


def mesh_for(model):
    """Resolve a dataset model name to its mesh: a reference model or, failing
    that, a generated/imported mesh under meshes/ in any supported format."""
    if model in MODEL_OBJS:
        return MODEL_OBJS[model]
    for ext in MESH_EXTS:
        path = MESH_DIR / f"{model}{ext}"
        if path.exists():
            return path
    return MESH_DIR / f"{model}.obj"


# log10 tames the 3e6 to 2e7 spread across datasets
FLOW_FEATURES = ["log10_reynolds"]
TARGET_NAMES = ["cd", "cl"]
N_INPUTS = N_DESCRIPTORS + len(FLOW_FEATURES)
N_TARGETS = len(TARGET_NAMES)
DEFAULT_DATA = (Path(__file__).resolve().parents[1]
                / "dataset" / "all" / "results.csv")

# ANYOBJ_OUT lets modal_train.py write outside its read-only code mount
OUT_DIR = Path(os.environ.get("ANYOBJ_OUT") or Path(__file__).resolve().parent)
CACHE_DIR = OUT_DIR / ".desc_cache"


def _encode_one(job):
    model, mesh_path, voxel_res = job
    enc = encode(mesh_path, voxel_res)
    return model, enc["descriptors"], enc["voxels"]


def encode_models(models, voxel_res, workers=None, voxels=False):
    """Descriptors (and voxels) per model, disk cached, one process per mesh"""
    out, jobs = {}, []
    for model in models:
        mesh = mesh_for(model)
        cached = CACHE_DIR / f"{model}.r{voxel_res}d{N_DESCRIPTORS}.npz"
        # streamed datasets keep only the cache, the mesh is gone
        fresh = cached.exists() and (
            not mesh.exists()
            or cached.stat().st_mtime >= mesh.stat().st_mtime)
        if fresh:
            data = np.load(cached)
            out[model] = ((data["descriptors"], data["voxels"]) if voxels
                          else data["descriptors"])
        elif not mesh.exists():
            raise FileNotFoundError(f"No mesh for model '{model}' ({mesh})")
        else:
            jobs.append((model, str(mesh), voxel_res))
    if not jobs:
        return out

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    workers = min(workers or 8, os.cpu_count() or 1, len(jobs))
    ctx = multiprocessing.get_context("spawn")
    print(f"  encoding {len(jobs)} meshes ({workers} workers)...")
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1,
                             mp_context=ctx) as pool:
        for model, vec, vox in pool.map(_encode_one, jobs):
            name = f"{model}.r{voxel_res}d{N_DESCRIPTORS}.npz"
            np.savez_compressed(CACHE_DIR / name, descriptors=vec, voxels=vox)
            out[model] = (vec, vox) if voxels else vec
    return out


def read_rows(results_csv):
    """CSV rows as (model, reynolds, cd, cl), duplicate (model, Re) collapsed"""
    rows, seen = [], set()
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if not row.get("cd_value") or not row.get("cl_value"):
                continue
            key = (row["model"], float(row["reynolds"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append((row["model"], float(row["reynolds"]),
                         float(row["cd_value"]), float(row["cl_value"])))
    if not rows:
        raise ValueError(f"No usable rows in {results_csv}")
    return rows


def load_dataset(results_csv, voxel_res, workers=None):
    """Read a data-gen results CSV into (X, Y, row_models): feature and target
    arrays plus the geometry name behind each row."""
    rows = read_rows(results_csv)
    desc = encode_models(sorted({r[0] for r in rows}), voxel_res, workers)
    feats = [np.concatenate([desc[m], [np.log10(re)]]) for m, re, _, _ in rows]
    targets = [[cd, cl] for _, _, cd, cl in rows]
    return (np.asarray(feats, np.float32),
            np.asarray(targets, np.float32), [r[0] for r in rows])


def zscore(arr):
    """Per-column (mean, std); zero-variance columns get std 1."""
    mean, std = arr.mean(0), arr.std(0)
    std[std < 1e-8] = 1.0
    return mean, std


# Model: 2 hidden ReLU layers, matching the original SurrogateNet.

def init_params(key, hidden):
    """He-initialized weights for [N_INPUTS, hidden, hidden, N_TARGETS]."""
    sizes = [N_INPUTS, hidden, hidden, N_TARGETS]
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, wkey = jax.random.split(key)
        w = jax.random.normal(wkey, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
        params.append({"w": w, "b": jnp.zeros(fan_out)})
    return params


def forward(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def mse(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)


def save_params(path, params, hidden):
    flat = {"hidden": np.array(hidden)}
    for i, layer in enumerate(params):
        flat[f"w{i}"] = np.asarray(layer["w"])
        flat[f"b{i}"] = np.asarray(layer["b"])
    np.savez(path, **flat)


def load_params(path):
    data = np.load(path)
    n_layers = sum(1 for k in data.files if k.startswith("w"))
    return [{"w": jnp.asarray(data[f"w{i}"]), "b": jnp.asarray(data[f"b{i}"])}
            for i in range(n_layers)]


def train(args):
    data_path = Path(args.data).resolve()
    X, Y, row_models = load_dataset(data_path, args.voxel_res,
                                    getattr(args, "workers", None))
    used = sorted(set(row_models))
    print(f"{data_path}: {len(X)} samples ({', '.join(used)}), "
          f"{N_INPUTS} inputs -> {TARGET_NAMES}")

    x_mean, x_std = zscore(X)
    y_mean, y_std = zscore(Y)
    Xn, Yn = (X - x_mean) / x_std, (Y - y_mean) / y_std

    n = len(Xn)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    if n >= 5:
        split = max(1, int(0.8 * n))
        train_idx, val_idx = idx[:split], idx[split:]
    else:
        print("  Few samples: validating on the training set.")
        train_idx = val_idx = idx

    batch_size = min(args.batch_size, len(train_idx))
    x_val = jnp.asarray(Xn[val_idx])
    y_val = jnp.asarray(Yn[val_idx])

    params = init_params(jax.random.PRNGKey(42), args.hidden)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=args.lr, weight_decay=1e-4),
    )
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, bx, by):
        loss, grads = jax.value_and_grad(mse)(params, bx, by)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    val_loss_fn = jax.jit(mse)

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "anyobj_model.npz"
    best_path = out_dir / "anyobj_model_best.npz"
    best_val, stale, plateau = float("inf"), 0, 0

    for epoch in range(args.epochs):
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), batch_size):
            rows = train_idx[order[start:start + batch_size]]
            params, opt_state, _ = step(
                params, opt_state, jnp.asarray(Xn[rows]), jnp.asarray(Yn[rows]))

        val_loss = float(val_loss_fn(params, x_val, y_val))

        # ReduceLROnPlateau(patience=20, factor=0.5, min_lr=1e-6) equivalent.
        hp = opt_state[1].hyperparams
        if val_loss < best_val - 1e-12:
            plateau = 0
        else:
            plateau += 1
            if plateau > 20:
                hp["learning_rate"] = jnp.maximum(
                    hp["learning_rate"] * 0.5, 1e-6)
                plateau = 0

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:4d}  val={val_loss:.6f}  "
                  f"lr={float(hp['learning_rate']):.1e}")

        if val_loss < best_val:
            best_val, stale = val_loss, 0
            save_params(best_path, params, args.hidden)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  early stop at epoch {epoch+1}")
                break

    save_params(weights_path, params, args.hidden)

    pred = np.asarray(forward(params, x_val))
    mae = np.abs(pred * y_std + y_mean - Y[val_idx]).mean(0)

    np.savez(out_dir / "anyobj_normalizer.npz",
             x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
             feature_names=np.array(DESCRIPTOR_NAMES + FLOW_FEATURES),
             target_names=np.array(TARGET_NAMES))

    print(f"\nbest val {best_val:.6f}  |  "
          f"val MAE Cd={mae[0]:.4f} Cl={mae[1]:.4f}")
    print(f"saved {weights_path.name}, {best_path.name}, anyobj_normalizer.npz")


def main():
    p = argparse.ArgumentParser(
        description="Train the any-mesh Cd/Cl surrogate")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--voxel-res", type=int, default=32)
    p.add_argument("--workers", type=int, default=None,
                   help="mesh-encoding processes (default: min(8, cores))")
    train(p.parse_args())


if __name__ == "__main__":
    main()
