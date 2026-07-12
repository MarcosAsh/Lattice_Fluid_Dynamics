"""
Train the Phase 2 voxel CNN Cd/Cl surrogate, in JAX.

Small 3D conv net over the solid voxel grid, with the Phase 1 descriptors
and log-Re concatenated into the head so it strictly extends the MLP.
Lateral mirror augmentation doubles effective data. Same contract as
train.py: z-score normalizer saved next to the weights.

Usage:
    python train_cnn.py                        # dataset/all/results.csv
    python train_cnn.py --width 24 --epochs 300
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import DEFAULT_DATA  # noqa: E402
from train import N_DESCRIPTORS  # noqa: E402
from train import OUT_DIR  # noqa: E402
from train import TARGET_NAMES  # noqa: E402
from train import encode_models  # noqa: E402
from train import read_rows  # noqa: E402
from train import zscore  # noqa: E402

N_FEATS = N_DESCRIPTORS + 1  # descriptors + log10 reynolds


def load_voxel_dataset(results_csv, voxel_res, workers=None):
    rows = read_rows(results_csv)
    enc = encode_models(sorted({r[0] for r in rows}), voxel_res, workers,
                        voxels=True)
    V = np.stack([enc[m][1] for m, *_ in rows])
    F = np.array([np.concatenate([enc[m][0], [np.log10(re)]])
                  for m, re, _, _ in rows], np.float32)
    Y = np.array([[cd, cl] for _, _, cd, cl in rows], np.float32)
    return V, F, Y, [r[0] for r in rows]


def init_params(key, width):
    chans = [1, width, 2 * width, 4 * width]
    keys = jax.random.split(key, len(chans) + 1)
    convs = []
    for i, (cin, cout) in enumerate(zip(chans[:-1], chans[1:])):
        w = (jax.random.normal(keys[i], (3, 3, 3, cin, cout))
             * jnp.sqrt(2.0 / (27 * cin)))
        convs.append({"w": w, "b": jnp.zeros(cout)})
    head = [4 * width + N_FEATS, 64, len(TARGET_NAMES)]
    dense = []
    for i, (fin, fout) in enumerate(zip(head[:-1], head[1:])):
        w = (jax.random.normal(keys[len(chans) + i - 1], (fin, fout))
             * jnp.sqrt(2.0 / fin))
        dense.append({"w": w, "b": jnp.zeros(fout)})
    return {"convs": convs, "dense": dense}


def forward(params, vox, feats):
    x = vox[..., None].astype(jnp.float32)
    for c in params["convs"]:
        x = jax.lax.conv_general_dilated(
            x, c["w"], (2, 2, 2), "SAME",
            dimension_numbers=("NDHWC", "DHWIO", "NDHWC")) + c["b"]
        x = jax.nn.relu(x)
    x = x.mean(axis=(1, 2, 3))
    x = jnp.concatenate([x, feats], axis=1)
    d1, d2 = params["dense"]
    x = jax.nn.relu(x @ d1["w"] + d1["b"])
    return x @ d2["w"] + d2["b"]


def mse(params, vox, feats, y):
    return jnp.mean((forward(params, vox, feats) - y) ** 2)


def save_params(path, params, width):
    flat = {"width": np.array(width)}
    for i, c in enumerate(params["convs"]):
        flat[f"c{i}w"], flat[f"c{i}b"] = np.asarray(c["w"]), np.asarray(c["b"])
    for i, d in enumerate(params["dense"]):
        flat[f"d{i}w"], flat[f"d{i}b"] = np.asarray(d["w"]), np.asarray(d["b"])
    np.savez(path, **flat)


def load_params(path):
    data = np.load(path)
    n_c = sum(1 for k in data.files if k.startswith("c") and k.endswith("w"))
    n_d = sum(1 for k in data.files if k.startswith("d") and k.endswith("w"))
    return {
        "convs": [{"w": jnp.asarray(data[f"c{i}w"]),
                   "b": jnp.asarray(data[f"c{i}b"])} for i in range(n_c)],
        "dense": [{"w": jnp.asarray(data[f"d{i}w"]),
                   "b": jnp.asarray(data[f"d{i}b"])} for i in range(n_d)],
    }


def train(args):
    data_path = Path(args.data).resolve()
    V, F, Y, models = load_voxel_dataset(data_path, args.voxel_res,
                                         getattr(args, "workers", None))
    print(f"{data_path}: {len(V)} samples, voxels {V.shape[1:]}, "
          f"{N_FEATS} scalar feats -> {TARGET_NAMES}")

    f_mean, f_std = zscore(F)
    y_mean, y_std = zscore(Y)
    Fn, Yn = (F - f_mean) / f_std, (Y - y_mean) / y_std

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(V))
    split = max(1, int(0.8 * len(V)))
    tr, va = idx[:split], idx[split:]
    v_val, f_val, y_val = (jnp.asarray(V[va]), jnp.asarray(Fn[va]),
                           jnp.asarray(Yn[va]))

    params = init_params(jax.random.PRNGKey(42), args.width)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=args.lr, weight_decay=1e-4),
    )
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, bv, bf, by):
        loss, grads = jax.value_and_grad(mse)(params, bv, bf, by)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    val_loss_fn = jax.jit(mse)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = OUT_DIR / "anyobj_cnn.npz"
    best_path = OUT_DIR / "anyobj_cnn_best.npz"
    best_val, stale, plateau = float("inf"), 0, 0

    for epoch in range(args.epochs):
        order = rng.permutation(len(tr))
        for start in range(0, len(order), args.batch_size):
            rows = tr[order[start:start + args.batch_size]]
            bv = V[rows]
            # mirror the lateral y axis, valid for z-up CFD meshes
            flip = rng.rand(len(rows)) < 0.5
            bv = bv.copy()
            bv[flip] = bv[flip][:, :, ::-1, :]
            params, opt_state, _ = step(
                params, opt_state, jnp.asarray(bv),
                jnp.asarray(Fn[rows]), jnp.asarray(Yn[rows]))

        val_loss = float(val_loss_fn(params, v_val, f_val, y_val))

        hp = opt_state[1].hyperparams
        if val_loss < best_val - 1e-12:
            plateau = 0
        else:
            plateau += 1
            if plateau > 10:
                hp["learning_rate"] = jnp.maximum(
                    hp["learning_rate"] * 0.5, 1e-6)
                plateau = 0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:4d}  val={val_loss:.6f}  "
                  f"lr={float(hp['learning_rate']):.1e}")

        if val_loss < best_val:
            best_val, stale = val_loss, 0
            save_params(best_path, params, args.width)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  early stop at epoch {epoch+1}")
                break

    save_params(weights_path, params, args.width)

    pred = np.asarray(forward(params, v_val, f_val))
    mae = np.abs(pred * y_std + y_mean - Y[va]).mean(0)

    np.savez(OUT_DIR / "anyobj_cnn_normalizer.npz",
             f_mean=f_mean, f_std=f_std, y_mean=y_mean, y_std=y_std,
             target_names=np.array(TARGET_NAMES))

    print(f"\nbest val {best_val:.6f}  |  "
          f"val MAE Cd={mae[0]:.4f} Cl={mae[1]:.4f}")
    print(f"saved {weights_path.name}, {best_path.name}, "
          f"anyobj_cnn_normalizer.npz")


def main():
    p = argparse.ArgumentParser(description="Train the voxel CNN surrogate")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--voxel-res", type=int, default=32)
    p.add_argument("--workers", type=int, default=None)
    train(p.parse_args())


if __name__ == "__main__":
    main()
