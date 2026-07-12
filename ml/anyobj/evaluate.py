"""
Evaluate the any-mesh Cd/Cl surrogate against LBM.

Two views:
  default   score a trained checkpoint on its held-out rows.
  --logo    leave-one-geometry-out cross-validation. This is the honest test
            of generalization: a random split puts the same mesh on both sides,
            so it only measures interpolation in Re, not unseen geometry. LOGO
            retrains on every other mesh and predicts the one held out.

Both compare against a mean-Cd baseline; the model is only useful if it beats
predicting the dataset average.

Usage:
    python evaluate.py --logo
    python evaluate.py --checkpoint anyobj_model_best.npz --plot
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (  # noqa: E402
    DEFAULT_DATA, N_INPUTS, TARGET_NAMES, forward, init_params, load_dataset,
    load_params, mse, zscore,
)

HERE = Path(__file__).resolve().parent


def fit(Xn, Yn, hidden, epochs, lr, seed=0):
    """Full-batch train a fresh surrogate (dataset is small)."""
    params = init_params(jax.random.PRNGKey(seed), hidden)
    tx = optax.adamw(lr, weight_decay=1e-4)
    opt_state = tx.init(params)
    x, y = jnp.asarray(Xn), jnp.asarray(Yn)

    @jax.jit
    def step(params, opt_state):
        grads = jax.grad(mse)(params, x, y)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    for _ in range(epochs):
        params, opt_state = step(params, opt_state)
    return params


def predict(params, Xn):
    return np.asarray(forward(params, jnp.asarray(Xn)))


def report(title, names, true, pred, baseline):
    """Print per-target MAE and the spread over geometries."""
    err = np.abs(pred - true)
    base = np.abs(baseline - true)
    print(f"\n=== {title} ===")
    print(f"  {'target':<6}{'model MAE':>12}{'baseline MAE':>14}")
    for i, t in enumerate(TARGET_NAMES):
        print(f"  {t:<6}{err[:, i].mean():>12.4f}{base[:, i].mean():>14.4f}")
    # Worst geometries by Cd error, to see where it struggles.
    order = np.argsort(-err[:, 0])
    print(f"  worst Cd: " + ", ".join(
        f"{names[j]} ({true[j,0]:.2f}->{pred[j,0]:.2f})" for j in order[:3]))


def run_logo(X, Y, names, args):
    names = np.array(names)
    geoms = sorted(set(names))
    if len(geoms) < 3:
        print(f"LOGO needs >=3 geometries, have {len(geoms)}: {geoms}. "
              f"Run the geometry sweep first (anyobj/sweep.yaml).")
        return
    pred = np.zeros_like(Y)
    base = np.zeros_like(Y)
    for g in geoms:
        te, tr = names == g, names != g
        x_mean, x_std = zscore(X[tr])
        y_mean, y_std = zscore(Y[tr])
        params = fit((X[tr] - x_mean) / x_std, (Y[tr] - y_mean) / y_std,
                     args.hidden, args.epochs, args.lr)
        pred[te] = predict(params, (X[te] - x_mean) / x_std) * y_std + y_mean
        base[te] = Y[tr].mean(0)  # mean-Cd baseline knows nothing about shape
    report("Leave-one-geometry-out (predicted vs LBM)", names, Y, pred, base)
    if args.plot:
        scatter(Y[:, 0], pred[:, 0], "logo")


def run_checkpoint(X, Y, names, args):
    norm = np.load(HERE / "anyobj_normalizer.npz")
    params = load_params(args.checkpoint)

    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    val = idx[max(1, int(0.8 * n)):] if n >= 5 else idx
    Xn = (X[val] - norm["x_mean"]) / norm["x_std"]
    pred = predict(params, Xn) * norm["y_std"] + norm["y_mean"]
    base = np.tile(Y.mean(0), (len(val), 1))
    print("Note: random split shares geometries with training; use --logo for "
          "the generalization number.")
    report("Held-out rows (predicted vs LBM)", np.array(names)[val],
           Y[val], pred, base)
    if args.plot:
        scatter(Y[val, 0], pred[:, 0], "checkpoint")


def scatter(true, pred, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    out = HERE / "eval_report"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true, pred, s=20)
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lims, lims, "r--", linewidth=0.8)
    ax.set_xlabel("LBM Cd")
    ax.set_ylabel("Predicted Cd")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out / f"cd_{tag}.png", dpi=150)
    print(f"\nPlot: {out / f'cd_{tag}.png'}")


def main():
    p = argparse.ArgumentParser(description="Evaluate the any-mesh Cd surrogate")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--checkpoint", default=str(HERE / "anyobj_model_best.npz"))
    p.add_argument("--logo", action="store_true",
                   help="leave-one-geometry-out cross-validation")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--voxel-res", type=int, default=32)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    X, Y, names = load_dataset(Path(args.data).resolve(), args.voxel_res)
    print(f"{len(X)} samples, {len(set(names))} geometries, {N_INPUTS} inputs")

    if args.logo:
        run_logo(X, Y, names, args)
    else:
        run_checkpoint(X, Y, names, args)


if __name__ == "__main__":
    main()
