"""
Evaluation framework for the super-resolution upscaler.

Loads trained weights, runs inference on the validation set,
and reports PSNR, per-channel MSE, and divergence error.
Optionally generates comparison plots.

Usage:
    python evaluate_superres.py --data dataset/superres_data.srbin --checkpoint dataset/sr_model_best.pt
    python evaluate_superres.py --data dataset/superres_data.srbin --checkpoint dataset/sr_model_best.pt --plot
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# Re-use model and data loading from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_superres import (
    COARSE_DIM,
    FINE_DIM,
    SRBIN_MAGIC,
    SuperResNet,
    load_srbin,
    trilinear_upsample,
)

CHANNEL_NAMES = ["ux", "uy", "uz", "rho"]


def compute_divergence(fine_field: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Compute velocity divergence on the fine grid.

    For incompressible flow, div(u) should be near zero.
    Uses central differences.

    Args:
        fine_field: (N, 32) predictions for 2x2x2 patches
        shape: not used for patch-level, computes within each patch

    Returns:
        Per-patch mean absolute divergence
    """
    # Reshape to (N, 2, 2, 2, 4)
    vol = fine_field.reshape(-1, 2, 2, 2, 4)
    # Finite differences along each axis (forward difference since only 2 cells)
    dudx = vol[:, :, :, 1, 0] - vol[:, :, :, 0, 0]  # du/dx
    dvdy = vol[:, :, 1, :, 1] - vol[:, :, 0, :, 1]  # dv/dy
    dwdz = vol[:, 1, :, :, 2] - vol[:, 0, :, :, 2]  # dw/dz
    div = dudx + dvdy + dwdz
    return np.abs(div).mean(axis=(1, 2))


def evaluate(args):
    print(f"Loading dataset: {args.data}")
    coarse, fine = load_srbin(args.data)

    # Load normalizer
    norm_path = Path(args.data).parent / "sr_normalizer.npz"
    if norm_path.exists():
        ndata = np.load(norm_path)
        ch_mean, ch_std = ndata["mean"], ndata["std"]
    else:
        # Recompute from data
        all_vals = np.concatenate([
            coarse.reshape(-1, 4), fine.reshape(-1, 4)
        ], axis=0)
        ch_mean = all_vals.mean(axis=0)
        ch_std = all_vals.std(axis=0)
        ch_std[ch_std < 1e-8] = 1.0

    coarse_norm = ((coarse.reshape(-1, 4) - ch_mean) / ch_std).reshape(coarse.shape)
    fine_norm = ((fine.reshape(-1, 4) - ch_mean) / ch_std).reshape(fine.shape)

    # Val split (same seed as training)
    n = len(coarse_norm)
    idx = np.random.RandomState(42).permutation(n)
    val_idx = idx[int(0.8 * n):]
    val_coarse = coarse_norm[val_idx]
    val_fine = fine_norm[val_idx]
    val_fine_raw = fine[val_idx]

    print(f"  Val patches: {len(val_coarse)}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuperResNet(hidden=args.hidden).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Run inference
    coarse_t = torch.from_numpy(val_coarse).float().to(device)
    with torch.no_grad():
        pred_t = model(coarse_t)
        skip_t = trilinear_upsample(coarse_t)

    pred = pred_t.cpu().numpy()
    skip = skip_t.cpu().numpy()

    # Denormalize predictions
    pred_raw = (pred.reshape(-1, 4) * ch_std + ch_mean).reshape(pred.shape)
    skip_raw = (skip.reshape(-1, 4) * ch_std + ch_mean).reshape(skip.shape)

    # Per-channel metrics
    print("\n=== Per-Channel Metrics (Model vs Ground Truth) ===")
    print(f"{'Channel':>8} {'MAE':>10} {'RMSE':>10} {'PSNR (dB)':>10}")
    print("-" * 42)

    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        # Extract this channel from all sub-cells
        pred_ch = pred_raw.reshape(-1, 4)[:, ch_idx]
        true_ch = val_fine_raw.reshape(-1, 4)[:, ch_idx]
        skip_ch = skip_raw.reshape(-1, 4)[:, ch_idx]

        mae = np.abs(pred_ch - true_ch).mean()
        mse = ((pred_ch - true_ch) ** 2).mean()
        rmse = np.sqrt(mse)
        data_range = true_ch.max() - true_ch.min()
        psnr = 10 * np.log10(data_range**2 / (mse + 1e-10)) if mse > 1e-10 else 100

        print(f"{ch_name:>8} {mae:>10.6f} {rmse:>10.6f} {psnr:>10.1f}")

    # Trilinear baseline comparison
    print("\n=== Trilinear Baseline Comparison ===")
    print(f"{'Channel':>8} {'Model MAE':>10} {'Trilinear MAE':>14} {'Improvement':>12}")
    print("-" * 50)

    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        pred_ch = pred_raw.reshape(-1, 4)[:, ch_idx]
        true_ch = val_fine_raw.reshape(-1, 4)[:, ch_idx]
        skip_ch = skip_raw.reshape(-1, 4)[:, ch_idx]

        model_mae = np.abs(pred_ch - true_ch).mean()
        trilin_mae = np.abs(skip_ch - true_ch).mean()
        improvement = (1 - model_mae / (trilin_mae + 1e-10)) * 100

        print(f"{ch_name:>8} {model_mae:>10.6f} {trilin_mae:>14.6f} {improvement:>11.1f}%")

    # Divergence error
    pred_div = compute_divergence(pred_raw, None)
    true_div = compute_divergence(val_fine_raw, None)
    skip_div = compute_divergence(skip_raw, None)

    print(f"\n=== Divergence Error (lower = better mass conservation) ===")
    print(f"  Ground truth: {true_div.mean():.6f} +/- {true_div.std():.6f}")
    print(f"  Trilinear:    {skip_div.mean():.6f} +/- {skip_div.std():.6f}")
    print(f"  Model:        {pred_div.mean():.6f} +/- {pred_div.std():.6f}")

    # Overall PSNR
    overall_mse = ((pred_raw - val_fine_raw) ** 2).mean()
    data_range = val_fine_raw.max() - val_fine_raw.min()
    overall_psnr = 10 * np.log10(data_range**2 / (overall_mse + 1e-10))
    print(f"\n  Overall PSNR: {overall_psnr:.1f} dB")

    if args.plot:
        generate_plots(pred_raw, skip_raw, val_fine_raw, args)


def generate_plots(pred, skip, truth, args):
    """Generate evaluation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    out_dir = Path(args.data).parent / "sr_eval_report"
    out_dir.mkdir(exist_ok=True)

    # Scatter: predicted vs true for each channel
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ax = axes[ch_idx]
        p = pred.reshape(-1, 4)[:, ch_idx]
        t = truth.reshape(-1, 4)[:, ch_idx]
        # Subsample for plotting
        n = min(5000, len(p))
        idx = np.random.choice(len(p), n, replace=False)
        ax.scatter(t[idx], p[idx], alpha=0.1, s=1)
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, "r--", linewidth=0.5)
        ax.set_xlabel(f"True {ch_name}")
        ax.set_ylabel(f"Predicted {ch_name}")
        ax.set_title(ch_name)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=150)
    plt.close()

    # Error distribution
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ax = axes[ch_idx]
        err = pred.reshape(-1, 4)[:, ch_idx] - truth.reshape(-1, 4)[:, ch_idx]
        ax.hist(err, bins=100, alpha=0.7, density=True)
        ax.set_xlabel(f"Error ({ch_name})")
        ax.set_ylabel("Density")
        ax.set_title(f"{ch_name} error dist")
        ax.axvline(0, color="r", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "error_distribution.png", dpi=150)
    plt.close()

    # Model vs trilinear improvement
    model_errs = []
    trilin_errs = []
    for ch_idx in range(4):
        p = pred.reshape(-1, 4)[:, ch_idx]
        s = skip.reshape(-1, 4)[:, ch_idx]
        t = truth.reshape(-1, 4)[:, ch_idx]
        model_errs.append(np.abs(p - t).mean())
        trilin_errs.append(np.abs(s - t).mean())

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(4)
    width = 0.35
    ax.bar(x - width / 2, trilin_errs, width, label="Trilinear")
    ax.bar(x + width / 2, model_errs, width, label="Model")
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES)
    ax.set_ylabel("MAE")
    ax.set_title("Model vs Trilinear Baseline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "baseline_comparison.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate super-resolution upscaler"
    )
    parser.add_argument("--data", required=True, help="Path to .srbin dataset")
    parser.add_argument("--checkpoint", required=True, help="Model .pt file")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
