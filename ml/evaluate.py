"""
Evaluation framework for the Cd/Cl surrogate model.

Loads trained weights (model.bin) and normalization stats (model_norm.bin),
runs inference on the dataset, and reports MAE, RMSE, R-squared, and max
error with per-model breakdowns.  Optionally generates plots.

Usage:
    python evaluate.py
    python evaluate.py --plot
    python evaluate.py --data dataset/training_data.bin --weights model.bin
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).parent
MODEL_IDS = {0: "car", 1: "ahmed25", 2: "ahmed35"}


# Weight loading (matches weights_io.cpp LTWS format)

def load_weights(path: str) -> list[np.ndarray]:
    """Load parameter tensors from a LTWS binary file."""
    with open(path, "rb") as f:
        magic, version, count = struct.unpack("<III", f.read(12))
        if magic != 0x4C545753:
            raise ValueError(f"Bad magic: 0x{magic:08X} (expected 0x4C545753)")
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")

        params = []
        for _ in range(count):
            (ndim,) = struct.unpack("<I", f.read(4))
            shape = struct.unpack(f"<{ndim}I", f.read(4 * ndim))
            numel = 1
            for d in shape:
                numel *= d
            data = np.frombuffer(f.read(4 * numel), dtype=np.float32).copy()
            params.append(data.reshape(shape))
        return params


def load_normalizer(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load z-score normalization params (3 means + 3 stds)."""
    with open(path, "rb") as f:
        mean = np.frombuffer(f.read(12), dtype=np.float32).copy()
        std = np.frombuffer(f.read(12), dtype=np.float32).copy()
    return mean, std


# Dataset loading (matches training_data.bin format)

def load_dataset(path: str) -> np.ndarray:
    """
    Load binary dataset. Returns (N, 5) array:
    columns = [wind_speed, reynolds, model_id, cd, cl]
    """
    with open(path, "rb") as f:
        magic, version, num_records, num_features = struct.unpack("<IIII", f.read(16))
        if magic != 0x4C415454:
            raise ValueError(f"Bad magic: 0x{magic:08X}")
        if num_features != 5:
            raise ValueError(f"Expected 5 features, got {num_features}")
        data = np.frombuffer(
            f.read(4 * num_records * num_features), dtype=np.float32
        ).copy()
        return data.reshape(num_records, num_features)


# Model forward pass (numpy, matches train.cpp SurrogateModel)

def swish(x: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
    return x * sig


def forward(x: np.ndarray, params: list[np.ndarray]) -> np.ndarray:
    """
    Run the surrogate MLP forward pass.

    x: (3, B) normalized input features
    params: 12 weight arrays in registration order
    Returns: (2, B) predictions [cd, cl]
    """
    fc1_W, fc1_b = params[0], params[1]
    sg1_Wg, sg1_Wu, sg1_Wd = params[2], params[3], params[4]
    fc2_W, fc2_b = params[5], params[6]
    sg2_Wg, sg2_Wu, sg2_Wd = params[7], params[8], params[9]
    fc3_W, fc3_b = params[10], params[11]

    # Linear(3, 256)
    h = fc1_W @ x + fc1_b  # (256, B)

    # SwiGLU(256, 512)
    gate = swish(sg1_Wg @ h)    # (512, B)
    up = sg1_Wu @ h              # (512, B)
    h = sg1_Wd @ (gate * up)    # (256, B)

    # Linear(256, 128)
    h = fc2_W @ h + fc2_b       # (128, B)

    # SwiGLU(128, 256)
    gate = swish(sg2_Wg @ h)    # (256, B)
    up = sg2_Wu @ h              # (256, B)
    h = sg2_Wd @ (gate * up)    # (128, B)

    # Linear(128, 2)
    out = fc3_W @ h + fc3_b     # (2, B)
    return out


# Metrics

def compute_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    abs_err = np.abs(err)
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(r2),
        "max_err": float(np.max(abs_err)),
        "mean_true": float(np.mean(true)),
        "mean_pred": float(np.mean(pred)),
    }


def print_metrics(name: str, m: dict, indent: int = 0):
    pad = " " * indent
    print(f"{pad}{name}:")
    print(f"{pad}  MAE       = {m['mae']:.6f}")
    print(f"{pad}  RMSE      = {m['rmse']:.6f}")
    print(f"{pad}  R-squared = {m['r2']:.6f}")
    print(f"{pad}  Max error = {m['max_err']:.6f}")
    print(f"{pad}  Mean true = {m['mean_true']:.6f}")
    print(f"{pad}  Mean pred = {m['mean_pred']:.6f}")


# Plots

def generate_plots(
    dataset: np.ndarray,
    cd_pred: np.ndarray,
    cl_pred: np.ndarray,
    output_dir: Path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    cd_true = dataset[:, 3]
    cl_true = dataset[:, 4]
    model_ids = dataset[:, 2]

    colors = {0: "#e06c75", 1: "#61afef", 2: "#98c379"}
    labels = MODEL_IDS

    # Cd: predicted vs true scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    for mid in sorted(MODEL_IDS.keys()):
        mask = model_ids == mid
        ax.scatter(
            cd_true[mask], cd_pred[mask],
            c=colors[mid], label=labels[mid], alpha=0.7, s=20,
        )
    lo = min(cd_true.min(), cd_pred.min()) * 0.95
    hi = max(cd_true.max(), cd_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("True Cd")
    ax.set_ylabel("Predicted Cd")
    ax.set_title("Cd: Predicted vs True")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_dir / "cd_scatter.png", dpi=150)
    plt.close(fig)

    # Cl: predicted vs true scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    for mid in sorted(MODEL_IDS.keys()):
        mask = model_ids == mid
        ax.scatter(
            cl_true[mask], cl_pred[mask],
            c=colors[mid], label=labels[mid], alpha=0.7, s=20,
        )
    lo = min(cl_true.min(), cl_pred.min()) * 0.95
    hi = max(cl_true.max(), cl_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("True Cl")
    ax.set_ylabel("Predicted Cl")
    ax.set_title("Cl: Predicted vs True")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_dir / "cl_scatter.png", dpi=150)
    plt.close(fig)

    # Error distribution histograms
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(cd_pred - cd_true, bins=30, color="#e06c75", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("Cd Error (pred - true)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Cd Error Distribution")

    axes[1].hist(cl_pred - cl_true, bins=30, color="#61afef", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="black", lw=0.8, ls="--")
    axes[1].set_xlabel("Cl Error (pred - true)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Cl Error Distribution")

    fig.tight_layout()
    fig.savefig(output_dir / "error_distribution.png", dpi=150)
    plt.close(fig)

    # Per-model error boxplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cd_errors_by_model = []
    cl_errors_by_model = []
    model_names = []
    for mid in sorted(MODEL_IDS.keys()):
        mask = model_ids == mid
        cd_errors_by_model.append(np.abs(cd_pred[mask] - cd_true[mask]))
        cl_errors_by_model.append(np.abs(cl_pred[mask] - cl_true[mask]))
        model_names.append(labels[mid])

    axes[0].boxplot(cd_errors_by_model, tick_labels=model_names)
    axes[0].set_ylabel("|Cd Error|")
    axes[0].set_title("Cd Absolute Error by Model")

    axes[1].boxplot(cl_errors_by_model, tick_labels=model_names)
    axes[1].set_ylabel("|Cl Error|")
    axes[1].set_title("Cl Absolute Error by Model")

    fig.tight_layout()
    fig.savefig(output_dir / "error_by_model.png", dpi=150)
    plt.close(fig)

    # Cd error vs wind speed
    fig, ax = plt.subplots(figsize=(8, 4))
    for mid in sorted(MODEL_IDS.keys()):
        mask = model_ids == mid
        ax.scatter(
            dataset[mask, 0], np.abs(cd_pred[mask] - cd_true[mask]),
            c=colors[mid], label=labels[mid], alpha=0.7, s=20,
        )
    ax.set_xlabel("Wind Speed")
    ax.set_ylabel("|Cd Error|")
    ax.set_title("Cd Absolute Error vs Wind Speed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cd_error_vs_windspeed.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {output_dir}/")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Cd/Cl surrogate model"
    )
    parser.add_argument(
        "--weights", default=str(ML_DIR / "model.bin"),
        help="Path to trained weights (default: ml/model.bin)",
    )
    parser.add_argument(
        "--norm", default=str(ML_DIR / "model_norm.bin"),
        help="Path to normalization params (default: ml/model_norm.bin)",
    )
    parser.add_argument(
        "--data", default=str(ML_DIR / "dataset" / "training_data.bin"),
        help="Path to binary dataset (default: ml/dataset/training_data.bin)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate plots to ml/eval_report/",
    )
    parser.add_argument(
        "--output-dir", default=str(ML_DIR / "eval_report"),
        help="Directory for plot output (default: ml/eval_report/)",
    )
    args = parser.parse_args()

    # Load everything
    print(f"Weights:    {args.weights}")
    print(f"Normalizer: {args.norm}")
    print(f"Dataset:    {args.data}")
    print()

    for path in [args.weights, args.norm, args.data]:
        if not Path(path).exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)

    params = load_weights(args.weights)
    norm_mean, norm_std = load_normalizer(args.norm)
    dataset = load_dataset(args.data)

    print(f"Loaded {len(params)} parameter tensors:")
    for i, p in enumerate(params):
        print(f"  [{i:2d}] shape={str(list(p.shape)):20s}  numel={p.size}")
    total_params = sum(p.size for p in params)
    print(f"  Total: {total_params:,} parameters")
    print()

    print(f"Normalizer: mean={norm_mean}  std={norm_std}")
    print(f"Dataset: {len(dataset)} samples")
    print()

    # Normalize inputs: (3, N)
    features = dataset[:, :3].T.copy()  # (3, N)
    for j in range(3):
        features[j] = (features[j] - norm_mean[j]) / norm_std[j]

    # Run inference
    predictions = forward(features, params)  # (2, N)
    cd_pred = predictions[0]
    cl_pred = predictions[1]
    cd_true = dataset[:, 3]
    cl_true = dataset[:, 4]

    # Overall metrics
    print("=" * 56)
    print("OVERALL METRICS")
    print("=" * 56)
    cd_metrics = compute_metrics(cd_true, cd_pred)
    cl_metrics = compute_metrics(cl_true, cl_pred)
    print_metrics("Cd", cd_metrics)
    print()
    print_metrics("Cl", cl_metrics)
    print()

    # Per-model breakdowns
    print("=" * 56)
    print("PER-MODEL BREAKDOWN")
    print("=" * 56)
    model_ids = dataset[:, 2]
    for mid in sorted(MODEL_IDS.keys()):
        mask = model_ids == mid
        n = int(np.sum(mask))
        name = MODEL_IDS[mid]
        print(f"\n{name} ({n} samples)")
        print("-" * 40)
        m_cd = compute_metrics(cd_true[mask], cd_pred[mask])
        m_cl = compute_metrics(cl_true[mask], cl_pred[mask])
        print_metrics("Cd", m_cd, indent=2)
        print()
        print_metrics("Cl", m_cl, indent=2)

    # Sample predictions
    print()
    print("=" * 56)
    print("SAMPLE PREDICTIONS")
    print("=" * 56)
    print(f"{'wind':>8s} {'Re':>8s} {'model':>8s} | "
          f"{'true_Cd':>8s} {'pred_Cd':>8s} {'err':>8s} | "
          f"{'true_Cl':>8s} {'pred_Cl':>8s} {'err':>8s}")
    print("-" * 88)
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(dataset), size=min(15, len(dataset)), replace=False)
    sample_idx.sort()
    for i in sample_idx:
        ws, re, mid, cd_t, cl_t = dataset[i]
        cd_p, cl_p = cd_pred[i], cl_pred[i]
        name = MODEL_IDS.get(int(mid), f"?{mid}")
        print(f"{ws:8.2f} {re:8.0f} {name:>8s} | "
              f"{cd_t:8.4f} {cd_p:8.4f} {cd_p - cd_t:+8.4f} | "
              f"{cl_t:8.4f} {cl_p:8.4f} {cl_p - cl_t:+8.4f}")

    # Plots
    if args.plot:
        generate_plots(dataset, cd_pred, cl_pred, Path(args.output_dir))

    print()


if __name__ == "__main__":
    main()
