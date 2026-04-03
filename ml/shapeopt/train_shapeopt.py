"""
Training script for the geometry-aware Cd/Cl surrogate model.

Takes FFD displacement parameters (plus wind speed and Reynolds number)
as input and predicts Cd and Cl. Uses a SwiGLU-based MLP architecture.

Usage:
    python train_shapeopt.py --data dataset/shapeopt_data.sobin
    python train_shapeopt.py --data dataset/shapeopt_data.sobin --epochs 500 --lr 1e-3
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SOBIN_MAGIC = 0x534F5054
LTWS_MAGIC = 0x4C545753
LTWS_VERSION = 1


# ---------------------------------------------------------------------------
# Dataset loading (.sobin)
# ---------------------------------------------------------------------------

def load_sobin(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load shape optimization dataset from .sobin binary.

    Returns:
        displacements: float32 array (N, num_ffd_params)
        outputs: float32 array (N, 2) -- [cd, cl] per sample
    """
    with open(path, "rb") as f:
        magic, version, num_samples, num_ffd_params, num_outputs = struct.unpack(
            "<IIIII", f.read(20)
        )
        if magic != SOBIN_MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X} (expected 0x{SOBIN_MAGIC:08X})")
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        if num_outputs != 2:
            raise ValueError(f"Expected 2 outputs (cd, cl), got {num_outputs}")

        record_bytes = (num_ffd_params + num_outputs) * 4
        raw = np.frombuffer(f.read(num_samples * record_bytes), dtype=np.float32)
        raw = raw.reshape(num_samples, num_ffd_params + num_outputs)

    displacements = raw[:, :num_ffd_params].copy()
    outputs = raw[:, num_ffd_params:].copy()
    return displacements, outputs


# ---------------------------------------------------------------------------
# SwiGLU building block
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """
    SwiGLU activation block.

    gate(x) = W_down @ (silu(W_gate @ x) * (W_up @ x))

    Input dimension: in_dim
    Hidden (expanded) dimension: hidden_dim
    Output dimension: in_dim (same as input for residual-friendly stacking)
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------

class ShapeOptSurrogate(nn.Module):
    """
    Geometry-aware surrogate for Cd/Cl prediction.

    Architecture:
        Linear(input_dim, 256) -> SwiGLU(256, 512) ->
        Linear(256, 128) -> SwiGLU(128, 256) -> Linear(128, 2)

    Input: FFD displacements + wind_speed + reynolds
    Output: [Cd, Cl]
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.swiglu1 = SwiGLU(256, 512)
        self.fc2 = nn.Linear(256, 128)
        self.swiglu2 = SwiGLU(128, 256)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.swiglu1(h)
        h = self.fc2(h)
        h = self.swiglu2(h)
        return self.fc3(h)


# ---------------------------------------------------------------------------
# LTWS export (matches ml/superres/export_weights.py)
# ---------------------------------------------------------------------------

def export_ltws(state_dict: dict, output_path: str):
    """
    Export PyTorch state dict to LTWS binary format.

    Format:
        Header: magic (u32) | version (u32) | param_count (u32)
        Per param: ndim (u32) | shape (u32 x ndim) | data (f32 x numel)
    """
    params = []
    for name in sorted(state_dict.keys()):
        t = state_dict[name].cpu().float().numpy()
        params.append((name, t))

    with open(output_path, "wb") as f:
        f.write(struct.pack("<III", LTWS_MAGIC, LTWS_VERSION, len(params)))

        for name, arr in params:
            ndim = len(arr.shape)
            f.write(struct.pack("<I", ndim))
            for d in arr.shape:
                f.write(struct.pack("<I", d))
            f.write(arr.tobytes())

    total_params = sum(arr.size for _, arr in params)
    print(f"Exported {len(params)} tensors ({total_params:,} parameters) to {output_path}")
    for name, arr in params:
        print(f"  {name}: {list(arr.shape)}")


def save_normalizer(means: np.ndarray, stds: np.ndarray, output_path: str):
    """
    Save z-score normalizer as raw binary.

    Format: float32[input_dim] means, float32[input_dim] stds
    """
    means = means.astype(np.float32)
    stds = stds.astype(np.float32)
    with open(output_path, "wb") as f:
        f.write(means.tobytes())
        f.write(stds.tobytes())
    print(f"Saved normalizer ({len(means)} dims) to {output_path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    print(f"Loading dataset: {args.data}")
    displacements, outputs = load_sobin(args.data)
    num_samples, num_ffd_params = displacements.shape
    print(f"  Samples: {num_samples}")
    print(f"  FFD params: {num_ffd_params}")
    print(f"  Cd range: [{outputs[:, 0].min():.4f}, {outputs[:, 0].max():.4f}]")
    print(f"  Cl range: [{outputs[:, 1].min():.4f}, {outputs[:, 1].max():.4f}]")

    # Build input features: displacements + wind_speed + reynolds
    # For now, wind_speed and reynolds are constant across the dataset
    # (set from config). They're appended as features so the model
    # generalizes when we later sweep those parameters.
    wind_speed = args.wind_speed
    reynolds = args.reynolds
    print(f"  Wind speed: {wind_speed}")
    print(f"  Reynolds: {reynolds}")

    extra_features = np.column_stack([
        np.full(num_samples, wind_speed, dtype=np.float32),
        np.full(num_samples, reynolds, dtype=np.float32),
    ])
    features = np.hstack([displacements, extra_features])
    input_dim = features.shape[1]
    print(f"  Input dim: {input_dim} ({num_ffd_params} FFD + 2 flow params)")

    # Z-score normalization
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    features_norm = (features - feat_mean) / feat_std

    print(f"  Feature means range: [{feat_mean.min():.4f}, {feat_mean.max():.4f}]")
    print(f"  Feature stds range:  [{feat_std.min():.4f}, {feat_std.max():.4f}]")

    # Train/val split (80/20)
    n = len(features_norm)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    train_ds = TensorDataset(
        torch.from_numpy(features_norm[train_idx]).float(),
        torch.from_numpy(outputs[train_idx]).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(features_norm[val_idx]).float(),
        torch.from_numpy(outputs[val_idx]).float(),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
    )

    model = ShapeOptSurrogate(input_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=30, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    out_dir = Path(args.data).parent
    best_path = out_dir / "shapeopt_model_best.pt"
    final_path = out_dir / "shapeopt_model.pt"

    best_val_cd_mae = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.shape[0]
            train_count += batch_x.shape[0]
        train_loss /= train_count

        # Validate
        model.eval()
        val_loss = 0.0
        val_cd_ae = 0.0
        val_cl_ae = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_x.shape[0]
                val_cd_ae += torch.abs(pred[:, 0] - batch_y[:, 0]).sum().item()
                val_cl_ae += torch.abs(pred[:, 1] - batch_y[:, 1]).sum().item()
                val_count += batch_x.shape[0]

        val_loss /= val_count
        val_cd_mae = val_cd_ae / val_count
        val_cl_mae = val_cl_ae / val_count
        scheduler.step(val_cd_mae)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1:4d}/{args.epochs}  "
                f"train_mse={train_loss:.6f}  val_mse={val_loss:.6f}  "
                f"val_cd_mae={val_cd_mae:.6f}  val_cl_mae={val_cl_mae:.6f}  "
                f"lr={lr:.1e}"
            )

        # Early stopping on val Cd MAE
        if val_cd_mae < best_val_cd_mae:
            best_val_cd_mae = val_cd_mae
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "num_ffd_params": num_ffd_params,
                "feat_mean": feat_mean,
                "feat_std": feat_std,
                "best_val_cd_mae": best_val_cd_mae,
                "epoch": epoch + 1,
            }, best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Save final checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "num_ffd_params": num_ffd_params,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "best_val_cd_mae": best_val_cd_mae,
        "epoch": epoch + 1,
    }, final_path)

    print(f"\nTraining complete.")
    print(f"  Best val Cd MAE: {best_val_cd_mae:.6f}")
    print(f"  Best weights: {best_path}")
    print(f"  Final weights: {final_path}")

    # Export LTWS format
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    ltws_path = out_dir / "shapeopt_model.bin"
    export_ltws(best_ckpt["model_state_dict"], str(ltws_path))

    # Save normalizer
    norm_path = out_dir / "shapeopt_normalizer.bin"
    save_normalizer(feat_mean, feat_std, str(norm_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train geometry-aware Cd/Cl surrogate model"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to .sobin dataset",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--wind-speed", type=float, default=1.0,
        help="Wind speed used during data generation (default: 1.0)",
    )
    parser.add_argument(
        "--reynolds", type=float, default=10000,
        help="Reynolds number used during data generation (default: 10000)",
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Error: dataset not found at {args.data}", file=sys.stderr)
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
