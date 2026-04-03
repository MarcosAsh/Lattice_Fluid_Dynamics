"""
Training script for the super-resolution velocity field upscaler.

Patch-based MLP: takes a 3x3x3 coarse velocity neighborhood (108 floats)
and predicts the 2x2x2 fine sub-cells (32 floats) for 2x upscaling.
A trilinear skip connection means the MLP only learns high-frequency
corrections to the interpolated baseline.

Usage:
    python train_superres.py --data dataset/superres_data.srbin
    python train_superres.py --data dataset/superres_data.srbin --epochs 300
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SRBIN_MAGIC = 0x53524553
COARSE_DIM = 3 * 3 * 3 * 4  # 108: 3x3x3 neighborhood, 4 channels
FINE_DIM = 2 * 2 * 2 * 4    # 32: 2x2x2 output, 4 channels


def load_srbin(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load super-resolution patch dataset from .srbin file."""
    with open(path, "rb") as f:
        magic, version, num_patches, channels = struct.unpack("<IIII", f.read(16))
        if magic != SRBIN_MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X}")
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")

        record_size = (COARSE_DIM + FINE_DIM) * 4  # float32
        data = np.frombuffer(f.read(num_patches * record_size), dtype=np.float32)
        data = data.reshape(num_patches, COARSE_DIM + FINE_DIM)

    coarse = data[:, :COARSE_DIM]
    fine = data[:, COARSE_DIM:]
    return coarse, fine


def trilinear_upsample(coarse_patch: torch.Tensor) -> torch.Tensor:
    """
    Compute trilinear interpolation of the center cell from a 3x3x3 patch.

    The center cell (index [1,1,1]) gets split into 2x2x2 sub-cells.
    Each sub-cell is interpolated from the 8 surrounding coarse cells.

    Args:
        coarse_patch: (batch, 108) -- 3x3x3 * 4 channels, layout:
                      [z][y][x][channel], x fastest

    Returns:
        (batch, 32) -- 2x2x2 * 4 channels
    """
    batch = coarse_patch.shape[0]
    ch = 4
    # Reshape to (batch, 3, 3, 3, 4)
    vol = coarse_patch.view(batch, 3, 3, 3, ch)

    out = torch.zeros(batch, 2, 2, 2, ch, device=coarse_patch.device)

    # Each sub-cell at offset (dz, dy, dx) in [0,1] maps to weights
    # at fractional position (0.25 + 0.5*d) within the center cell.
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                # Fractional position within center cell [0,1]
                fx = 0.25 + 0.5 * dx
                fy = 0.25 + 0.5 * dy
                fz = 0.25 + 0.5 * dz

                # Trilinear weights from 8 corners
                # Corner indices: center is (1,1,1), neighbors at (1+d, 1+d, 1+d)
                # where d is 0 or 1 depending on which side of center
                val = torch.zeros(batch, ch, device=coarse_patch.device)
                for cz in range(2):
                    for cy in range(2):
                        for cx in range(2):
                            wz = fz if cz == 1 else (1.0 - fz)
                            wy = fy if cy == 1 else (1.0 - fy)
                            wx = fx if cx == 1 else (1.0 - fx)
                            w = wz * wy * wx
                            # Map to grid indices: center is 1,
                            # cz=0 means same as center (index 1),
                            # cz=1 means +1 direction (index 2 or 1+dz)
                            iz = 1 + cz if fz >= 0.5 else 1 - (1 - cz)
                            iy = 1 + cy if fy >= 0.5 else 1 - (1 - cy)
                            ix = 1 + cx if fx >= 0.5 else 1 - (1 - cx)
                            iz = max(0, min(2, iz))
                            iy = max(0, min(2, iy))
                            ix = max(0, min(2, ix))
                            val += w * vol[:, iz, iy, ix, :]

                out[:, dz, dy, dx, :] = val

    return out.reshape(batch, FINE_DIM)


class SuperResNet(nn.Module):
    """
    Patch-based MLP for 2x velocity field upscaling.

    Input:  3x3x3 coarse neighborhood (108 floats)
    Output: 2x2x2 fine sub-cells (32 floats)

    Uses a residual skip connection from trilinear interpolation
    so the network only learns high-frequency corrections.
    """

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(COARSE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, FINE_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = trilinear_upsample(x)
        correction = self.net(x)
        return skip + correction


def compute_psnr(mse: float, data_range: float = 1.0) -> float:
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(data_range ** 2 / mse)


def train(args):
    print(f"Loading dataset: {args.data}")
    coarse, fine = load_srbin(args.data)
    print(f"  Patches: {len(coarse)}")
    print(f"  Coarse shape: {coarse.shape}")
    print(f"  Fine shape: {fine.shape}")

    # Normalize per-channel (4 channels: ux, uy, uz, rho)
    coarse_flat = coarse.reshape(-1, 4)
    fine_flat = fine.reshape(-1, 4)
    all_vals = np.concatenate([coarse_flat, fine_flat], axis=0)
    ch_mean = all_vals.mean(axis=0)
    ch_std = all_vals.std(axis=0)
    ch_std[ch_std < 1e-8] = 1.0

    print(f"  Channel means: {ch_mean}")
    print(f"  Channel stds:  {ch_std}")

    # Normalize
    coarse_norm = (coarse.reshape(-1, 4) - ch_mean) / ch_std
    coarse_norm = coarse_norm.reshape(coarse.shape)
    fine_norm = (fine.reshape(-1, 4) - ch_mean) / ch_std
    fine_norm = fine_norm.reshape(fine.shape)

    # Train/val split
    n = len(coarse_norm)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    train_ds = TensorDataset(
        torch.from_numpy(coarse_norm[train_idx]).float(),
        torch.from_numpy(fine_norm[train_idx]).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(coarse_norm[val_idx]).float(),
        torch.from_numpy(fine_norm[val_idx]).float(),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
    )

    model = SuperResNet(hidden=args.hidden).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6
    )

    # Weighted MSE: velocity channels (0-2) weight 1.0, density (3) weight 0.5
    channel_weights = torch.tensor(
        [1.0, 1.0, 1.0, 0.5] * 8, device=device  # 8 sub-cells * 4 channels
    )

    best_val_loss = float("inf")
    patience_counter = 0
    out_dir = Path(args.data).parent
    weights_path = out_dir / "sr_model.pt"
    best_path = out_dir / "sr_model_best.pt"

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            diff = (pred - batch_y) ** 2
            loss = (diff * channel_weights).mean()
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
        val_mse = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                diff = (pred - batch_y) ** 2
                loss = (diff * channel_weights).mean()
                val_loss += loss.item() * batch_x.shape[0]
                val_mse += diff.mean().item() * batch_x.shape[0]
                val_count += batch_x.shape[0]

        val_loss /= val_count
        val_mse /= val_count
        val_psnr = compute_psnr(val_mse)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1:4d}/{args.epochs}  "
                f"train={train_loss:.6f}  val={val_loss:.6f}  "
                f"PSNR={val_psnr:.1f} dB  lr={lr:.1e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Save final and best
    torch.save(model.state_dict(), weights_path)
    print(f"\nTraining complete.")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Best PSNR: {compute_psnr(best_val_loss):.1f} dB")
    print(f"  Weights: {weights_path}")
    print(f"  Best weights: {best_path}")

    # Save normalizer for export
    norm_path = out_dir / "sr_normalizer.npz"
    np.savez(norm_path, mean=ch_mean, std=ch_std)
    print(f"  Normalizer: {norm_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train super-resolution velocity field upscaler"
    )
    parser.add_argument(
        "--data", required=True, help="Path to .srbin dataset"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
