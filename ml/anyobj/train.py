"""
Train the any-OBJ Cd/Cl surrogate (Phase 1: descriptor MLP).

Each training row becomes [geometry descriptors, wind_speed, reynolds] -> [Cd, Cl],
so the model reads geometry directly instead of a model-id label. Descriptors
come from geometry.encode; inputs and targets are z-score normalized and the
normalizer is saved next to the weights.

Usage:
    python train.py                                  # dataset/v2/results.csv
    python train.py --data ../dataset/v2/results.csv --epochs 500 --hidden 128
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import DESCRIPTOR_NAMES, N_DESCRIPTORS, encode  # noqa: E402

# Dataset model name -> OBJ, mirroring simulation/modal_worker.py.
ASSETS = Path(__file__).resolve().parents[2] / "simulation" / "assets" / "3d-files"
MODEL_OBJS = {
    "car": ASSETS / "car-model.obj",
    "ahmed25": ASSETS / "ahmed_25deg_m.obj",
    "ahmed35": ASSETS / "ahmed_35deg_m.obj",
}

FLOW_FEATURES = ["wind_speed", "reynolds"]
TARGET_NAMES = ["cd", "cl"]
N_INPUTS = N_DESCRIPTORS + len(FLOW_FEATURES)
N_TARGETS = len(TARGET_NAMES)
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "dataset" / "v2" / "results.csv"


def load_dataset(results_csv, voxel_res):
    """Read a data-gen results CSV into (X, Y) arrays and the models used."""
    cache = {}

    def descriptors(model):
        if model not in cache:
            obj = MODEL_OBJS.get(model)
            if obj is None or not obj.exists():
                raise FileNotFoundError(f"No OBJ for model '{model}' ({obj})")
            cache[model] = encode(str(obj), voxel_res)["descriptors"]
        return cache[model]

    feats, targets, used = [], [], set()
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if not row.get("cd_value") or not row.get("cl_value"):
                continue
            flow = [float(row["wind_speed"]), float(row["reynolds"])]
            feats.append(np.concatenate([descriptors(row["model"]), flow]))
            targets.append([float(row["cd_value"]), float(row["cl_value"])])
            used.add(row["model"])

    if not feats:
        raise ValueError(f"No usable rows in {results_csv}")
    return (np.asarray(feats, np.float32),
            np.asarray(targets, np.float32), sorted(used))


def zscore(arr):
    """Per-column (mean, std); zero-variance columns get std 1."""
    mean, std = arr.mean(0), arr.std(0)
    std[std < 1e-8] = 1.0
    return mean, std


class SurrogateNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_INPUTS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_TARGETS),
        )

    def forward(self, x):
        return self.net(x)


def train(args):
    data_path = Path(args.data).resolve()
    X, Y, used = load_dataset(data_path, args.voxel_res)
    print(f"{data_path}: {len(X)} samples ({', '.join(used)}), "
          f"{N_INPUTS} inputs -> {TARGET_NAMES}")

    x_mean, x_std = zscore(X)
    y_mean, y_std = zscore(Y)
    Xn, Yn = (X - x_mean) / x_std, (Y - y_mean) / y_std

    n = len(Xn)
    idx = np.random.RandomState(42).permutation(n)
    if n >= 5:
        split = max(1, int(0.8 * n))
        train_idx, val_idx = idx[:split], idx[split:]
    else:
        print("  Few samples: validating on the training set.")
        train_idx = val_idx = idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loader(rows, batch, shuffle):
        ds = TensorDataset(torch.from_numpy(Xn[rows]).float(),
                           torch.from_numpy(Yn[rows]).float())
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    train_loader = loader(train_idx, min(args.batch_size, len(train_idx)), True)
    val_loader = loader(val_idx, max(1, len(val_idx)), False)

    model = SurrogateNet(args.hidden).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    out_dir = Path(__file__).resolve().parent
    weights_path, best_path = out_dir / "anyobj_model.pt", out_dir / "anyobj_model_best.pt"
    best_val, stale = float("inf"), 0

    for epoch in range(args.epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            loss = loss_fn(model(bx), by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = vc = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += loss_fn(model(bx), by).item() * bx.shape[0]
                vc += bx.shape[0]
        val_loss /= vc
        scheduler.step(val_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch+1:4d}  val={val_loss:.6f}  lr={lr:.1e}")

        if val_loss < best_val:
            best_val, stale = val_loss, 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  early stop at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), weights_path)

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xn[val_idx]).float().to(device)).cpu().numpy()
    mae = np.abs(pred * y_std + y_mean - Y[val_idx]).mean(0)

    np.savez(out_dir / "anyobj_normalizer.npz",
             x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
             feature_names=np.array(DESCRIPTOR_NAMES + FLOW_FEATURES),
             target_names=np.array(TARGET_NAMES))

    print(f"\nbest val {best_val:.6f}  |  val MAE Cd={mae[0]:.4f} Cl={mae[1]:.4f}")
    print(f"saved {weights_path.name}, {best_path.name}, anyobj_normalizer.npz")


def main():
    p = argparse.ArgumentParser(description="Train the any-OBJ Cd/Cl surrogate")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--voxel-res", type=int, default=32)
    train(p.parse_args())


if __name__ == "__main__":
    main()
