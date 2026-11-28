"""
FFD shape optimisation through the trained surrogate, checked by the solver.

Runs Adam on the FFD displacements to lower the predicted Cd. With
several checkpoints (an ensemble over seeds) it lowers the ensemble mean
plus kappa standard deviations instead. Every step keeps the parameters
inside the range of the training samples. The result is reported with
its distance to the nearest training sample and the parameters that sit
on a bound.

The optimised OBJ and a candidate file (same name, .json) are written,
then `verify` runs the shape and the baseline through the solver. The
change in Cd is only reported with that table. With --no-verify it is
labelled a surrogate prediction.

Usage:
    python optimize.py --weights out/shapeopt_model_best.pt --base-model ahmed25 \\
        --output opt.obj --binary build/3d_fluid_simulation_car
    python optimize.py --weights s0/shapeopt_model_best.pt s1/shapeopt_model_best.pt \\
        --base-model ahmed25 --output opt.obj --binary build/3d_fluid_simulation_car
    python optimize.py --weights out/shapeopt_model_best.pt --base-model ahmed25 --no-verify
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import verify
from ffd import FFDLattice
from ffd import compute_bbox
from ffd import load_obj
from ffd import save_obj
from train_shapeopt import load_checkpoint
from train_shapeopt import load_sobin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = PROJECT_ROOT / "build" / "3d_fluid_simulation_car"
UNVERIFIED = "surrogate prediction, not verified"

MODEL_OBJ_PATHS = {
    "ahmed25": PROJECT_ROOT / "simulation" / "assets" / "3d-files" / "ahmed_25deg_m.obj",
    "ahmed35": PROJECT_ROOT / "simulation" / "assets" / "3d-files" / "ahmed_35deg_m.obj",
    "car": PROJECT_ROOT / "simulation" / "assets" / "3d-files" / "car-model.obj",
}


# Model definition (must match train_shapeopt.py)


class SwiGLU(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_gate = torch.nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_up = torch.nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_down = torch.nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ShapeOptSurrogate(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.swiglu1 = SwiGLU(256, 512)
        self.fc2 = torch.nn.Linear(256, 128)
        self.swiglu2 = SwiGLU(128, 256)
        self.fc3 = torch.nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.swiglu1(h)
        h = self.fc2(h)
        h = self.swiglu2(h)
        return self.fc3(h)


# Surrogate, one checkpoint or an ensemble


class Surrogate:
    """Mean and spread of Cd over one or more trained members"""

    def __init__(self, members: list[tuple[torch.nn.Module, np.ndarray, np.ndarray]]) -> None:
        self.members = [
            (m, torch.from_numpy(np.asarray(mu)).float(), torch.from_numpy(np.asarray(sd)).float())
            for m, mu, sd in members
        ]

    def __len__(self) -> int:
        return len(self.members)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cd mean, Cd std and Cl mean for a batch of full input rows"""
        outs = torch.stack([m((x - mu) / sd) for m, mu, sd in self.members])
        cd_std = outs[..., 0].std(dim=0) if len(self) > 1 else torch.zeros_like(outs[0, :, 0])
        return outs[..., 0].mean(dim=0), cd_std, outs[..., 1].mean(dim=0)

    def objective(self, x: torch.Tensor, kappa: float) -> torch.Tensor:
        mean, std, _ = self.predict(x)
        return mean + kappa * std


def load_surrogate(paths: list[str]) -> tuple[Surrogate, list[dict]]:
    ckpts = [load_checkpoint(p) for p in paths]
    dims = {(c["input_dim"], c["num_ffd_params"]) for c in ckpts}
    if len(dims) != 1:
        raise ValueError(f"checkpoints disagree on input size: {sorted(dims)}")
    members = []
    for c in ckpts:
        model = ShapeOptSurrogate(c["input_dim"])
        model.load_state_dict(c["model_state_dict"])
        model.eval()
        members.append((model, c["feat_mean"], c["feat_std"]))
    return Surrogate(members), ckpts


# Guard rails


def free_mask(lattice: FFDLattice, symmetric: bool = True) -> np.ndarray:
    """Flat mask of the independent parameters under Y symmetry"""
    mask = np.zeros((lattice.nx, lattice.ny, lattice.nz, 3), dtype=bool)
    if not symmetric:
        mask[:] = True
        return mask.ravel()
    half = lattice.ny // 2
    mask[:, :half] = True
    if lattice.ny % 2:
        mask[:, half, :, 0] = True
        mask[:, half, :, 2] = True
    return mask.ravel()


def parameter_bounds(
    lattice: FFDLattice, train_x: np.ndarray | None, max_fraction: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-parameter range of the training samples, inside +/- max_fraction of the box"""
    n = lattice.get_num_params()
    lim = np.full(n, np.inf)
    if max_fraction is not None:
        lim = np.tile(max_fraction * lattice.extent, n // 3)
    if train_x is None:
        if max_fraction is None:
            raise ValueError("no training samples or --max-displacement, nothing bounds it")
        return -lim, lim
    lo = np.maximum(train_x.min(axis=0), -lim)
    hi = np.minimum(train_x.max(axis=0), lim)
    return lo, hi


def project(
    d: np.ndarray, lo: np.ndarray, hi: np.ndarray, lattice: FFDLattice, symmetric: bool = True
) -> np.ndarray:
    """Clip, then mirror for Y symmetry, which keeps the bounds"""
    d = np.clip(d, lo, hi)
    return lattice.enforce_symmetry(d, axis=1) if symmetric else d


def distance_report(d: np.ndarray, train_x: np.ndarray, scale: np.ndarray) -> dict:
    """Nearest training sample in the surrogate's normalised inputs"""
    t = train_x / scale
    x = d / scale
    dist = np.linalg.norm(t - x, axis=1)
    i = int(np.argmin(dist))
    pair = np.linalg.norm(t[:, None, :] - t[None, :, :], axis=-1)
    np.fill_diagonal(pair, np.inf)
    typical = float(np.median(pair.min(axis=1))) if len(t) > 1 else float("nan")
    return {
        "nearest_index": i,
        "distance": float(dist[i]),
        "median_train_nn": typical,
        "ratio": float(dist[i] / typical) if typical > 0 else float("nan"),
        "baseline_distance": float(np.linalg.norm(t, axis=1).min()),
        "samples": len(t),
    }


def bound_report(
    d: np.ndarray, lo: np.ndarray, hi: np.ndarray, mask: np.ndarray, tol: float = 0.01
) -> dict:
    """Free parameters within tol of their range from a bound"""
    span = hi - lo
    live = mask & (span > 0)
    at = live & ((d <= lo + tol * span) | (d >= hi - tol * span))
    return {"free": int(live.sum()), "at_bound": int(at.sum()), "indices": np.flatnonzero(at)}


def fit_report(
    surrogate: Surrogate, extra: np.ndarray, train_x: np.ndarray, train_y: np.ndarray, dist: dict
) -> list[dict]:
    """Surrogate against the solver at the nearest training sample and the baseline sample"""
    picks = [("nearest training sample", dist["nearest_index"])]
    picks += [("baseline sample", int(i)) for i in np.flatnonzero(~train_x.any(axis=1))[:1]]
    return [
        {
            "what": what,
            "index": i,
            "pred": predict_one(surrogate, train_x[i], extra)["cd"],
            "cd": float(train_y[i, 0]),
        }
        for what, i in picks
    ]


def calibration(surrogate: Surrogate, extra: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Held-out error of the ensemble mean and how often 2 std cover it"""
    inp = torch.from_numpy(np.hstack([x, np.tile(extra, (len(x), 1))])).float()
    with torch.no_grad():
        mean, std, _ = surrogate.predict(inp)
    err = (mean.numpy() - y[:, 0]).astype(np.float64)
    std = std.numpy().astype(np.float64)
    return {
        "samples": len(x),
        "mae": float(np.abs(err).mean()),
        "coverage_2std": float((np.abs(err) <= 2 * std).mean()),
        "mean_std": float(std.mean()),
    }


# Optimisation


def optimise(
    surrogate: Surrogate,
    lattice: FFDLattice,
    lo: np.ndarray,
    hi: np.ndarray,
    extra: np.ndarray,
    steps: int,
    lr: float,
    kappa: float = 0.0,
    symmetric: bool = True,
    log=print,
) -> tuple[np.ndarray, list[tuple]]:
    """Projected Adam on the displacements, from the baseline"""
    d = torch.zeros(lattice.get_num_params(), requires_grad=True)
    opt = torch.optim.Adam([d], lr=lr)
    extra_t = torch.from_numpy(np.asarray(extra, dtype=np.float32))
    trajectory = []
    for step in range(steps + 1):
        with torch.no_grad():
            d.copy_(torch.from_numpy(project(d.detach().numpy(), lo, hi, lattice, symmetric)))
        x = torch.cat([d, extra_t]).unsqueeze(0)
        loss = surrogate.objective(x, kappa)[0]
        with torch.no_grad():
            mean, std, cl = (v[0].item() for v in surrogate.predict(x))
        trajectory.append((step, loss.item(), mean, std, cl))
        if step % 20 == 0 or step == steps:
            log(f"{step:6d}  {loss.item():10.5f}  {mean:10.5f}  {std:8.5f}  {cl:9.5f}")
        if step == steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return project(d.detach().numpy().astype(np.float64), lo, hi, lattice, symmetric), trajectory


def predict_one(surrogate: Surrogate, d: np.ndarray, extra: np.ndarray) -> dict:
    x = torch.from_numpy(np.concatenate([d, extra])).float().unsqueeze(0)
    with torch.no_grad():
        mean, std, cl = (v[0].item() for v in surrogate.predict(x))
    return {"cd": mean, "cd_std": std if len(surrogate) > 1 else None, "cl": cl}


# Reporting


class Unverified(RuntimeError):
    """Cd change reported without a solver check"""


def summary(pred: dict, result: dict | None, no_verify: bool) -> list[str]:
    """Final lines, refused without a verification table or --no-verify"""
    b, c = pred["baseline_cd"], pred["cd"]
    line = f"Cd {b:.4f} -> {c:.4f} ({c - b:+.4f}, {100 * (c - b) / b:+.2f}%)"
    if result is None:
        if not no_verify:
            raise Unverified("no verification table, run verify or pass --no-verify")
        return [f"{UNVERIFIED}: {line}"]
    lines = [f"surrogate prediction: {line}"]
    shape = result["shapes"][0]
    for e in shape["rows"]:
        if e["status"] != "ok":
            lines.append(f"solver on {e['grid']}: {e['status']}")
            continue
        lines.append(
            f"solver on {e['grid']}: Cd {e['cd'] - e['delta']:.4f} -> {e['cd']:.4f} "
            f"({e['delta']:+.4f} ± {e['delta_ci95']:.4f}, {e['delta_pct']:+.2f}%), "
            f"{e['verdict']} (predicted {e['pred_delta']:+.4f}). "
            f"CdA/A0 {e['cda_delta']:+.4f} ± {e['cda_delta_ci95']:.4f}, "
            f"{e['cda_verdict']}"
            if e["delta_ci95"] is not None
            else f"solver on {e['grid']}: no interval on Cd, {e['verdict']}"
        )
    lines.append(f"verdict: {shape['overall']}")
    return lines


def flow_inputs(ckpt: dict, wind: float | None, reynolds: float | None, warn) -> np.ndarray:
    """Wind and Re inputs, the training values unless given"""
    mean, std = np.asarray(ckpt["feat_mean"]), np.asarray(ckpt["feat_std"])
    train = mean[-2:]
    # constant columns have std 1 and their value as mean
    constant = std[-2:] == 1.0
    vals = np.array(
        [train[0] if wind is None else wind, train[1] if reynolds is None else reynolds]
    )
    for name, v, t, c in zip(("wind speed", "Re"), vals, train, constant, strict=True):
        if c and not np.isclose(v, t):
            warn(f"{name} {v:g} is outside the training data, which used {t:g} only")
    return vals.astype(np.float64)


def run(args: argparse.Namespace) -> int:
    def warn(msg: str) -> None:
        print(f"WARNING: {msg}")

    obj_path = MODEL_OBJ_PATHS[args.base_model]
    vertices, faces = load_obj(str(obj_path))
    print(f"Base model: {args.base_model} ({obj_path}), {len(vertices)} vertices")

    surrogate, ckpts = load_surrogate(args.weights)
    ck = ckpts[0]
    proto = ck.get("protocol")
    print(f"Surrogate: {len(surrogate)} checkpoint(s), best val Cd MAE {ck.get('best_val_cd_mae')}")

    ffd_grid = tuple(args.ffd_grid or (proto or {}).get("ffd_grid") or (4, 3, 3))
    lattice = FFDLattice(*compute_bbox(vertices), n_control=ffd_grid)
    if lattice.get_num_params() != ck["num_ffd_params"]:
        print(
            f"Error: FFD grid {list(ffd_grid)} gives {lattice.get_num_params()} params "
            f"but the checkpoint expects {ck['num_ffd_params']}",
            file=sys.stderr,
        )
        return 1
    if proto and proto.get("base_model") != args.base_model:
        warn(f"surrogate was trained on {proto.get('base_model')}, not {args.base_model}")
    symmetric = (proto or {}).get("enforce_symmetry", True)

    train_x, train_y = ck.get("train_x"), ck.get("train_y")
    if train_x is None and args.data:
        train_x, train_y = load_sobin(args.data)
    if train_x is None:
        warn("no training samples in the checkpoint or --data, bounds are +/- --max-displacement")
    lo, hi = parameter_bounds(lattice, train_x, args.max_displacement)
    extra = flow_inputs(ck, args.wind_speed, args.reynolds, warn)

    kappa = args.kappa if args.kappa is not None else (1.0 if len(surrogate) > 1 else 0.0)
    if kappa and len(surrogate) == 1:
        warn("--kappa needs several checkpoints, one gives no spread")
    objective = "mean" if kappa == 0 else f"mean + {kappa:g} std over {len(surrogate)} members"
    if len(surrogate) > 1 and ck.get("val_x") is not None:
        same = all(np.array_equal(c.get("val_x"), ck["val_x"]) for c in ckpts)
        if same:
            cal = calibration(surrogate, extra, ck["val_x"], ck["val_y"])
            print(
                f"Ensemble on {cal['samples']} held-out samples: Cd MAE {cal['mae']:.4f}, "
                f"mean std {cal['mean_std']:.4f}, 2 std cover {100 * cal['coverage_2std']:.0f}%"
            )
            if cal["coverage_2std"] < 0.8:
                warn("the ensemble spread understates its held-out error")

    print(f"\nOptimising {objective}: {args.steps} steps, lr {args.lr}, FFD grid {list(ffd_grid)}")
    print(f"{'step':>6s}  {'objective':>10s}  {'Cd mean':>10s}  {'Cd std':>8s}  {'Cl':>9s}")
    d, _ = optimise(surrogate, lattice, lo, hi, extra, args.steps, args.lr, kappa, symmetric)

    zero = np.zeros_like(d)
    p_base, p_opt = predict_one(surrogate, zero, extra), predict_one(surrogate, d, extra)
    guard = {"bounds": bound_report(d, lo, hi, free_mask(lattice, symmetric))}
    print(f"\nGuard rails ({'training range' if train_x is not None else 'box only'}):")
    if guard["bounds"]["at_bound"]:
        warn(
            f"{guard['bounds']['at_bound']} of {guard['bounds']['free']} free parameters "
            "sit on a bound of the training range"
        )
    else:
        print(f"  no free parameter on a bound ({guard['bounds']['free']} free)")
    if train_x is not None:
        scale = np.asarray(ck["feat_std"])[: lattice.get_num_params()]
        dist = distance_report(d, np.asarray(train_x), scale)
        guard["distance"] = dist
        print(
            f"  nearest of {dist['samples']} training samples: #{dist['nearest_index']} at "
            f"{dist['distance']:.3f}, typical spacing {dist['median_train_nn']:.3f} "
            f"(ratio {dist['ratio']:.2f})"
        )
        if dist["ratio"] > 1.5:
            warn("the optimum is far from the training samples, expect a large surrogate error")
        if train_y is not None:
            guard["fit"] = fit_report(
                surrogate, extra, np.asarray(train_x), np.asarray(train_y), dist
            )
            limit = 2 * float(ck.get("best_val_cd_mae") or np.inf)
            for f in guard["fit"]:
                print(f"  {f['what']}: surrogate {f['pred']:.4f}, solver {f['cd']:.4f}")
                if abs(f["pred"] - f["cd"]) > limit:
                    warn(f"surrogate misses the {f['what']} by more than twice its val MAE")
    guard["bounds"]["indices"] = guard["bounds"]["indices"].tolist()

    output = Path(args.output)
    deformed = lattice.deform(vertices, d)
    save_obj(output, deformed, faces)
    candidate = {
        "kind": "shapeopt-candidate",
        "version": 1,
        "name": args.name or output.stem,
        "obj": str(output.resolve()),
        "base_model": args.base_model,
        "base_obj": str(obj_path),
        "ffd_grid": list(ffd_grid),
        "displacements": d.tolist(),
        "objective": objective,
        "weights": [str(Path(w).resolve()) for w in args.weights],
        "predicted": {
            "cd": p_opt["cd"],
            "cd_std": p_opt["cd_std"],
            "cl": p_opt["cl"],
            "baseline_cd": p_base["cd"],
            "baseline_cd_std": p_base["cd_std"],
        },
        "flow": {"wind_speed": float(extra[0]), "reynolds": float(extra[1])},
        "protocol": proto,
        "guard": guard,
    }
    cand_path = output.with_suffix(".json")
    cand_path.write_text(json.dumps(candidate, indent=2) + "\n")
    print(f"  wrote {output} and {cand_path}")

    result = None
    if not args.no_verify:
        binary = args.binary or DEFAULT_BINARY
        if not Path(binary).is_file():
            print(
                f"Error: no solver binary at {binary}, give --binary or pass --no-verify",
                file=sys.stderr,
            )
            return 2
        args.binary = Path(binary)
        out = Path(args.verify_out or output.with_name(output.stem + "_verify"))
        print(f"\nVerifying in the solver, output in {out}")
        result = verify.verify_files([cand_path], args, out)

    print()
    for line in summary(candidate["predicted"], result, args.no_verify):
        print(line)
    return 0


# CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optimise a shape through the surrogate, then check it in the solver"
    )
    parser.add_argument(
        "--weights", required=True, nargs="+", help="checkpoint(s), several for an ensemble"
    )
    parser.add_argument("--base-model", required=True, choices=sorted(MODEL_OBJ_PATHS))
    parser.add_argument("--output", default="optimized.obj", help="output OBJ path")
    parser.add_argument("--name", help="shape name in the verification table")
    parser.add_argument("--wind-speed", type=float, help="surrogate input (default: training)")
    parser.add_argument("--steps", type=int, default=200, help="optimisation steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate")
    parser.add_argument(
        "--ffd-grid",
        type=int,
        nargs=3,
        default=None,
        metavar=("NX", "NY", "NZ"),
        help="FFD control points (default: from the training data, else 4 3 3)",
    )
    parser.add_argument(
        "--max-displacement",
        type=float,
        default=0.15,
        help="cap as a fraction of the box, on top of the training range (default: 0.15)",
    )
    parser.add_argument("--data", help="training .sobin, for checkpoints without samples")
    parser.add_argument(
        "--kappa", type=float, help="std weight of the objective (default: 1 for an ensemble)"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help=f"skip the solver, output is a {UNVERIFIED}"
    )
    parser.add_argument("--verify-out", help="verification output directory")
    parser.add_argument("--force", action="store_true", help="rerun solver cases that are done")
    verify.add_run_arguments(parser)
    args = parser.parse_args(argv)

    for w in args.weights:
        if not Path(w).exists():
            print(f"Error: weights not found at {w}", file=sys.stderr)
            return 1
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
