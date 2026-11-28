"""
Compare the shipped super-resolution net with plain interpolation.

The net is run as the GLSL shader runs it (superres_upscale.comp), on the
whole field, with the weights and normalizer from simulation/assets. Needs
numpy only.

Two tasks, kept apart:
  real      time-averaged coarse solver run -> fine solver run of the case
  filtered  2x2x2 block mean of the fine run -> fine run, as gen_pairs.py
            builds the training pairs

Methods: nearest, trilinear, tricubic (4-point Lagrange), Catmull-Rom, the
net as the shader runs it, and the net with the skip connection it was
trained with (train_superres.trilinear_upsample).

Both grids are cell centred over the same box, so coarse cell i covers fine
cells 2i and 2i+1 and its centre sits a quarter of a coarse cell from each.

Usage:
    python evaluate_baselines.py \\
        --case sphere_re50 coarse/field_001800.vti fine/field_003600.vti \\
        --case ahmed25_re140 ... --out docs/validation/superres --plot
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_pairs import downsample_2x
from gen_pairs import downsample_2x_scalar

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "simulation" / "assets" / "sr_model.bin"
NORM = ROOT / "simulation" / "assets" / "sr_model_norm.bin"
LTWS_MAGIC = 0x4C545753

# 1D taps, offset to weight, for the children at -1/4 and +1/4 coarse cell
NEAREST = ({0: 1.0}, {0: 1.0})
LINEAR = ({-1: 0.25, 0: 0.75}, {0: 0.75, 1: 0.25})
CUBIC = (
    {-2: -0.0390625, -1: 0.2734375, 0: 0.8203125, 1: -0.0546875},
    {-1: -0.0546875, 0: 0.8203125, 1: 0.2734375, 2: -0.0390625},
)
CATMULL_ROM = (
    {-2: -0.0234375, -1: 0.2265625, 0: 0.8671875, 1: -0.0703125},
    {-1: -0.0703125, 0: 0.8671875, 1: 0.2265625, 2: -0.0234375},
)
# skip connections of the net, same form
SHADER_SKIP = ({0: 0.75, 1: 0.25}, {0: 0.25, 1: 0.75})
TRAIN_SKIP = ({-1: 0.75, 0: 0.25}, {0: 0.25, 1: 0.75})

INTERPOLANTS = {
    "nearest": NEAREST,
    "trilinear": LINEAR,
    "tricubic": CUBIC,
    "catmull_rom": CATMULL_ROM,
}
METHODS = [*INTERPOLANTS, "net_shader", "net_train_skip"]


# I/O


def read_vti(path: Path) -> dict[str, np.ndarray]:
    """Read every appended DataArray of a VTI written by writeVTI."""
    raw = Path(path).read_bytes()
    marker = b'<AppendedData encoding="raw">\n_'
    pos = raw.find(marker)
    if pos < 0:
        raise ValueError(f"{path}: no appended data")
    head = raw[:pos].decode("ascii", errors="replace")
    blob = raw[pos + len(marker) :]
    ext = re.search(r'WholeExtent="([^"]+)"', head)
    if not ext:
        raise ValueError(f"{path}: no WholeExtent")
    e = [int(v) for v in ext.group(1).split()]
    nx, ny, nz = e[1], e[3], e[5]
    out = {}
    for m in re.finditer(r"<DataArray ([^>]*)/>", head):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        comps = int(attrs.get("NumberOfComponents", 1))
        dtype = "<f4" if attrs["type"] == "Float32" else "<i4"
        off = int(attrs["offset"])
        size = struct.unpack_from("<Q", blob, off)[0]
        count = nx * ny * nz * comps
        if size != count * 4:
            raise ValueError(f"{path}: {attrs['Name']} has {size} bytes")
        arr = np.frombuffer(blob, dtype=dtype, count=count, offset=off + 8)
        shape = (nz, ny, nx, comps) if comps > 1 else (nz, ny, nx)
        out[attrs["Name"]] = arr.reshape(shape).copy()
    return out


def load_ltws(path: Path) -> list[np.ndarray]:
    """Tensors of an LTWS file in file order (sorted state dict keys)."""
    raw = Path(path).read_bytes()
    magic, _version, count = struct.unpack_from("<III", raw, 0)
    if magic != LTWS_MAGIC:
        raise ValueError(f"{path}: bad magic")
    off = 12
    tensors = []
    for _ in range(count):
        ndim = struct.unpack_from("<I", raw, off)[0]
        shape = struct.unpack_from(f"<{ndim}I", raw, off + 4)
        off += 4 + 4 * ndim
        n = int(np.prod(shape))
        tensors.append(np.frombuffer(raw, "<f4", n, off).reshape(shape).astype(np.float64))
        off += 4 * n
    return tensors


def load_norm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    v = np.fromfile(path, dtype="<f4").astype(np.float64)
    return v[:4], v[4:8]


# Upsampling


def upsample_axis(a: np.ndarray, axis: int, taps: tuple[dict, dict]) -> np.ndarray:
    """2x upsample along one axis, edge values repeated past the ends."""
    n = a.shape[axis]
    reach = max(abs(o) for t in taps for o in t)
    pad = [(0, 0)] * a.ndim
    pad[axis] = (reach, reach)
    p = np.pad(a, pad, mode="edge")
    children = []
    for t in taps:
        acc = np.zeros_like(a, dtype=np.float64)
        for o, w in t.items():
            acc = acc + w * np.take(p, np.arange(n) + reach + o, axis=axis)
        children.append(acc)
    out = np.stack(children, axis=axis + 1)
    shape = list(a.shape)
    shape[axis] = 2 * n
    return out.reshape(shape)


def upsample(field: np.ndarray, taps: tuple[dict, dict]) -> np.ndarray:
    """Separable 2x upsampling of a (nz, ny, nx, ...) field."""
    for axis in range(3):
        field = upsample_axis(field, axis, taps)
    return field


def patches(c4: np.ndarray) -> np.ndarray:
    """3x3x3 neighbourhoods as the shader gathers them, (nz, ny, nx, 108)."""
    nz, ny, nx, _ = c4.shape
    p = np.pad(c4, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="edge")
    cols = [
        p[1 + dz : 1 + dz + nz, 1 + dy : 1 + dy + ny, 1 + dx : 1 + dx + nx]
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    ]
    return np.concatenate(cols, axis=-1)


def net_upscale(
    c4: np.ndarray,
    params: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    skip: tuple[dict, dict] = SHADER_SKIP,
) -> np.ndarray:
    """The shader's forward pass on a (nz, ny, nx, 4) field, (2nz, 2ny, 2nx, 4) out."""
    b0, w0, b1, w1, b2, w2, b3, w3 = params
    nz, ny, nx, _ = c4.shape
    x = patches((c4 - mean) / std).reshape(-1, 108)
    h = np.maximum(x @ w0.T + b0, 0.0)
    h = np.maximum(h @ w1.T + b1, 0.0)
    h = np.maximum(h @ w2.T + b2, 0.0)
    corr = (h @ w3.T + b3).reshape(nz, ny, nx, 2, 2, 2, 4)
    vol = x.reshape(nz, ny, nx, 3, 3, 3, 4)
    w = np.zeros((2, 3))
    for k, t in enumerate(skip):
        for o, v in t.items():
            w[k, o + 1] = v
    base = np.einsum("ai,bj,ck,zyxijkl->zyxabcl", w, w, w, vol, optimize=True)
    out = (base + corr) * std + mean
    return out.transpose(0, 3, 1, 4, 2, 5, 6).reshape(2 * nz, 2 * ny, 2 * nx, 4)


def run_method(name: str, c4: np.ndarray, net: tuple) -> np.ndarray:
    if name in INTERPOLANTS:
        return upsample(c4[..., :3], INTERPOLANTS[name])
    params, mean, std = net
    skip = SHADER_SKIP if name == "net_shader" else TRAIN_SKIP
    return net_upscale(c4, params, mean, std, skip)[..., :3]


# Field operators and masks


def gradient(u: np.ndarray) -> np.ndarray:
    """Central-difference velocity gradient, g[..., i, j] = d u_i / d x_j."""
    g = np.zeros((*u.shape[:3], 3, 3))
    for i in range(3):
        dz, dy, dx = np.gradient(u[..., i])
        g[..., i, 0], g[..., i, 1], g[..., i, 2] = dx, dy, dz
    return g


def divergence(u: np.ndarray) -> np.ndarray:
    g = gradient(u)
    return g[..., 0, 0] + g[..., 1, 1] + g[..., 2, 2]


def vorticity(u: np.ndarray) -> np.ndarray:
    g = gradient(u)
    return np.stack(
        [g[..., 2, 1] - g[..., 1, 2], g[..., 0, 2] - g[..., 2, 0], g[..., 1, 0] - g[..., 0, 1]],
        axis=-1,
    )


def dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    """Grow a mask by `steps` cells in the 26-neighbour sense."""
    m = mask.copy()
    for _ in range(steps):
        p = np.pad(m, 1)
        grown = np.zeros_like(m)
        for dz in range(3):
            for dy in range(3):
                for dx in range(3):
                    grown |= p[dz : dz + m.shape[0], dy : dy + m.shape[1], dx : dx + m.shape[2]]
        m = grown
    return m


def regions(solid: np.ndarray, near: int = 6) -> dict[str, np.ndarray]:
    """Masks on the fine grid, without the outlet sponge (last nx/8 cells)."""
    nx = solid.shape[2]
    s = solid > 0
    fluid = ~s
    keep = np.zeros_like(s)
    keep[:, :, : nx - nx // 8] = True
    interior = fluid.copy()
    for ax in range(3):
        interior &= ~np.roll(s, 1, ax) & ~np.roll(s, -1, ax)
    edge = np.zeros_like(s)
    edge[[0, -1]] = edge[:, [0, -1]] = edge[:, :, [0, -1]] = True
    interior &= ~edge
    zz, yy, xx = np.nonzero(s)
    out = {"all": fluid & keep, "interior": interior & keep, "solid": s}
    out["near"] = fluid & keep & dilate(s, near)
    if len(xx):
        length = xx.max() - xx.min() + 1
        wy = (yy.max() - yy.min() + 1) // 2
        wz = (zz.max() - zz.min() + 1) // 2
        wake = np.zeros_like(s)
        wake[
            max(zz.min() - wz, 0) : zz.max() + wz + 1,
            max(yy.min() - wy, 0) : yy.max() + wy + 1,
            xx.max() + 1 : xx.max() + 1 + 2 * length,
        ] = True
        out["wake"] = fluid & keep & wake
    else:
        out["wake"] = np.zeros_like(s)
    return out


def train_like(coarse_solid: np.ndarray) -> np.ndarray:
    """Fine cells gen_pairs would keep: parent's 3x3x3 neighbourhood all fluid."""
    s = coarse_solid > 0
    ok = ~dilate(s, 1)
    ok[[0, -1]] = ok[:, [0, -1]] = ok[:, :, [0, -1]] = False
    return ok.repeat(2, 0).repeat(2, 1).repeat(2, 2)


# Metrics with a block bootstrap


def block_ids(shape: tuple[int, int, int], block: int) -> np.ndarray:
    nz, ny, nx = shape
    bz, by, bx = np.meshgrid(
        np.arange(nz) // block, np.arange(ny) // block, np.arange(nx) // block, indexing="ij"
    )
    nby, nbx = -(-ny // block), -(-nx // block)
    return (bz * nby + by) * nbx + bx


def block_sums(columns: list[np.ndarray], mask: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Per-block sums of each column over the mask, (blocks, columns)."""
    n = int(ids.max()) + 1
    sel = ids[mask]
    return np.stack([np.bincount(sel, c[mask], minlength=n) for c in columns], axis=1)


def rel_l2(s: np.ndarray) -> np.ndarray:
    """Columns [sum e^2, sum ref^2] -> ||e|| / ||ref||."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(s[..., 0] / s[..., 1])


def ratio_of_sums(s: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return s[..., 0] / s[..., 1]


def resample_counts(n_blocks: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=n_boot)


def bootstrap(sums: list[np.ndarray], stat, counts: list[np.ndarray]) -> tuple[float, float, float]:
    """Point value and 95% interval, blocks resampled within each case."""
    point = stat(sum(s.sum(axis=0) for s in sums))
    draws = stat(sum(c @ s for c, s in zip(counts, sums, strict=True)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def case_columns(
    pred: dict[str, np.ndarray], ref: np.ndarray, masks: dict, u_inf: float
) -> dict[str, dict[str, tuple[list[np.ndarray], str, str]]]:
    """For each method and metric: (columns, mask name, statistic)."""
    free = np.array([u_inf, 0.0, 0.0])
    ref_vort = vorticity(ref)
    ref_grad = gradient(ref)
    dke_ref = 0.5 * ((ref - free) ** 2).sum(-1)
    out = {}
    for name, u in pred.items():
        e2 = ((u - ref) ** 2).sum(-1)
        r2 = (ref**2).sum(-1)
        vort = vorticity(u)
        grad = gradient(u)
        ev2 = ((vort - ref_vort) ** 2).sum(-1)
        rv2 = (ref_vort**2).sum(-1)
        eg2 = ((grad - ref_grad) ** 2).sum((-1, -2))
        rg2 = (ref_grad**2).sum((-1, -2))
        dke = 0.5 * ((u - free) ** 2).sum(-1)
        speed = np.sqrt((u**2).sum(-1)) / u_inf
        one = np.ones_like(e2)
        cols = {
            "vel_all": ([e2, r2], "all", "rel_l2"),
            "vel_near": ([e2, r2], "near", "rel_l2"),
            "vel_wake": ([e2, r2], "wake", "rel_l2"),
            "grad_all": ([eg2, rg2], "interior", "rel_l2"),
            "vort_all": ([ev2, rv2], "interior", "rel_l2"),
            "vort_near": ([ev2, rv2], "near_interior", "rel_l2"),
            "dke_err": ([dke - dke_ref, dke_ref], "all", "ratio"),
            "div_abs": ([np.abs(divergence(u)) / u_inf, one], "interior", "ratio"),
            "solid_speed": ([speed, one], "solid", "ratio"),
            "solid_leak": ([(speed > 0.1).astype(float), one], "solid", "ratio"),
        }
        if "train_like" in masks:
            cols["vel_train_like"] = ([e2, r2], "train_like", "rel_l2")
        out[name] = cols
    out["reference"] = {
        "div_abs": ([np.abs(divergence(ref)) / u_inf, np.ones_like(r2)], "interior", "ratio"),
    }
    return out


STATS = {"rel_l2": rel_l2, "ratio": ratio_of_sums}


def coarse_input(coarse: dict[str, np.ndarray], key: str) -> np.ndarray:
    vel = coarse[key].astype(np.float64)
    return np.concatenate([vel, coarse["rho"][..., None].astype(np.float64)], axis=-1)


def filtered_input(fine: dict[str, np.ndarray], key: str) -> tuple[np.ndarray, np.ndarray]:
    vel, solid = downsample_2x(fine[key], fine["solid"])
    rho = downsample_2x_scalar(fine["rho"])
    return np.concatenate([vel, rho[..., None]], axis=-1).astype(np.float64), solid


def evaluate_task(
    c4: np.ndarray, ref: np.ndarray, fine_solid: np.ndarray, coarse_solid: np.ndarray, args, net
) -> tuple[dict, dict]:
    masks = regions(fine_solid)
    masks["near_interior"] = masks["near"] & masks["interior"]
    masks["train_like"] = train_like(coarse_solid) & masks["all"]
    preds = {m: run_method(m, c4, net) for m in METHODS}
    ids = block_ids(fine_solid.shape, args.block)
    cols = case_columns(preds, ref.astype(np.float64), masks, args.u_inf)
    sums = {
        meth: {k: (block_sums(c, masks[mk], ids), st) for k, (c, mk, st) in mc.items()}
        for meth, mc in cols.items()
    }
    return sums, preds


def summarise(case_sums: list[dict], args) -> dict:
    """Point and interval per method and metric, plus net / trilinear ratios."""
    rng = np.random.default_rng(args.seed)
    n_blocks = [len(cs["trilinear"]["vel_all"][0]) for cs in case_sums]
    counts = [resample_counts(n, args.boot, rng) for n in n_blocks]
    out = {}
    for meth in case_sums[0]:
        out[meth] = {}
        for metric, (_, st) in case_sums[0][meth].items():
            parts = [cs[meth][metric][0] for cs in case_sums]
            out[meth][metric] = bootstrap(parts, STATS[st], counts)
    for meth in ("net_shader", "net_train_skip", "tricubic"):
        for metric in ("vel_all", "vel_near", "vel_wake", "vort_all"):
            a = [cs[meth][metric][0] for cs in case_sums]
            b = [cs["trilinear"][metric][0] for cs in case_sums]
            paired = [np.concatenate([x, y], axis=1) for x, y in zip(a, b, strict=True)]

            def ratio(s):
                return rel_l2(s[..., :2]) / rel_l2(s[..., 2:])

            out[meth][f"{metric}_vs_trilinear"] = bootstrap(paired, ratio, counts)
    return out


def fmt(v: tuple[float, float, float], scale: float = 1.0, digits: int = 3) -> str:
    p, lo, hi = (x * scale for x in v)
    return f"{p:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


TABLE_ROWS = [
    ("vel_all", "velocity rel L2, all fluid", 1.0),
    ("vel_near", "velocity rel L2, near body", 1.0),
    ("vel_wake", "velocity rel L2, wake", 1.0),
    ("vel_train_like", "velocity rel L2, cells like training", 1.0),
    ("grad_all", "gradient rel L2", 1.0),
    ("vort_all", "vorticity rel L2", 1.0),
    ("vort_near", "vorticity rel L2, near body", 1.0),
    ("dke_err", "disturbance KE rel error", 1.0),
    ("div_abs", "mean abs div / U (x1e3)", 1e3),
    ("solid_speed", "mean speed in solid / U", 1.0),
    ("solid_leak", "solid cells above 0.1 U", 1.0),
]


def table(summary: dict) -> str:
    cols = METHODS
    lines = ["| Metric | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for key, label, scale in TABLE_ROWS:
        if key not in summary["trilinear"]:
            continue
        cells = [fmt(summary[m][key], scale) for m in cols]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    ref = summary["reference"]["div_abs"]
    lines.append(f"\nReference field mean abs div / U (x1e3): {fmt(ref, 1e3)}")
    lines.append("\nRatio to trilinear (below 1 is better than trilinear):\n")
    lines.append("| Method | vel all | vel near | vel wake | vort all |")
    lines.append("|---|---|---|---|---|")
    for m in ("net_shader", "net_train_skip", "tricubic"):
        r = [fmt(summary[m][f"{k}_vs_trilinear"]) for k in ("vel_all", "vel_near", "vel_wake")]
        r.append(fmt(summary[m]["vort_all_vs_trilinear"]))
        lines.append(f"| {m} | " + " | ".join(r) + " |")
    return "\n".join(lines)


def plot(results: dict, slices: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = [t for t in ("real_mean", "filtered_mean") if t in results]
    cases = list(results[tasks[0]]["cases"])
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    meths = [("net_shader", "#257de0"), ("net_train_skip", "#f47230"), ("tricubic", "#00b894")]
    for ax, task in zip(axes, tasks, strict=True):
        x = np.arange(len(cases) + 1)
        for j, (m, color) in enumerate(meths):
            vals = [results[task]["cases"][c][m]["vel_all_vs_trilinear"] for c in cases]
            vals.append(results[task]["pooled"][m]["vel_all_vs_trilinear"])
            p = np.array([v[0] for v in vals])
            err = np.array([[v[0] - v[1], v[2] - v[0]] for v in vals]).T
            ax.bar(
                x + (j - 1) * 0.26,
                p,
                0.24,
                yerr=err,
                color=color,
                label=m,
                capsize=2,
                error_kw={"elinewidth": 1, "ecolor": "#191919"},
            )
        ax.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
        ax.grid(axis="y", color="#efefef", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(x, [*cases, "pooled"], rotation=30, ha="right", fontsize=8)
        ax.set_title(task.replace("_", " "), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("velocity rel L2 error / trilinear\n(below 1 beats trilinear)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "error_ratio.png", dpi=150)
    plt.close(fig)

    if slices:
        name, panels = next(iter(slices.items()))
        fig, axes = plt.subplots(
            len(panels), 1, figsize=(4.6, 2.0 * len(panels)), layout="constrained"
        )
        errs = [np.abs(v) for k, v in panels.items() if k != "reference ux / U"]
        lim = float(np.nanpercentile(np.concatenate([e.ravel() for e in errs]), 99.5))
        for ax, (title, img) in zip(axes, panels.items(), strict=True):
            if title == "reference ux / U":
                im = ax.imshow(img, origin="lower", cmap="Blues")
            else:
                im = ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"{name}: {title}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(
                im, ax=ax, fraction=0.03, extend="neither" if "reference" in title else "both"
            )
        fig.savefig(out / "error_slice.png", dpi=150)
        plt.close(fig)


def shown(v: object) -> str:
    """Argument as text, with paths in the repo made relative."""
    if isinstance(v, Path) and v.resolve().is_relative_to(ROOT):
        return str(v.resolve().relative_to(ROOT))
    return str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--case",
        nargs=3,
        action="append",
        required=True,
        metavar=("NAME", "COARSE_VTI", "FINE_VTI"),
        help="coarse and fine VTI dumps of one case, written with --vtk-stats",
    )
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--norm", type=Path, default=NORM)
    ap.add_argument("--u-inf", type=float, default=0.05, help="inlet velocity, lattice units")
    ap.add_argument("--block", type=int, default=8, help="bootstrap block edge, fine cells")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="write results.json, tables.md and plots here")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--slice", metavar="NAME", help="case for the mid-plane plot (default first)")
    args = ap.parse_args(argv)

    mean, std = load_norm(args.norm)
    net = (load_ltws(args.weights), mean, std)
    tasks = {"real_mean": {}, "filtered_mean": {}, "filtered_instant": {}}
    extra = {}
    slices = {}
    for name, cpath, fpath in args.case:
        coarse, fine = read_vti(Path(cpath)), read_vti(Path(fpath))
        if "mean_velocity" not in coarse or "mean_velocity" not in fine:
            raise SystemExit(f"{name}: run the solver with --vtk-stats")
        fz, fy, fx = fine["solid"].shape
        if coarse["solid"].shape != (fz // 2, fy // 2, fx // 2):
            raise SystemExit(f"{name}: fine grid must be twice the coarse grid")
        c4 = coarse_input(coarse, "mean_velocity")
        filt, filt_solid = filtered_input(fine, "mean_velocity")
        fluid = (coarse["solid"] == 0) & (filt_solid == 0)
        e2 = ((c4[..., :3] - filt[..., :3]) ** 2).sum(-1)[fluid].sum()
        r2 = (filt[..., :3] ** 2).sum(-1)[fluid].sum()
        extra[name] = {
            "coarse_vs_filtered_rel_l2": float(np.sqrt(e2 / r2)),
            "solid_cells_coarse": int((coarse["solid"] > 0).sum()),
            "solid_cells_fine": int((fine["solid"] > 0).sum()),
            "solid_cells_filtered": int((filt_solid > 0).sum()),
        }
        jobs = {
            "real_mean": (c4, fine["mean_velocity"], coarse["solid"]),
            "filtered_mean": (filt, fine["mean_velocity"], filt_solid),
            "filtered_instant": (filtered_input(fine, "velocity")[0], fine["velocity"], filt_solid),
        }
        for task, (inp, ref, csolid) in jobs.items():
            sums, preds = evaluate_task(inp, ref, fine["solid"], csolid, args, net)
            tasks[task][name] = sums
            if task == "real_mean" and name == (args.slice or args.case[0][0]):
                z = fz // 2
                u = args.u_inf
                ref = np.where(fine["solid"][..., None] > 0, np.nan, ref)
                slices[name] = {
                    "reference ux / U": ref[z, :, :, 0] / u,
                    "trilinear error / U": (preds["trilinear"][z, :, :, 0] - ref[z, :, :, 0]) / u,
                    "net (shader) error / U": (preds["net_shader"][z, :, :, 0] - ref[z, :, :, 0])
                    / u,
                }
        print(f"{name}: done", flush=True)

    results = {}
    report = []
    for task, per_case in tasks.items():
        results[task] = {"cases": {}, "pooled": summarise(list(per_case.values()), args)}
        for name, sums in per_case.items():
            results[task]["cases"][name] = summarise([sums], args)
            report.append(f"\n### {task}, {name}\n\n{table(results[task]['cases'][name])}")
        report.append(f"\n### {task}, pooled\n\n{table(results[task]['pooled'])}")
    report.append("\n### Coarse run against the block mean of the fine run\n")
    report.extend(f"- {k}: {json.dumps(v)}" for k, v in extra.items())
    text = "\n".join(report)
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        meta = {"args": {k: shown(v) for k, v in vars(args).items()}, "extra": extra}
        (args.out / "results.json").write_text(json.dumps({**meta, **results}, indent=1))
        (args.out / "tables.md").write_text(text + "\n")
        if args.plot:
            plot(results, slices, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
