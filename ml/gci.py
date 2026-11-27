"""
Grid convergence index for a study's results.

Groups the results of one case at three or more grid resolutions and
applies the procedure of Celik et al. (2008), "Procedure for estimation
and reporting of uncertainty due to discretization in CFD applications",
J. Fluids Eng. 130(7) 078001. That gives the observed order p by
fixed-point iteration, Richardson extrapolation, relative errors and the
GCI with Fs = 1.25.

No order is reported when the grid changes are within the runs' own noise,
when convergence is oscillatory or divergent, or when p falls outside
0.5 to 2 times the formal order. The reason is printed instead.

Usage:
    python ml/gci.py build/studies/ahmed25/results.json
    python ml/gci.py results.csv --value cd_corr_mean --no-sem
    python ml/gci.py results.json --json gci.json --plot gci.png
"""

import argparse
import csv
import itertools
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Celik et al. recommend refinement ratios above this
R_MIN = 1.3
# relative change treated as no change between grids
FLAT = 1e-12


@dataclass(frozen=True)
class Options:
    fs: float = 1.25
    formal_order: float = 2.0
    p_range: tuple[float, float] = (0.5, 2.0)  # times the formal order
    k: float = 2.0  # coverage factor on each run's standard error
    asymptotic_tol: float = 0.05


def observed_order(
    phi: Sequence[float], r21: float, r32: float, tol: float = 1e-12, max_iter: int = 500
) -> tuple[float, int, bool]:
    """Celik eq. 3 by fixed-point iteration, returns (p, iterations, converged)"""
    e21 = phi[1] - phi[0]
    e32 = phi[2] - phi[1]
    s = math.copysign(1.0, e32 / e21)
    ln_ratio = math.log(abs(e32 / e21))
    ln_r21 = math.log(r21)
    p = abs(ln_ratio) / ln_r21
    for i in range(1, max_iter + 1):
        try:
            q = math.log((r21**p - s) / (r32**p - s))
        except (ValueError, ZeroDivisionError, OverflowError):
            return math.nan, i, False
        new = abs(ln_ratio + q) / ln_r21
        if abs(new - p) <= tol * max(1.0, new):
            return new, i, True
        p = new
    return p, max_iter, False


def extrapolate(fine: float, coarse: float, r: float, p: float) -> float:
    """Celik eq. 4"""
    rp = r**p
    return (rp * fine - coarse) / (rp - 1)


def estimates(phi: Sequence[float], r21: float, r32: float, p: float, fs: float = 1.25) -> dict:
    """Celik eqs. 4 to 7 for a given order, plus Roache's coarse-grid GCI"""
    rp21, rp32 = r21**p, r32**p
    ext21 = extrapolate(phi[0], phi[1], r21, p)
    ea21 = abs((phi[0] - phi[1]) / phi[0])
    ea32 = abs((phi[1] - phi[2]) / phi[1])
    return {
        "extrapolated": ext21,
        "extrapolated32": extrapolate(phi[1], phi[2], r32, p),
        "e_a21": ea21,
        "e_ext21": abs((ext21 - phi[0]) / ext21),
        "gci_fine21": fs * ea21 / (rp21 - 1),
        "gci_coarse21": fs * ea21 * rp21 / (rp21 - 1),
        "gci_fine32": fs * ea32 / (rp32 - 1),
    }


def _classify(phi: Sequence[float], r21: float, r32: float) -> tuple[str, float | None]:
    e21 = phi[1] - phi[0]
    e32 = phi[2] - phi[1]
    scale = max(abs(v) for v in phi) or 1.0
    if abs(e21) <= FLAT * scale or abs(e32) <= FLAT * scale:
        return "flat", None
    ratio = e32 / e21
    if ratio < 0:
        return "oscillatory", ratio
    # p -> 0 gives this ratio, below it the differences grow under refinement
    if ratio <= math.log(r32) / math.log(r21):
        return "divergent", ratio
    return "monotone", ratio


def _ext_sensitivity(phi: list[float], r21: float, r32: float) -> list[float] | None:
    """d(extrapolated)/d(phi_i) by central differences, None if the order breaks"""
    step = 1e-4 * min(abs(phi[1] - phi[0]), abs(phi[2] - phi[1]))
    grads = []
    for i in range(3):
        vals = []
        for sign in (1, -1):
            trial = list(phi)
            trial[i] += sign * step
            if _classify(trial, r21, r32)[0] != "monotone":
                return None
            p, _, ok = observed_order(trial, r21, r32)
            if not ok:
                return None
            vals.append(extrapolate(trial[0], trial[1], r21, p))
        grads.append((vals[0] - vals[1]) / (2 * step))
    return grads


def _g(v: float) -> str:
    return f"{v:.4g}"


def assess(
    phi: Sequence[float],
    r21: float,
    r32: float,
    sem: Sequence[float | None] | None = None,
    opts: Options = Options(),
) -> dict:
    """Three-grid estimate, fine grid first, with a reason when it is refused"""
    phi = [float(v) for v in phi]
    sem = list(sem) if sem is not None else [None, None, None]
    e21, e32 = phi[1] - phi[0], phi[2] - phi[1]
    kind, ratio = _classify(phi, r21, r32)
    out: dict[str, Any] = {
        "phi": phi,
        "sem": sem,
        "r21": r21,
        "r32": r32,
        "eps21": e21,
        "eps32": e32,
        "ratio": ratio,
        "convergence": kind,
        "ok": False,
        "reason": None,
        "order": None,
        "extrapolated": None,
        "extrapolated32": None,
        "e_a21": None,
        "e_ext21": None,
        "gci_fine21": None,
        "gci_coarse21": None,
        "gci_fine32": None,
        "asymptotic": None,
        "uncertainty": None,
        "diagnostic": {},
    }

    def refuse(reason: str) -> dict:
        out["reason"] = reason
        return out

    if kind == "flat":
        return refuse(f"no change between grids (eps21={_g(e21)}, eps32={_g(e32)})")

    p, iterations, converged = observed_order(phi, r21, r32)
    out["diagnostic"] = {"p_raw": p if math.isfinite(p) else None, "iterations": iterations}

    if all(s is not None for s in sem):
        n21 = opts.k * math.hypot(sem[0], sem[1])
        n32 = opts.k * math.hypot(sem[1], sem[2])
        if abs(e21) <= n21 or abs(e32) <= n32:
            return refuse(
                f"grid changes are within noise: |eps21|={_g(abs(e21))} vs {_g(n21)}, "
                f"|eps32|={_g(abs(e32))} vs {_g(n32)} ({opts.k:g} x combined sem)"
            )
    if kind == "oscillatory":
        return refuse(f"oscillatory convergence, eps32/eps21={_g(ratio)}")
    if kind == "divergent":
        return refuse(f"divergent, differences grow under refinement (eps32/eps21={_g(ratio)})")
    if not converged:
        return refuse(f"order iteration did not converge in {iterations} steps")
    lo, hi = (f * opts.formal_order for f in opts.p_range)
    if not lo - 1e-9 <= p <= hi + 1e-9:
        return refuse(f"observed order p={p:.3g} outside [{lo:g}, {hi:g}]")
    if phi[0] == 0 or phi[1] == 0:
        return refuse("a grid value is zero, relative errors are undefined")

    est = estimates(phi, r21, r32, p, opts.fs)
    gci21 = est["gci_fine21"]
    gci_ratio = est["gci_fine32"] / (r21**p * gci21)

    u_num = gci21 * abs(phi[0])
    u_stat = opts.k * sem[0] if sem[0] is not None else None
    ext_stat = None
    if all(s is not None for s in sem):
        grads = _ext_sensitivity(phi, r21, r32)
        if grads is not None:
            ext_stat = opts.k * math.sqrt(
                sum((g * s) ** 2 for g, s in zip(grads, sem, strict=True))
            )

    out.update(
        ok=True,
        order=p,
        **est,
        asymptotic={
            "gci_ratio": gci_ratio,
            "gci_ratio_ok": abs(gci_ratio - 1) <= opts.asymptotic_tol,
            "formal_order": opts.formal_order,
            "p_over_formal": p / opts.formal_order,
        },
        uncertainty={
            "value": phi[0],
            "numerical": u_num,
            "statistical": u_stat,
            "combined": math.hypot(u_num, u_stat or 0.0),
            "extrapolated_statistical": ext_stat,
            "k": opts.k,
        },
    )
    return out


def spacing(points: list[dict], ratio_from: str = "cells", dims: int = 3) -> list[float]:
    """Representative cell size h for each point, up to a common factor"""
    if ratio_from == "cells":
        return [p["cells"] ** (-1.0 / dims) for p in points]
    return [1.0 / p["char_length"] for p in points]


def analyse(
    points: list[dict], ratio_from: str = "cells", dims: int = 3, opts: Options = Options()
) -> dict:
    """GCI for one group of points, each with a value, cells and optional sem"""
    warnings: list[str] = []
    result: dict[str, Any] = {"ratio_from": ratio_from, "dims": dims, "warnings": warnings}
    usable = [p for p in points if p.get(ratio_from) and p["value"] is not None]
    if len(usable) < len(points):
        warnings.append(f"{len(points) - len(usable)} result(s) without {ratio_from} left out")
    h = spacing(usable, ratio_from, dims)
    order = sorted(range(len(usable)), key=lambda i: h[i])
    pts = [{**usable[i], "h": h[i] / h[order[0]] if order else 1.0} for i in order]
    result["grids"] = pts
    result["triplets"] = []
    result["primary"] = None

    if len(pts) < 3:
        result["reason"] = f"needs results on 3 grids, got {len(pts)}"
        return result
    for a, b in itertools.pairwise(pts):
        if math.isclose(a["h"], b["h"], rel_tol=1e-9):
            result["reason"] = f"grids {a['grid']} and {b['grid']} have the same resolution"
            return result

    for i in range(len(pts) - 2):
        f, m, c = pts[i : i + 3]
        r21, r32 = m["h"] / f["h"], c["h"] / m["h"]
        for r, pair in ((r21, (f, m)), (r32, (m, c))):
            if r < R_MIN:
                warnings.append(
                    f"r={r:.3f} between {pair[0]['grid']} and {pair[1]['grid']} is below {R_MIN}"
                )
        result["triplets"].append(
            assess(
                [f["value"], m["value"], c["value"]],
                r21,
                r32,
                [f.get("sem"), m.get("sem"), c.get("sem")],
                opts,
            )
        )

    if ratio_from == "cells" and all(p.get("char_length") for p in pts):
        alt = spacing(pts, "char_length")
        for a, b, ha, hb in zip(pts, pts[1:], alt, alt[1:], strict=False):
            r_cells, r_len = b["h"] / a["h"], hb / ha
            if abs(r_len / r_cells - 1) > 0.05:
                warnings.append(
                    f"{a['grid']} to {b['grid']}: r={r_cells:.3f} from cells but "
                    f"{r_len:.3f} from char_length"
                )
    re = [p.get("reynolds") for p in pts]
    if all(re) and (max(re) - min(re)) > 0.01 * max(re):
        shown = ", ".join(f"{v:.4g}" for v in re)
        warnings.append(f"Re differs across grids ({shown}), so they are not one flow")
    if any(p.get("sem") is None for p in pts[:3]):
        warnings.append("no standard error on some grids, noise is not checked")

    primary = result["triplets"][0]
    result["primary"] = primary
    result["reason"] = primary["reason"]
    good = [t["order"] for t in result["triplets"] if t["ok"]]
    if primary["ok"] and len(good) > 1:
        spread = (max(good) - min(good)) / primary["order"]
        primary["asymptotic"]["p_spread"] = spread
    return result


# Reading results


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def load_rows(path: Path) -> list[dict]:
    """Rows from a study's results.json or results.csv"""
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    data = json.loads(path.read_text())
    rows = data["rows"] if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of rows")
    return rows


def _cells(row: dict) -> float | None:
    cells = _num(row.get("cells"))
    if cells is None and row.get("grid"):
        try:
            cells = float(math.prod(int(n) for n in str(row["grid"]).split("x")))
        except ValueError:
            return None
    return cells


def _group_key(row: dict) -> str:
    if row.get("group"):
        return str(row["group"])
    params = row.get("params")
    if isinstance(params, dict):
        rest = {k: v for k, v in params.items() if k != "grid"}
        return json.dumps(rest, sort_keys=True)
    return str(row.get("model") or "all")


def group_points(rows: list[dict], value: str, sem: str | None) -> tuple[dict, list[str]]:
    """Points per group, plus the rows that were left out and why"""
    groups: dict[str, list[dict]] = {}
    skipped = []
    for row in rows:
        name = row.get("name") or row.get("grid") or "?"
        status = row.get("status", "ok")
        if status != "ok":
            skipped.append(f"{name}: status {status}")
            continue
        v = _num(row.get(value))
        if v is None:
            skipped.append(f"{name}: no {value}")
            continue
        groups.setdefault(_group_key(row), []).append(
            {
                "name": name,
                "grid": row.get("grid"),
                "cells": _cells(row),
                "char_length": _num(row.get("char_length")),
                "reynolds": _num(row.get("reynolds")),
                "value": v,
                "sem": _num(row.get(sem)) if sem else None,
            }
        )
    return groups, skipped


# Output


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.2f}%"


def _opt(v: float | None, spec: str = ".4g") -> str:
    return "-" if v is None else format(v, spec)


def format_group(name: str, res: dict, value: str) -> str:
    pts = res["grids"]
    lines = [f"group {name}: {value} on {len(pts)} grids, h from {res['ratio_from']}"]
    if pts:
        lines.append(f"  {'grid':<14}{'cells':>10}{'h/h1':>8}{value:>12}{'sem':>10}")
        lines += [
            f"  {p['grid']!s:<14}{_opt(p['cells'], '.0f'):>10}{p['h']:>8.3f}"
            f"{p['value']:>12.5g}{_opt(p['sem']):>10}"
            for p in pts
        ]
    t = res["primary"]
    if t is None:
        lines.append(f"  no estimate: {res['reason']}")
        lines += [f"  warning: {w}" for w in res["warnings"]]
        return "\n".join(lines)

    lines.append(
        f"  r21 {t['r21']:.3f}  r32 {t['r32']:.3f}  eps21 {t['eps21']:.4g}  "
        f"eps32 {t['eps32']:.4g}  {t['convergence']}"
    )
    if not t["ok"]:
        lines.append(f"  no order reported: {t['reason']}")
    else:
        a, u = t["asymptotic"], t["uncertainty"]
        lines += [
            f"  observed order p {t['order']:.3f} (formal {a['formal_order']:g})",
            f"  extrapolated {value} {t['extrapolated']:.5g}",
            f"  e_a21 {_pct(t['e_a21'])}  e_ext21 {_pct(t['e_ext21'])}  "
            f"GCI_fine21 {_pct(t['gci_fine21'])}  GCI_coarse21 {_pct(t['gci_coarse21'])}  "
            f"GCI_fine32 {_pct(t['gci_fine32'])}",
            f"  asymptotic GCI32/(r21^p GCI21) {a['gci_ratio']:.3f} "
            f"({'near 1' if a['gci_ratio_ok'] else 'not near 1'})",
        ]
        if a.get("p_spread") is not None:
            lines.append(f"  order spread over {len(res['triplets'])} triplets {a['p_spread']:.1%}")
        stat = "" if u["statistical"] is None else f", statistical {u['statistical']:.3g}"
        lines.append(
            f"  {value} = {u['value']:.5g} +/- {u['combined']:.3g} "
            f"(about 95%, numerical {u['numerical']:.3g}{stat})"
        )
        if u["extrapolated_statistical"] is not None:
            lines.append(
                f"  extrapolated {value} {t['extrapolated']:.5g} +/- "
                f"{u['extrapolated_statistical']:.3g} from the runs' noise"
            )
    for i, extra in enumerate(res["triplets"][1:], start=1):
        detail = f"p {extra['order']:.3f}" if extra["ok"] else extra["reason"]
        lines.append(f"  triplet from {pts[i]['grid']}: {detail}")
    lines += [f"  warning: {w}" for w in res["warnings"]]
    return "\n".join(lines)


def plot_group(path: Path, name: str, res: dict, value: str, k: float) -> None:
    """Value against h/h1 with k x sem bars and the Richardson fit"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, ink, grid = "#257de0", "#3b3b3b", "#e7e7e7"
    pts = res["grids"]
    h = [p["h"] for p in pts]
    v = [p["value"] for p in pts]
    err = [k * p["sem"] if p.get("sem") is not None else 0.0 for p in pts]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.errorbar(h, v, yerr=err, fmt="o", color=blue, ms=8, lw=2, capsize=3, label=value)
    t = res["primary"]
    if t is not None and t["ok"]:
        p, ext = t["order"], t["extrapolated"]
        xs = [h[2] * i / 100 for i in range(101)]
        ys = [ext + (v[0] - ext) * x**p for x in xs]
        ax.plot(xs, ys, "--", color=blue, lw=2, label=f"fit, p = {p:.2f}")
        ax.plot([0], [ext], "o", mfc="white", mec=blue, mew=2, ms=8, label="extrapolated")
        title = f"{name}: GCI_fine {100 * t['gci_fine21']:.2f}%"
        ax.legend(frameon=False)
    else:
        title = f"{name}: no estimate"
    ax.set_xlim(-0.05 * h[-1], 1.05 * h[-1])
    ax.set_xlabel("grid spacing h / h_fine", color=ink)
    ax.set_ylabel(value, color=ink)
    ax.set_title(title, color=ink, fontsize=10)
    ax.grid(color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _clean(obj: Any) -> Any:
    """NaN and inf become null so the JSON stays valid"""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    return obj


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grid convergence index from study results")
    ap.add_argument("results", type=Path, help="results.json or results.csv from study.py")
    ap.add_argument("--value", default="cd", help="column to analyse (default: cd)")
    ap.add_argument("--sem", default="cd_sem", help="standard error column (default: cd_sem)")
    ap.add_argument("--no-sem", action="store_true", help="ignore standard errors")
    ap.add_argument("--group", help="only this group")
    ap.add_argument("--ratio-from", choices=("cells", "char_length"), default="cells")
    ap.add_argument("--dims", type=int, default=3, help="mesh dimensions for h from cells")
    ap.add_argument("--formal-order", type=float, default=2.0)
    ap.add_argument("--fs", type=float, default=1.25, help="safety factor")
    ap.add_argument("--k", type=float, default=2.0, help="coverage factor on the sem")
    ap.add_argument("--json", type=Path, help="write the full result here")
    ap.add_argument("--plot", type=Path, help="write a PNG plot here (needs matplotlib)")
    args = ap.parse_args(argv)

    try:
        rows = load_rows(args.results)
    except (OSError, ValueError, KeyError) as e:
        print(f"gci: cannot read {args.results}: {e}", file=sys.stderr)
        return 2
    groups, skipped = group_points(rows, args.value, None if args.no_sem else args.sem)
    if args.group:
        if args.group not in groups:
            print(f"gci: no group {args.group!r}, have {sorted(groups)}", file=sys.stderr)
            return 2
        groups = {args.group: groups[args.group]}
    if not groups:
        print(f"gci: no usable rows with {args.value!r} in {args.results}", file=sys.stderr)
        return 2

    opts = Options(fs=args.fs, formal_order=args.formal_order, k=args.k)
    results = {g: analyse(pts, args.ratio_from, args.dims, opts) for g, pts in groups.items()}
    for g, res in results.items():
        print(format_group(g, res, args.value))
    for s in skipped:
        print(f"skipped {s}")

    if args.json:
        payload = {
            "value": args.value,
            "options": asdict(opts),
            "groups": results,
            "skipped": skipped,
        }
        args.json.write_text(json.dumps(_clean(payload), indent=2) + "\n")
    if args.plot:
        for g, res in results.items():
            path = args.plot
            if len(results) > 1:
                path = path.with_name(f"{path.stem}-{g}{path.suffix}")
            plot_group(path, g, res, args.value, args.k)
            print(f"wrote {path}")
    return 0 if all(r["primary"] and r["primary"]["ok"] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
