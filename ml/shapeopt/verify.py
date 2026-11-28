"""
Check optimised shapes against the solver.

Takes the candidate files that optimize.py writes next to each optimised
OBJ, runs the baseline and every candidate through the study runner at
the same grid, Reynolds number and run length, and writes a table:
predicted Cd, solver Cd with its 95% interval, the surrogate's error,
predicted and solver change from the baseline, and a verdict.

The interval on a change combines the two runs' 95% half-widths in
quadrature. A change is verified when that interval excludes zero on the
side the surrogate predicted, not distinguishable when it includes zero,
and wrong direction when it excludes zero on the other side. The
intervals cover sampling noise only, not grid error. Run a second grid
with another --grid to see whether a change survives refinement.

Cd uses each shape's own frontal area. CdA/A0 is Cd times the frontal
area over the baseline's, so it compares drag forces. A shape can lower
Cd just by growing its frontal area.

Usage:
    python verify.py opt.json --binary build/3d_fluid_simulation_car --out verify_out
    python verify.py opt.json --binary ... --out ... --grid 64x32x32 --grid 128x64x64
    python verify.py opt.json --binary ... --out ... --reynolds 100 --duration 30
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mesh_checks
import protocol
import study
from ffd import FFDLattice
from ffd import compute_bbox
from ffd import load_obj
from gci import spacing
from pipeline_config import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERDICTS = ("verified", "not distinguishable", "wrong direction")
CANDIDATE_KIND = "shapeopt-candidate"


# Arithmetic


def combine(*half_widths: float | None) -> float | None:
    """Half-width of a sum or difference of independent estimates"""
    if any(h is None or not math.isfinite(h) for h in half_widths):
        return None
    return math.hypot(*half_widths)


def verdict(delta: float, half: float | None, predicted: float) -> str:
    """Compare a solver change and its interval with the predicted change"""
    if half is None:
        return "no interval"
    if abs(delta) <= half:
        return "not distinguishable"
    # the optimiser lowers drag, so a zero prediction counts as a reduction
    direction = 1.0 if predicted > 0 else -1.0
    return "verified" if math.copysign(1.0, delta) == direction else "wrong direction"


def compare(base: dict, shape: dict) -> dict:
    """
    One shape against the baseline on one grid.

    base and shape hold cd and ci95 (solver mean and 95% half-width) and
    area (solver frontal area). shape also holds pred and pred_base (the
    surrogate's Cd for the shape and the baseline) and pred_area_ratio
    (frontal area ratio of the meshes).
    """
    r = shape["area"] / base["area"]
    delta = shape["cd"] - base["cd"]
    half = combine(shape["ci95"], base["ci95"])
    pred_delta = shape["pred"] - shape["pred_base"]
    cda = shape["cd"] * r
    cda_half = combine(None if shape["ci95"] is None else r * shape["ci95"], base["ci95"])
    pred_cda_delta = shape["pred"] * shape["pred_area_ratio"] - shape["pred_base"]
    return {
        "pred_cd": shape["pred"],
        "cd": shape["cd"],
        "cd_ci95": shape["ci95"],
        "surrogate_error": shape["pred"] - shape["cd"],
        "pred_delta": pred_delta,
        "delta": delta,
        "delta_ci95": half,
        "delta_pct": 100 * delta / base["cd"],
        "delta_pct_ci95": None if half is None else 100 * half / base["cd"],
        "verdict": verdict(delta, half, pred_delta),
        "area_ratio": r,
        "pred_area_ratio": shape["pred_area_ratio"],
        "cda": cda,
        "cda_ci95": None if shape["ci95"] is None else r * shape["ci95"],
        "pred_cda_delta": pred_cda_delta,
        "cda_delta": cda - base["cd"],
        "cda_delta_ci95": cda_half,
        "cda_verdict": verdict(cda - base["cd"], cda_half, pred_cda_delta),
    }


def overall(verdicts: list[str]) -> str:
    """One verdict over every grid, grid dependent when they disagree"""
    if not verdicts:
        return "not run"
    return verdicts[0] if len(set(verdicts)) == 1 else "grid dependent"


# Inputs


def load_candidate(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("kind") != CANDIDATE_KIND:
        raise ValueError(f"{path}: not a candidate file from optimize.py")
    data["path"] = str(path)
    return data


def settings_from(args: argparse.Namespace, proto: dict | None) -> tuple[dict, list[str]]:
    """Grids, Re and run length, from the command line or the training protocol"""
    notes = []
    proto = proto or {}
    grids = args.grid or ([proto["grid"]] if proto.get("grid") else [])
    if not grids:
        raise ValueError("no training protocol in the candidate, give --grid")
    reynolds = args.reynolds
    if reynolds is None:
        reynolds = (proto.get("reference") or {}).get("reynolds")
    if reynolds is None:
        raise ValueError("no training protocol in the candidate, give --reynolds")

    if args.duration is not None or args.converge is not None:
        run = protocol.run_settings(
            args.duration, args.converge, args.max_duration, args.converge_window
        )
        run_grid = grids[0]
    elif proto.get("run"):
        run, run_grid = dict(proto["run"]), proto["grid"]
    else:
        raise ValueError("no training protocol in the candidate, give --duration or --converge")

    if not proto:
        notes.append("surrogate training settings unknown, its error may mix settings")
    else:
        if proto.get("grid") not in grids:
            notes.append(f"surrogate trained on {proto['grid']}, not one of the grids run here")
        if float(reynolds) != float(proto["reference"]["reynolds"]):
            notes.append(f"surrogate trained at Re {proto['reference']['reynolds']:g}")
        if run_grid == proto.get("grid") and run != proto.get("run"):
            notes.append(f"surrogate data ran with {proto.get('run')}, here {run}")
    return {"grids": grids, "reynolds": float(reynolds), "run": run, "run_grid": run_grid}, notes


# Running


def _case_info(row: dict) -> dict:
    manifest = json.loads(Path(row["manifest"]).read_text())
    return {
        "cd": row["cd"],
        "ci95": row["cd_ci95"],
        "area": (manifest.get("model") or {}).get("ref_area"),
        "cl": (manifest.get("results") or {}).get("cl_mean"),
        "tau": row["tau"],
        "reynolds": row["reynolds"],
        "stationary": row["cd_stationary"],
        "converged": row["converged"],
    }


def _cells(grid: str) -> int:
    return math.prod(int(n) for n in grid.split("x"))


def _caveats(info: dict) -> list[str]:
    out = []
    if info["stationary"] is False:
        out.append("Cd drifting")
    if info["converged"] is False:
        out.append("not converged")
    return out


def run(
    candidates: list[dict],
    settings: dict,
    binary: Path,
    out: Path,
    workdir: Path = PROJECT_ROOT / "simulation",
    timeout: float | None = None,
    backend: study.Backend | None = None,
    force: bool = False,
    notes: list[str] | None = None,
) -> dict:
    """Geometry checks, solver runs and the comparison table"""
    base_objs = {c["base_obj"] for c in candidates}
    if len(base_objs) != 1:
        raise ValueError(f"candidates have different baselines: {sorted(base_objs)}")
    base_obj = Path(base_objs.pop())
    base_v, faces = load_obj(base_obj)
    ref = protocol.reference(base_v, settings["reynolds"])
    area0 = mesh_checks.frontal_area(base_v, faces)

    shapes = []
    for i, cand in enumerate(candidates):
        verts, cand_faces = load_obj(cand["obj"])
        lattice = None
        if cand.get("ffd_grid") and cand.get("displacements") is not None:
            lattice = FFDLattice(*compute_bbox(base_v), n_control=tuple(cand["ffd_grid"]))
        geo = mesh_checks.check(
            base_v,
            verts,
            cand_faces,
            lattice,
            None if lattice is None else np.asarray(cand["displacements"]),
            area0,
        )
        shapes.append(
            {
                "name": cand.get("name") or f"shape{i + 1}",
                "obj": cand["obj"],
                "candidate": cand["path"],
                "objective": cand.get("objective"),
                "pred": cand["predicted"],
                "geometry": geo,
                "vertices": verts,
                "rows": [],
            }
        )

    cases = {}
    for grid in settings["grids"]:
        run_keys = protocol.scale_run(settings["run"], grid, settings["run_grid"])
        cases[("baseline", grid)] = protocol.case_params(
            base_obj, base_v, ref, grid=grid, **run_keys
        )
        for s in shapes:
            if s["geometry"]["ok"]:
                cases[(s["name"], grid)] = protocol.case_params(
                    Path(s["obj"]).resolve(), s["vertices"], ref, grid=grid, **run_keys
                )
    study_file = protocol.write_study(
        out / "study.yaml", binary, workdir, out / "runs", list(cases.values()), "verify", timeout
    )
    loaded = study.load_study(study_file)
    result, _ = study.run_study(loaded, backend or study.LocalBackend(), force)
    rows = {r["name"]: r for r in result["rows"]}
    info = {}
    for key, params in cases.items():
        row = rows[study.case_name(params)]
        info[key] = _case_info(row) if row["status"] == "ok" and row["cd"] is not None else None

    warnings = list(notes or []) + list(result["warnings"])
    # finest grid first, as in gci
    grids = sorted(settings["grids"], key=lambda g: spacing([{"cells": _cells(g)}])[0])
    baseline = {}
    for grid in grids:
        b = info[("baseline", grid)]
        baseline[grid] = b
        taus = {round(v["tau"], 6) for k, v in info.items() if k[1] == grid and v and v["tau"]}
        if len(taus) > 1:
            warnings.append(f"{grid}: shapes ran at different tau {sorted(taus)}")

    training_grid = (candidates[0].get("protocol") or {}).get("grid")
    for s in shapes:
        s.pop("vertices")
        verdicts = []
        for grid in grids:
            b, got = baseline[grid], info.get((s["name"], grid))
            entry = {"grid": grid, "training_grid": grid == training_grid}
            if not s["geometry"]["ok"]:
                entry["status"] = "geometry rejected"
            elif b is None or got is None:
                entry["status"] = "baseline failed" if b is None else "failed"
            else:
                p = s["pred"]
                cmp = compare(
                    b,
                    {
                        "pred": p["cd"],
                        "pred_base": p["baseline_cd"],
                        "pred_area_ratio": s["geometry"]["frontal_area_ratio"],
                        **got,
                    },
                )
                if not entry["training_grid"]:
                    cmp["surrogate_error"] = None
                cmp["caveats"] = _caveats(got) + [f"baseline {c}" for c in _caveats(b)]
                entry.update(status="ok", **cmp)
                verdicts.append(cmp["verdict"])
            s["rows"].append(entry)
        if len(verdicts) == len(grids):
            s["overall"] = overall(verdicts)
        else:
            s["overall"] = s["rows"][0]["status"] if s["rows"] else "not run"

    return {
        "settings": {**settings, "reference": ref.to_dict(), "binary": str(binary)},
        "base_obj": str(base_obj),
        "base_frontal_area": area0,
        "training_grid": training_grid,
        "grids": grids,
        "baseline": baseline,
        "baseline_pred": {s["name"]: s["pred"].get("baseline_cd") for s in shapes},
        "shapes": shapes,
        "warnings": warnings,
        "study": str(study_file),
    }


# Output


def _f(v: float | None, spec: str = ".4f") -> str:
    return "-" if v is None else format(v, spec)


def _pm(v: float | None, h: float | None, spec: str = ".4f") -> str:
    if v is None:
        return "-"
    return f"{v:{spec}}" if h is None else f"{v:{spec}} ± {h:{spec}}"


def _signed(v: float | None, h: float | None = None) -> str:
    return _pm(v, h, "+.4f")


HEADER = [
    "shape",
    "grid",
    "pred Cd",
    "solver Cd (95%)",
    "surrogate error",
    "pred ΔCd",
    "solver ΔCd (95%)",
    "ΔCd %",
    "verdict",
    "A/A0",
    "pred ΔCdA/A0",
    "solver ΔCdA/A0 (95%)",
    "CdA verdict",
]


def table_rows(result: dict) -> list[list[str]]:
    lines = []
    preds = set(result["baseline_pred"].values())
    pred0 = preds.pop() if len(preds) == 1 else None
    for grid in result["grids"]:
        b = result["baseline"][grid]
        if b is None:
            lines.append(["baseline", grid, *["-"] * 6, "failed", *["-"] * 4])
            continue
        err = None if pred0 is None or grid != result["training_grid"] else pred0 - b["cd"]
        lines.append(
            ["baseline", grid, _f(pred0), _pm(b["cd"], b["ci95"]), _signed(err)]
            + ["-"] * 4
            + ["1.0000", "-", "-", "-"]
        )
    for s in result["shapes"]:
        for e in s["rows"]:
            if e["status"] != "ok":
                lines.append([s["name"], e["grid"], *["-"] * 6, e["status"], *["-"] * 4])
                continue
            v = e["verdict"] + (f" ({', '.join(e['caveats'])})" if e["caveats"] else "")
            lines.append(
                [
                    s["name"],
                    e["grid"],
                    _f(e["pred_cd"]),
                    _pm(e["cd"], e["cd_ci95"]),
                    _signed(e["surrogate_error"]),
                    _signed(e["pred_delta"]),
                    _signed(e["delta"], e["delta_ci95"]),
                    _pm(e["delta_pct"], e["delta_pct_ci95"], "+.2f"),
                    v,
                    _f(e["area_ratio"]),
                    _signed(e["pred_cda_delta"]),
                    _signed(e["cda_delta"], e["cda_delta_ci95"]),
                    e["cda_verdict"],
                ]
            )
    return lines


def markdown(result: dict) -> str:
    rows = table_rows(result)
    out = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    st = result["settings"]
    out += [
        "",
        f"Re {st['reynolds']:g} on the baseline length, scale {st['reference']['scale']:.6g}, "
        f"run {st['run']} on {st['run_grid']}, durations scaled with Nx on other grids.",
        "Intervals are 95% and cover sampling noise only. The surrogate error is shown on "
        "the training grid only.",
    ]
    for s in result["shapes"]:
        g = s["geometry"]
        out.append(
            f"{s['name']}: overall {s['overall']}, frontal area x{g['frontal_area_ratio']:.4f}, "
            f"volume x{g['volume_ratio']:.4f}, length x{g['length_ratio']:.4f}"
            + (f", objective {s['objective']}" if s.get("objective") else "")
            + (f", rejected: {'; '.join(g['problems'])}" if g["problems"] else "")
        )
    out += [f"WARNING: {w}" for w in result["warnings"]]
    return "\n".join(out) + "\n"


def write_outputs(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    (out / "verify.md").write_text(markdown(result))
    with (out / "verify.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(table_rows(result))


def add_run_arguments(ap: argparse.ArgumentParser) -> None:
    """Solver options shared with optimize.py"""
    ap.add_argument("--binary", type=Path, help="solver binary")
    ap.add_argument(
        "--workdir",
        type=Path,
        default=PROJECT_ROOT / "simulation",
        help="directory the solver runs from (default: simulation/)",
    )
    ap.add_argument(
        "--grid", action="append", help="grid to run, repeat for a grid check (default: training)"
    )
    ap.add_argument("--reynolds", type=float, help="Re on the baseline length (default: training)")
    ap.add_argument("--duration", type=int, help="seconds on the first grid (default: training)")
    ap.add_argument("--converge", type=float, help="stop when the 95%% CI is within this fraction")
    ap.add_argument("--max-duration", type=int, help="cap for --converge on the first grid")
    ap.add_argument("--converge-window", type=int, help="Cd samples between convergence checks")
    ap.add_argument("--timeout", type=float, help="seconds per solver run")


def verify_files(
    paths: list[Path], args: argparse.Namespace, out: Path, backend: study.Backend | None = None
) -> dict:
    """Load candidates, run them, write and print the table"""
    candidates = [load_candidate(p) for p in paths]
    settings, notes = settings_from(args, candidates[0].get("protocol"))
    result = run(
        candidates,
        settings,
        args.binary,
        out,
        args.workdir,
        args.timeout,
        backend,
        getattr(args, "force", False),
        notes,
    )
    write_outputs(result, out)
    print(markdown(result), end="")
    print(f"wrote {out / 'verify.md'}, verify.csv and verify.json")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check optimised shapes against the solver")
    ap.add_argument("candidates", nargs="+", type=Path, help="candidate JSON from optimize.py")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--force", action="store_true", help="rerun solver cases that are done")
    add_run_arguments(ap)
    args = ap.parse_args(argv)
    if args.binary is None or not args.binary.is_file():
        print(f"verify: no solver binary at {args.binary}, give --binary", file=sys.stderr)
        return 2
    try:
        result = verify_files(args.candidates, args, args.out)
    except (ValueError, OSError, ConfigError) as e:
        print(f"verify: {e}", file=sys.stderr)
        return 2
    return 0 if all(s["overall"] in VERDICTS for s in result["shapes"]) else 1


if __name__ == "__main__":
    sys.exit(main())
