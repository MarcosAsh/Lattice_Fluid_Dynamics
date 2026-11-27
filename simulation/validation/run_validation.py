"""Run the GPU validation suite and write a results table and plots.

Run from simulation/ with a built tree, for example:

    uv run --with numpy --with matplotlib \
        python validation/run_validation.py --build build \
        --out ../docs/validation

Every case is a run of build/validation/lbm_validate (or the wind tunnel
binary for the Ahmed body). Each one prints a RESULT line that this
script parses. Exit status is nonzero if any check fails.
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def cgw(re_num):
    """Clift, Grace & Weber (1978) standard drag curve for a sphere."""
    if re_num <= 20:
        return 24 / re_num * (1 + 0.1315 * re_num ** (0.82 - 0.05 * math.log10(re_num)))
    if re_num <= 260:
        return 24 / re_num * (1 + 0.1935 * re_num**0.6305)
    w = math.log10(re_num)
    return 10 ** (1.6435 - 1.1242 * w + 0.1558 * w * w)


def schiller_naumann(re_num):
    return 24 / re_num * (1 + 0.15 * re_num**0.687)


def parse_result(text):
    rows = []
    for line in text.splitlines():
        if line.startswith("RESULT "):
            row = {}
            for tok in line.split()[1:]:
                k, _, v = tok.partition("=")
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


class Suite:
    def __init__(self, build, out, log_dir):
        self.validate = str(Path(build) / "validation" / "lbm_validate")
        self.tunnel = str(Path(build) / "3d_fluid_simulation_car")
        self.out = Path(out)
        self.logs = Path(log_dir)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.checks = []
        self.results = {}

    def run(self, name, args, binary=None):
        cmd = [binary or self.validate, *args]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        text = proc.stdout + proc.stderr
        (self.logs / f"{name}.log").write_text(" ".join(cmd) + "\n" + text)
        rows = parse_result(text)
        for r in rows:
            r["wall_seconds"] = round(dt, 1)
            r["command"] = " ".join(["./build/validation/lbm_validate", *args])
        self.results[name] = rows
        print(f"{name}: {len(rows)} result(s) in {dt:.0f} s, rc={proc.returncode}")
        return rows, text, proc.returncode

    def check(self, case, quantity, measured, reference, tol, ok, unit=""):
        self.checks.append(
            {
                "case": case,
                "quantity": quantity,
                "measured": measured,
                "reference": reference,
                "tolerance": tol,
                "pass": bool(ok),
            }
        )
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {case}: {quantity} = {measured} (ref {reference}, tol {tol}){unit}")


def uniform(s):
    rows, _, _ = s.run("uniform_mrt", ["uniform", "--op=mrt", "--steps=4000"])
    for r in rows:
        if r["bc"] == "periodic":
            s.check(
                "uniform periodic box",
                "relative mass drift",
                f"{r['mass_rel_drift']:.2e}",
                "0",
                "1e-6",
                abs(r["mass_rel_drift"]) < 1e-6,
            )
        else:
            s.check(
                "uniform open tunnel",
                "max |u - U|",
                f"{r['max_du']:.2e}",
                "0",
                "1e-4",
                r["max_du"] < 1e-4 and r["resets"] == 0,
            )


def tgv(s):
    table = []
    for op in ["bgk", "reg", "mrt", "ent"]:
        for n in [32, 64, 128]:
            rows, _, _ = s.run(f"tgv_{op}_{n}", ["tgv", f"--op={op}", f"--n={n}", "--tau=0.6"])
            if not rows:
                s.check(f"TGV {op} n={n}", "run", "no result", "", "", False)
                continue
            r = rows[0]
            table.append(r)
            if n == 64:
                s.check(
                    f"TGV {op} n=64 tau=0.6",
                    "nu_eff/nu - 1",
                    f"{r['nu_err']:.2e}",
                    "0",
                    "1e-3",
                    abs(r["nu_err"]) < 1e-3 and r["resets"] == 0,
                )
    # Low viscosity
    for op in ["mrt", "ent"]:
        rows, _, _ = s.run(
            f"tgv_{op}_lowvisc",
            ["tgv", f"--op={op}", "--n=64", "--tau=0.503", "--u=0.05"],
        )
        if rows:
            r = rows[0]
            table.append(r)
            s.check(
                f"TGV {op} n=64 tau=0.503",
                "nu_eff/nu - 1",
                f"{r['nu_err']:.2e}",
                "0",
                "1e-2",
                abs(r["nu_err"]) < 1e-2 and r["resets"] == 0,
            )
    # Smagorinsky eddy viscosity against the analytic prediction
    base, _, _ = s.run(
        "tgv_smag0",
        ["tgv", "--op=bgk", "--n=32", "--tau=0.51", "--u=0.05", "--steps=1000"],
    )
    for cs in [0.1, 0.2]:
        rows, _, _ = s.run(
            f"tgv_smag{cs}",
            [
                "tgv",
                "--op=bgk",
                "--n=32",
                "--tau=0.51",
                "--u=0.05",
                f"--smag={cs}",
                "--steps=1000",
            ],
        )
        if rows and base:
            r = rows[0]
            ratio = (r["nu_eff"] - base[0]["nu_eff"]) / r["nu_t_pred"]
            r["nu_t_ratio"] = ratio
            table.append(r)
            s.check(
                f"TGV Smagorinsky Cs={cs}",
                "nu_t measured / predicted",
                f"{ratio:.3f}",
                "1",
                "0.05",
                abs(ratio - 1) < 0.05,
            )
    # Periodic z seam across SSBO slabs
    a, _, _ = s.run("tgv_xz_whole", ["tgv", "--op=bgk", "--n=64", "--tau=0.6", "--plane=xz"])
    b, _, _ = s.run(
        "tgv_xz_slabs",
        [
            "tgv",
            "--op=bgk",
            "--n=64",
            "--tau=0.6",
            "--plane=xz",
            "--max-slab-z=16",
        ],
    )
    if a and b:
        same = a[0]["nu_eff"] == b[0]["nu_eff"] and a[0]["l2"] == b[0]["l2"]
        s.check(
            "TGV xz, 4 slabs vs 1",
            "nu_eff and L2 identical",
            f"{b[0]['nu_eff']:.6e}",
            f"{a[0]['nu_eff']:.6e}",
            "bitwise",
            same and b[0]["resets"] == 0,
        )
    s.results["tgv_table"] = table
    if table:
        fig, ax = plt.subplots(figsize=(6, 4))
        for op, mk in zip(["bgk", "reg", "mrt", "ent"], "osd^", strict=True):
            pts = [
                (r["n"], abs(r["nu_err"]))
                for r in table
                if r.get("op") == op and r.get("tau") == 0.6 and r.get("smag", 0) == 0
            ]
            if pts:
                pts.sort()
                ax.loglog(*zip(*pts, strict=True), marker=mk, label=op)
        ax.set_xlabel("N (cells per wavelength)")
        ax.set_ylabel("|nu_eff / nu - 1|")
        ax.set_title("Taylor-Green decay, tau = 0.6")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(s.out / "tgv_viscosity.png", dpi=120)
        plt.close(fig)


def poiseuille(s):
    table = []
    for op in ["bgk", "mrt"]:
        for q in [0.5, 0.25, 0.75]:
            errs = []
            for h in [8, 16, 32]:
                rows, _, _ = s.run(
                    f"pois_{op}_q{q}_h{h}",
                    [
                        "poiseuille",
                        f"--op={op}",
                        f"--n={h}",
                        f"--q={q}",
                        "--tau=0.8",
                    ],
                )
                if not rows:
                    continue
                r = rows[0]
                table.append(r)
                errs.append((h, r["l2"]))
                s.check(
                    f"Poiseuille {op} q={q} H={h}",
                    "wall force / body force - 1",
                    f"{r['force_err']:.1e}",
                    "0",
                    "1e-3",
                    abs(r["force_err"]) < 1e-3 and r["resets"] == 0,
                )
            if len(errs) >= 2:
                order = math.log(errs[0][1] / errs[1][1]) / math.log(errs[1][0] / errs[0][0])
                s.check(
                    f"Poiseuille {op} q={q}",
                    "L2 order, H 8 to 16",
                    f"{order:.2f}",
                    "2",
                    "> 1.7",
                    order > 1.7,
                )
    s.results["poiseuille_table"] = table
    if table:
        fig, ax = plt.subplots(figsize=(6, 4))
        for op, ls in [("bgk", "-"), ("mrt", "--")]:
            for q, mk in [(0.5, "o"), (0.25, "s"), (0.75, "^")]:
                pts = sorted((r["h"], r["l2"]) for r in table if r["op"] == op and r["q"] == q)
                if pts:
                    ax.loglog(*zip(*pts, strict=True), ls=ls, marker=mk, label=f"{op} q={q}")
        h = np.array([8, 32])
        ax.loglog(h, 0.02 * (h / 8.0) ** -2, "k:", label="slope -2")
        ax.set_xlabel("H (fluid nodes across channel)")
        ax.set_ylabel("relative L2 error of u(y)")
        ax.set_title("Body-force Poiseuille, tau = 0.8")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(s.out / "poiseuille_convergence.png", dpi=120)
        plt.close(fig)


def qcheck(s):
    for d in [8, 16]:
        rows, _, _ = s.run(f"qcheck_d{d}", ["qcheck", f"--d={d}"])
        if rows:
            r = rows[0]
            s.check(
                f"mesh vs analytic sphere D={d}",
                "links stuck at q=0.5",
                int(r["stuck_at_half"]),
                "0",
                "0",
                r["stuck_at_half"] == 0,
            )
            s.check(
                f"mesh vs analytic sphere D={d}",
                "mean |q_mesh - q_exact|",
                f"{r['mean_dq']:.4f}",
                "0",
                "0.01",
                r["mean_dq"] < 0.01,
            )
            rel = r["mesh_area"] / r["exact_area"] - 1
            s.check(
                f"mesh frontal area D={d}",
                "A_mesh / (pi D^2/4) - 1",
                f"{rel:.4f}",
                "0",
                "0.01",
                abs(rel) < 0.01,
            )


def sphere(s, quick):
    table = []
    cases = [(20, 8), (20, 12), (20, 16), (100, 8), (100, 12), (100, 16)]
    if quick:
        cases = [(20, 8), (100, 8)]
    for re_num, d in cases:
        rows, _, _ = s.run(f"sphere_re{re_num}_d{d}", ["sphere", f"--re={re_num}", f"--d={d}"])
        if not rows:
            s.check(f"sphere Re={re_num} D={d}", "run", "no result", "", "", False)
            continue
        r = rows[0]
        ref = cgw(re_num)
        r["cd_ref"] = ref
        table.append(r)
        err = r["cd"] / ref - 1
        tol = 0.05 if d >= 16 else 0.10
        s.check(
            f"sphere Re={re_num} D={d}",
            "Cd / Cd_CGW - 1",
            f"{err:+.3f}",
            f"Cd={ref:.3f}",
            f"{tol}",
            abs(err) < tol and r["resets"] == 0,
        )
    # Domain size at D=8, 16D box against 8D
    if not quick:
        for re_num in [20, 100]:
            rows, _, _ = s.run(
                f"sphere_re{re_num}_d8_wide",
                [
                    "sphere",
                    f"--re={re_num}",
                    "--d=8",
                    "--up=12",
                    "--down=20",
                    "--lat=16",
                ],
            )
            if rows:
                rows[0]["cd_ref"] = cgw(re_num)
                table.append(rows[0])
        # Icosphere through LBM_SetSolidMesh, and Smagorinsky on
        rows, _, _ = s.run(
            "sphere_re100_d16_mesh",
            ["sphere", "--re=100", "--d=16", "--mesh=1"],
        )
        if rows:
            rows[0]["cd_ref"] = cgw(100)
            table.append(rows[0])
        rows, _, _ = s.run(
            "sphere_re100_d16_smag",
            ["sphere", "--re=100", "--d=16", "--smag=0.1"],
        )
        if rows:
            rows[0]["cd_ref"] = cgw(100)
            table.append(rows[0])
        rows, _, _ = s.run("sphere_re100_d12_bgk", ["sphere", "--re=100", "--d=12", "--op=bgk"])
        if rows:
            rows[0]["cd_ref"] = cgw(100)
            table.append(rows[0])
    s.results["sphere_table"] = table
    if table:
        fig, ax = plt.subplots(figsize=(6, 4))
        for re_num, mk in [(20, "o"), (100, "s")]:
            pts = sorted(
                (r["D"], r["cd"])
                for r in table
                if r["re"] == re_num
                and r["grid"].startswith(str(int(16 * r["D"])))
                and r["path"] == "analytic"
                and r["smag"] == 0
            )
            if pts:
                ax.plot(*zip(*pts, strict=True), marker=mk, label=f"LBM Re={re_num}")
                ax.axhline(cgw(re_num), ls="--", color=ax.lines[-1].get_color(), lw=0.8)
        ax.set_xlabel("sphere diameter D (cells)")
        ax.set_ylabel("Cd")
        ax.set_title("Sphere drag, dashed = Clift, Grace & Weber")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(s.out / "sphere_cd.png", dpi=120)
        plt.close(fig)


def ahmed(s, quick):
    table = []
    grids = ["128x64x64"] if quick else ["128x64x64", "256x128x128"]
    for angle in [25, 35]:
        for grid in grids:
            dur = 20 if grid == "128x64x64" else 30
            name = f"ahmed{angle}_{grid}"
            manifest = s.logs / f"{name}.json"
            args = [
                f"--model=assets/3d-files/ahmed_{angle}deg_m.obj",
                f"--grid={grid}",
                "--wind=1.0",
                f"--duration={dur}",
            ]
            _, text, rc = s.run(name, [*args, f"--manifest={manifest}"], binary=s.tunnel)
            # The per-sample series is only on stdout
            cd = [float(m) for m in re.findall(r" Cd=([-0-9.]+)", text)]
            cl = [float(m) for m in re.findall(r" Cl=([-0-9.]+)", text)]
            man = json.loads(manifest.read_text()) if manifest.exists() else {}
            lat = man.get("lattice", {})
            area = re.search(r"Reference area: ([0-9.]+) cells\^2 \(voxel ([0-9.]+)\)", text)
            if len(cd) < 8:
                s.check(
                    f"Ahmed {angle} {grid}",
                    "Cd samples",
                    len(cd),
                    ">= 8",
                    "",
                    False,
                )
                continue
            tail = np.array(cd[len(cd) // 2 :])
            row = {
                "angle": angle,
                "grid": grid,
                "re": lat.get("reynolds_effective", float("nan")),
                "tau": lat.get("tau", float("nan")),
                "mach": lat.get("mach", float("nan")),
                "operator": lat.get("collision_operator", "?"),
                "cd_mean": float(tail.mean()),
                "cd_sd": float(tail.std()),
                "cl_mean": float(np.mean(cl[len(cl) // 2 :])),
                "samples": len(cd),
                "area_mesh": float(area.group(1)) if area else float("nan"),
                "area_voxel": float(area.group(2)) if area else float("nan"),
                "rc": rc,
                "command": "./build/3d_fluid_simulation_car " + " ".join(args),
                "series": cd,
            }
            table.append(row)
            s.check(
                f"Ahmed {angle} {grid}",
                "clean exit, Cd sample spread",
                f"rc={rc} sd/mean={row['cd_sd'] / row['cd_mean']:.3f}",
                "rc=0",
                "< 0.05",
                rc == 0 and row["cd_sd"] < 0.05 * row["cd_mean"],
            )
    s.results["ahmed_table"] = table
    if table:
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in table:
            ax.plot(r["series"], label=f"{r['angle']} deg {r['grid']}")
        ax.set_xlabel("Cd sample (200 steps each)")
        ax.set_ylabel("Cd")
        ax.set_title("Ahmed body, wind tunnel binary defaults")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(s.out / "ahmed_cd_series.png", dpi=120)
        plt.close(fig)


def write_table(s):
    lines = [
        "| Case | Quantity | Measured | Reference | Tolerance | Result |",
        "|---|---|---|---|---|---|",
    ]
    for c in s.checks:
        cells = [
            c["case"],
            c["quantity"],
            c["measured"],
            c["reference"],
            c["tolerance"],
            "pass" if c["pass"] else "FAIL",
        ]
        lines.append("| " + " | ".join(str(x) for x in cells) + " |")
    (s.out / "results.md").write_text("\n".join(lines) + "\n")
    (s.out / "results.json").write_text(
        json.dumps({"checks": s.checks, "results": s.results}, indent=1, default=str)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", default="build")
    ap.add_argument("--out", default="../docs/validation")
    ap.add_argument("--logs", default=None, help="raw run logs (default: <out>/logs)")
    ap.add_argument("--quick", action="store_true", help="small grids only")
    ap.add_argument("--cases", default="uniform,tgv,poiseuille,qcheck,sphere,ahmed")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = Suite(a.build, out, a.logs or out / "logs")
    for case in a.cases.split(","):
        if case in ("sphere", "ahmed"):
            globals()[case](s, a.quick)
        else:
            globals()[case](s)
    write_table(s)
    failed = [c for c in s.checks if not c["pass"]]
    print(f"{len(s.checks) - len(failed)}/{len(s.checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
