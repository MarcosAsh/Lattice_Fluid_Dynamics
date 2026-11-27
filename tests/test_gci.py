import csv
import json
import math

import pytest

import gci


def seq(p, h=(1.0, 2.0, 4.0), exact=1.0, c=0.01):
    """phi(h) = exact + c h^p, fine grid first"""
    return [exact + c * x**p for x in h]


def ratios(h):
    return h[1] / h[0], h[2] / h[1]


def test_exactly_second_order():
    t = gci.assess(seq(2), 2.0, 2.0)
    assert t["ok"]
    assert t["convergence"] == "monotone"
    assert t["order"] == pytest.approx(2.0, abs=1e-9)
    assert t["extrapolated"] == pytest.approx(1.0, abs=1e-12)
    assert t["gci_coarse21"] == pytest.approx(t["gci_fine21"] * 4.0)


def test_first_order_sits_on_the_lower_bound():
    t = gci.assess(seq(1), 2.0, 2.0)
    assert t["ok"]
    assert t["order"] == pytest.approx(1.0, abs=1e-9)
    assert t["extrapolated"] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("p", [1.0, 1.6, 2.0, 3.0])
def test_non_uniform_ratios_use_the_iteration(p):
    h = (1.0, 4.0 / 3.0, 2.0)  # 64, 48 and 32 cells across
    t = gci.assess(seq(p, h), *ratios(h))
    assert t["ok"]
    assert t["order"] == pytest.approx(p, abs=1e-9)
    assert t["extrapolated"] == pytest.approx(1.0, abs=1e-10)
    assert t["diagnostic"]["iterations"] > 1


def test_asymptotic_ratio_for_an_exact_sequence():
    phi = seq(2)
    t = gci.assess(phi, 2.0, 2.0)
    # with three grids the ratio reduces to phi1 / phi2
    assert t["asymptotic"]["gci_ratio"] == pytest.approx(phi[0] / phi[1])
    assert t["asymptotic"]["gci_ratio_ok"]


def test_oscillatory_is_refused():
    t = gci.assess([1.0, 1.01, 0.995], 2.0, 2.0)
    assert not t["ok"]
    assert t["convergence"] == "oscillatory"
    assert t["reason"].startswith("oscillatory convergence, eps32/eps21=-1.5")
    assert t["order"] is None
    assert t["extrapolated"] is None


def test_divergent_is_refused():
    t = gci.assess([1.0, 1.1, 1.15], 2.0, 2.0)
    assert not t["ok"]
    assert t["convergence"] == "divergent"
    assert "differences grow" in t["reason"]


def test_divergent_with_non_uniform_ratios():
    h = (1.0, 4.0 / 3.0, 2.0)
    t = gci.assess(seq(-0.5, h), *ratios(h))
    assert t["convergence"] == "divergent"
    assert not t["ok"]
    # the absolute value in Celik eq. 3a still returns a positive p here
    assert t["diagnostic"]["p_raw"] > 0


@pytest.mark.parametrize(("p", "shown"), [(0.3, "p=0.3 outside [1, 4]"), (5.0, "p=5 outside")])
def test_order_outside_the_sane_range(p, shown):
    t = gci.assess(seq(p), 2.0, 2.0)
    assert t["convergence"] == "monotone"
    assert not t["ok"]
    assert shown in t["reason"]


def test_flat_results_are_refused():
    t = gci.assess([0.4, 0.4, 0.4], 2.0, 2.0)
    assert t["convergence"] == "flat"
    assert t["reason"].startswith("no change between grids")


def test_changes_within_noise_are_refused():
    t = gci.assess(seq(2), 2.0, 2.0, sem=[0.02, 0.02, 0.02])
    assert not t["ok"]
    assert t["reason"].startswith("grid changes are within noise")


def test_uncertainty_carries_the_standard_errors():
    phi = seq(2)
    quiet = gci.assess(phi, 2.0, 2.0)
    t = gci.assess(phi, 2.0, 2.0, sem=[1e-4, 1e-4, 1e-4])
    u = t["uncertainty"]
    assert u["numerical"] == pytest.approx(quiet["uncertainty"]["numerical"])
    assert u["statistical"] == pytest.approx(2e-4)
    assert u["combined"] == pytest.approx(math.hypot(u["numerical"], 2e-4))
    assert quiet["uncertainty"]["statistical"] is None
    assert quiet["uncertainty"]["extrapolated_statistical"] is None

    double = gci.assess(phi, 2.0, 2.0, sem=[2e-4, 2e-4, 2e-4])["uncertainty"]
    assert double["extrapolated_statistical"] == pytest.approx(
        2 * u["extrapolated_statistical"], rel=1e-3
    )


# Celik et al. (2008) Table 1, 2D grids so r = (N_fine / N_coarse)^(1/2)
CELIK = [
    # cells, phi, p, phi_ext21, e_a21 %, e_ext21 %, GCI_fine21 %
    ((18000, 8000, 4500), (6.063, 5.972, 5.863), 1.53, 6.1685, 1.5, 1.7, 2.2),
    ((18000, 4500, 980), (10.7880, 10.7250, 10.6050), 0.75, 10.8801, 0.6, 0.9, 1.1),
    ((18000, 4500, 980), (6.0042, 5.9624, 6.0909), 1.51, 6.0269, 0.7, 0.4, 0.5),
]


@pytest.mark.parametrize(("cells", "phi", "p", "ext", "ea", "eext", "gci_fine"), CELIK)
def test_celik_table_1(cells, phi, p, ext, ea, eext, gci_fine):
    r21 = (cells[0] / cells[1]) ** 0.5
    r32 = (cells[1] / cells[2]) ** 0.5
    got, _, converged = gci.observed_order(phi, r21, r32)
    assert converged
    assert got == pytest.approx(p, abs=0.005)
    est = gci.estimates(phi, r21, r32, got)
    assert est["extrapolated"] == pytest.approx(ext, abs=5e-5)
    # table values are rounded, and e_ext21 = 0.8465% is printed as 0.9%
    assert 100 * est["e_a21"] == pytest.approx(ea, abs=0.06)
    assert 100 * est["e_ext21"] == pytest.approx(eext, abs=0.06)
    assert 100 * est["gci_fine21"] == pytest.approx(gci_fine, abs=0.06)


def test_celik_verdicts():
    def run(i):
        cells, phi = CELIK[i][:2]
        return gci.assess(phi, (cells[0] / cells[1]) ** 0.5, (cells[1] / cells[2]) ** 0.5)

    assert run(0)["ok"]
    assert "p=0.752 outside [1, 4]" in run(1)["reason"]
    assert run(2)["convergence"] == "oscillatory"


def points(values, grids=("64x32x32", "48x24x24", "32x16x16"), sem=None):
    out = []
    for i, (g, v) in enumerate(zip(grids, values, strict=True)):
        cells = math.prod(int(n) for n in g.split("x"))
        nx = int(g.split("x")[0])
        out.append(
            {
                "name": g,
                "grid": g,
                "cells": cells,
                "char_length": 0.3 * nx,
                "value": v,
                "sem": sem[i] if sem else None,
            }
        )
    return out


def test_analyse_sorts_and_uses_cell_counts():
    h = (1.0, 4.0 / 3.0, 2.0)
    pts = points(seq(2, h))
    res = gci.analyse(list(reversed(pts)))
    assert [p["grid"] for p in res["grids"]] == ["64x32x32", "48x24x24", "32x16x16"]
    assert [p["h"] for p in res["grids"]] == pytest.approx(h)
    assert res["primary"]["ok"]
    assert res["primary"]["order"] == pytest.approx(2.0)
    assert "no standard error on some grids, noise is not checked" in res["warnings"]


def test_analyse_needs_three_distinct_grids():
    assert (
        "needs results on 3 grids, got 2"
        in gci.analyse(points([1.0, 1.1], grids=("64x32x32", "32x16x16")))["reason"]
    )
    res = gci.analyse(points([1.0, 1.1, 1.2], grids=("64x32x32", "64x32x32", "32x16x16")))
    assert "same resolution" in res["reason"]


def test_small_ratio_and_mismatched_length_warn():
    pts = points(seq(2, (1.0, 1.1, 2.0)), grids=("110x55x55", "100x50x50", "55x28x28"))
    pts[1]["char_length"] = 40.0
    res = gci.analyse(pts)
    assert any("below 1.3" in w for w in res["warnings"])
    assert any("from char_length" in w for w in res["warnings"])


def test_reynolds_drift_warns():
    pts = points(seq(2, (1.0, 4.0 / 3.0, 2.0)))
    for p, re in zip(pts, (144.0, 108.0, 72.0), strict=True):
        p["reynolds"] = re
    res = gci.analyse(pts)
    assert "Re differs across grids (144, 108, 72), so they are not one flow" in res["warnings"]


def test_four_grids_give_two_triplets():
    grids = ("128x64x64", "64x32x32", "32x16x16", "16x8x8")
    res = gci.analyse(points(seq(2, (1, 2, 4, 8)), grids=grids))
    assert len(res["triplets"]) == 2
    assert all(t["ok"] for t in res["triplets"])
    assert res["primary"]["asymptotic"]["p_spread"] == pytest.approx(0.0, abs=1e-9)


def study_rows(values, sems=None, group="g1"):
    rows = [
        {
            "name": f"m-{p['grid']}",
            "group": group,
            "status": "ok",
            "grid": p["grid"],
            "cells": p["cells"],
            "char_length": p["char_length"],
            "cd": p["value"],
            "cd_sem": p["sem"],
        }
        for p in points(values, sem=sems)
    ]
    rows.append({"name": "m-bad", "group": group, "status": "failed", "grid": "16x8x8"})
    return rows


def test_group_points_from_rows():
    rows = study_rows([0.41, 0.42, 0.45]) + study_rows([0.5, 0.6, 0.7], group="g2")
    groups, skipped = gci.group_points(rows, "cd", "cd_sem")
    assert sorted(groups) == ["g1", "g2"]
    assert len(groups["g1"]) == 3
    assert skipped == ["m-bad: status failed", "m-bad: status failed"]


def test_csv_rows_are_parsed(tmp_path):
    rows = study_rows(seq(2, (1.0, 4.0 / 3.0, 2.0)))
    path = tmp_path / "results.csv"
    cols = ["name", "group", "status", "grid", "cells", "char_length", "cd", "cd_sem"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows({c: ("" if r.get(c) is None else r.get(c)) for c in cols} for r in rows)
    groups, _ = gci.group_points(gci.load_rows(path), "cd", "cd_sem")
    res = gci.analyse(groups["g1"])
    assert res["primary"]["order"] == pytest.approx(2.0)


def write_results(tmp_path, rows):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"rows": rows}))
    return path


def test_main_reports_an_estimate(tmp_path, capsys):
    path = write_results(tmp_path, study_rows(seq(2, (1.0, 4.0 / 3.0, 2.0)), [1e-5] * 3))
    out_json = tmp_path / "gci.json"
    assert gci.main([str(path), "--json", str(out_json)]) == 0
    out = capsys.readouterr().out
    assert "observed order p 2.000 (formal 2)" in out
    assert "cd = 1.01 +/- " in out
    assert "skipped m-bad: status failed" in out
    data = json.loads(out_json.read_text())
    assert data["groups"]["g1"]["primary"]["ok"]
    assert data["options"]["fs"] == 1.25


def test_main_refuses_bad_data(tmp_path, capsys):
    path = write_results(tmp_path, study_rows([0.40, 0.42, 0.41]))
    out_json = tmp_path / "gci.json"
    assert gci.main([str(path), "--json", str(out_json)]) == 1
    out = capsys.readouterr().out
    assert "no order reported: oscillatory convergence" in out
    assert "observed order" not in out
    assert json.loads(out_json.read_text())["groups"]["g1"]["primary"]["order"] is None


def test_main_input_errors(tmp_path, capsys):
    path = write_results(tmp_path, study_rows([0.41, 0.42, 0.45]))
    assert gci.main([str(path), "--group", "nope"]) == 2
    assert "no group 'nope'" in capsys.readouterr().err
    assert gci.main([str(path), "--value", "missing"]) == 2
    assert gci.main([str(tmp_path / "none.json")]) == 2


def test_plot(tmp_path):
    pytest.importorskip("matplotlib")
    path = write_results(tmp_path, study_rows(seq(2, (1.0, 4.0 / 3.0, 2.0)), [1e-5] * 3))
    png = tmp_path / "gci.png"
    assert gci.main([str(path), "--plot", str(png)]) == 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
