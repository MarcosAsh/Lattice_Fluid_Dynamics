import pytest

pytest.importorskip("modal")

import grid_convergence as gc


def test_cd_stats_uses_the_back_half():
    mean, std, rel = gc._cd_stats([100.0, 100.0, 2.0, 4.0])
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(1.0)
    assert rel == pytest.approx(1.0 / 3.0)


def test_cd_stats_empty_is_nan():
    assert all(v != v for v in gc._cd_stats([]))


def test_richardson_recovers_first_order_limit():
    # successive differences halve, so the observed order is 1
    rows = [{"cd_mean": v} for v in (4.0, 3.0, 2.5)]
    out = gc._richardson(rows)
    assert out["order"] == pytest.approx(1.0)
    assert out["cd_extrapolated"] == pytest.approx(2.0)


def test_richardson_gives_up_on_bad_input():
    assert gc._richardson([{"cd_mean": 1.0}] * 2) is None
    assert gc._richardson([{"cd_mean": 1.0}] * 3) is None
