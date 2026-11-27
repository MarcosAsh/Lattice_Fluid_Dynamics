import copy
from pathlib import Path

import pytest
import yaml

import pipeline_config as cfg

ML = Path(__file__).resolve().parents[1] / "ml"

SHIPPED = [
    ("sweep", ML / "sweep_config.yaml"),
    ("sweep", ML / "anyobj" / "sweep.yaml"),
    ("shapeopt", ML / "shapeopt" / "shapeopt_config.yaml"),
    ("superres", ML / "superres" / "superres_config.yaml"),
    ("superres", ML / "superres" / "superres_config_test.yaml"),
]


@pytest.mark.parametrize(("kind", "path"), SHIPPED, ids=[p.name for _, p in SHIPPED])
def test_shipped_configs_are_valid(kind, path):
    assert cfg.load_config(kind, path)


@pytest.mark.parametrize(("kind", "path"), SHIPPED, ids=[p.name for _, p in SHIPPED])
def test_kind_is_guessed_from_keys(kind, path):
    assert cfg.guess_kind(yaml.safe_load(path.read_text())) == kind


def load(kind, path):
    return yaml.safe_load(path.read_text())


@pytest.fixture
def sweep():
    return load("sweep", ML / "sweep_config.yaml")


def test_loader_returns_the_plain_dict(sweep):
    assert cfg.load_sweep_config(ML / "sweep_config.yaml") == sweep


def test_typo_gets_a_suggestion(sweep):
    sweep["reynold_numbers"] = sweep.pop("reynolds_numbers")
    problems = cfg.validate("sweep", sweep)
    assert any("reynold_numbers" in p and "did you mean 'reynolds_numbers'" in p for p in problems)
    assert any("reynolds_numbers: missing" in p for p in problems)


def test_all_problems_are_reported_together(sweep):
    sweep["duration"] = -1
    sweep["grid"] = "big"
    sweep["wind_speeds"]["step"] = 0
    sweep["max_concurrent"] = 2.5
    problems = cfg.validate("sweep", sweep)
    assert len(problems) == 4
    assert any(p.startswith("wind_speeds.step") for p in problems)


def test_error_message_lists_the_file(tmp_path, sweep):
    sweep["duration"] = "long"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(sweep))
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load_sweep_config(path)
    assert str(path) in str(exc.value)
    assert "duration: expected a number" in str(exc.value)
    assert exc.value.problems


def test_booleans_are_not_numbers(sweep):
    sweep["duration"] = True
    assert cfg.validate("sweep", sweep)


def test_quality_ranges(sweep):
    sweep["quality"]["cd_min"] = 20.0
    assert any("cd_min must be below cd_max" in p for p in cfg.validate("sweep", sweep))
    sweep["quality"]["cd_min"] = 0.0
    sweep["quality"]["tail_fraction"] = 1.5
    assert any("tail_fraction" in p for p in cfg.validate("sweep", sweep))


def test_wind_range(sweep):
    sweep["wind_speeds"].update(min=3.0, max=1.0)
    assert any("min must not exceed max" in p for p in cfg.validate("sweep", sweep))


def test_sweep_needs_a_body(sweep):
    sweep["models"] = []
    assert any("at least one model" in p for p in cfg.validate("sweep", sweep))
    sweep["meshes_dir"] = "anyobj/meshes"
    assert not cfg.validate("sweep", sweep)


def test_shapeopt_checks():
    good = load("shapeopt", ML / "shapeopt" / "shapeopt_config.yaml")
    bad = copy.deepcopy(good)
    bad["ffd_grid"] = [4, 3]
    bad["displacement_range"] = 2.0
    problems = cfg.validate("shapeopt", bad)
    assert any("ffd_grid: needs exactly 3" in p for p in problems)
    assert any("displacement_range" in p for p in problems)


def test_superres_grid_ratio_and_odd_patch():
    good = load("superres", ML / "superres" / "superres_config.yaml")
    bad = copy.deepcopy(good)
    bad["fine_grid"] = "100x96x96"
    assert any("coarse_grid times upscale_factor" in p for p in cfg.validate("superres", bad))
    bad = copy.deepcopy(good)
    bad["patch_neighborhood"] = 4
    assert any("must be odd" in p for p in cfg.validate("superres", bad))


def test_non_mapping_and_bad_yaml(tmp_path):
    assert cfg.validate("sweep", ["a"])
    path = tmp_path / "x.yaml"
    path.write_text("a: [1, 2")
    with pytest.raises(cfg.ConfigError, match="not valid YAML"):
        cfg.load_sweep_config(path)


def test_unknown_kind():
    with pytest.raises(ValueError, match="unknown config kind"):
        cfg.load_config("nope", "x.yaml")


def test_cli(tmp_path, capsys):
    assert cfg.main([str(ML / "sweep_config.yaml")]) == 0
    assert "ok (sweep)" in capsys.readouterr().out
    bad = tmp_path / "bad.yaml"
    bad.write_text("reynolds_numbers: []\n")
    assert cfg.main([str(bad)]) == 1
    assert "reynolds_numbers" in capsys.readouterr().err
    unknown = tmp_path / "u.yaml"
    unknown.write_text("x: 1\n")
    assert cfg.main([str(unknown)]) == 1


def test_shapeopt_seed_key():
    good = load("shapeopt", ML / "shapeopt" / "shapeopt_config.yaml")
    good["seed"] = 7
    assert not cfg.validate("shapeopt", good)
    good["seed"] = -1
    assert cfg.validate("shapeopt", good)
