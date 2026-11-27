"""
Validated loaders for the YAML configs.

Each loader returns the plain dict the pipelines already use, after
checking keys, types and ranges. All problems are reported at once.

Usage:
    from pipeline_config import load_sweep_config

    cfg = load_sweep_config("ml/sweep_config.yaml")

    python ml/pipeline_config.py ml/sweep_config.yaml        # kind is guessed
    python ml/pipeline_config.py --kind superres some.yaml
"""

import argparse
import difflib
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

GRID_RE = re.compile(r"^\d+x\d+x\d+$")


class ConfigError(ValueError):
    """Raised with every problem found in a config file."""

    def __init__(self, path: str | Path, problems: list[str]) -> None:
        self.path = str(path)
        self.problems = problems
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"invalid config {self.path}:\n{lines}")


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_str(v: Any) -> bool:
    return isinstance(v, str)


@dataclass(frozen=True)
class Field:
    key: str
    kind: str  # int, number, str, bool, grid, list, dict
    required: bool = True
    min: float | None = None
    max: float | None = None
    item: str | None = None  # element kind for lists
    fields: tuple["Field", ...] = ()  # sub-schema for dicts
    length: int | None = None  # exact list length
    nonempty: bool = False
    exclusive_min: bool = False
    check: Callable[[Any], str | None] | None = None


_TYPE_CHECKS: dict[str, tuple[Callable[[Any], bool], str]] = {
    "int": (_is_int, "an integer"),
    "number": (_is_number, "a number"),
    "str": (_is_str, "a string"),
    "bool": (lambda v: isinstance(v, bool), "true or false"),
    "grid": (lambda v: _is_str(v) and bool(GRID_RE.match(v)), 'a grid like "128x96x96"'),
    "list": (lambda v: isinstance(v, list), "a list"),
    "dict": (lambda v: isinstance(v, dict), "a mapping"),
}


def _validate_value(f: Field, value: Any, where: str, problems: list[str]) -> None:
    ok, expected = _TYPE_CHECKS[f.kind]
    if not ok(value):
        problems.append(f"{where}: expected {expected}, got {value!r}")
        return
    if f.kind in ("int", "number"):
        if f.min is not None and (value <= f.min if f.exclusive_min else value < f.min):
            op = ">" if f.exclusive_min else ">="
            problems.append(f"{where}: must be {op} {f.min:g}, got {value!r}")
        if f.max is not None and value > f.max:
            problems.append(f"{where}: must be <= {f.max:g}, got {value!r}")
    if f.kind == "list":
        if f.nonempty and not value:
            problems.append(f"{where}: must not be empty")
        if f.length is not None and len(value) != f.length:
            problems.append(f"{where}: needs exactly {f.length} entries, got {len(value)}")
        if f.item:
            sub = Field(f.key, f.item, min=f.min, max=f.max, exclusive_min=f.exclusive_min)
            for i, v in enumerate(value):
                _validate_value(sub, v, f"{where}[{i}]", problems)
    if f.kind == "dict" and f.fields:
        _validate_mapping(value, f.fields, f"{where}.", problems)
    if f.check is not None:
        msg = f.check(value)
        if msg:
            problems.append(f"{where}: {msg}")


def _validate_mapping(
    data: dict, fields: tuple[Field, ...], prefix: str, problems: list[str]
) -> None:
    known = {f.key for f in fields}
    for key in data:
        if key not in known:
            near = difflib.get_close_matches(str(key), sorted(known), n=1)
            hint = f" (did you mean {near[0]!r}?)" if near else ""
            problems.append(f"{prefix}{key}: unknown key{hint}")
    for f in fields:
        where = f"{prefix}{f.key}"
        if f.key not in data:
            if f.required:
                problems.append(f"{where}: missing required key")
            continue
        _validate_value(f, data[f.key], where, problems)


def _wind_range(v: dict) -> str | None:
    if all(_is_number(v.get(k)) for k in ("min", "max")) and v["min"] > v["max"]:
        return "min must not exceed max"
    return None


def _cd_range(v: dict) -> str | None:
    if all(_is_number(v.get(k)) for k in ("cd_min", "cd_max")) and v["cd_min"] >= v["cd_max"]:
        return "cd_min must be below cd_max"
    return None


SWEEP_SCHEMA = (
    Field("models", "list", required=False, item="str"),
    Field("meshes_dir", "str", required=False),
    Field(
        "wind_speeds",
        "dict",
        fields=(
            Field("min", "number", min=0, exclusive_min=True),
            Field("max", "number", min=0, exclusive_min=True),
            Field("step", "number", min=0, exclusive_min=True),
        ),
        check=_wind_range,
    ),
    Field("reynolds_numbers", "list", item="number", min=0, nonempty=True),
    Field("grid", "grid", required=False),
    Field("duration", "number", min=0, exclusive_min=True),
    Field("viz_mode", "int", required=False, min=0),
    Field("collision_mode", "int", required=False, min=0),
    Field("dataset_dir", "str", required=False),
    Field("max_concurrent", "int", required=False, min=1),
    Field("retry_limit", "int", required=False, min=0),
    Field("delay_between_batches", "number", required=False, min=0),
    Field(
        "quality",
        "dict",
        required=False,
        fields=(
            Field("cd_max", "number"),
            Field("cd_min", "number"),
            Field("min_cd_samples", "int", min=1),
            Field("tail_fraction", "number", required=False, min=0, max=1, exclusive_min=True),
            Field("convergence_threshold", "number", min=0, exclusive_min=True),
        ),
        check=_cd_range,
    ),
)

SHAPEOPT_SCHEMA = (
    Field("base_models", "list", item="str", nonempty=True),
    Field("ffd_grid", "list", item="int", min=2, length=3),
    Field("num_samples", "int", min=1),
    Field("displacement_range", "number", min=0, max=1, exclusive_min=True),
    Field("enforce_symmetry", "bool", required=False),
    Field("wind_speed", "number", min=0, exclusive_min=True),
    Field("reynolds", "number", min=0),
    Field("duration", "number", min=0, exclusive_min=True),
    Field("collision_mode", "int", required=False, min=0),
    Field("max_concurrent", "int", required=False, min=1),
    Field("retry_limit", "int", required=False, min=0),
    Field("optimize_steps", "int", required=False, min=1),
    Field("optimize_lr", "number", required=False, min=0, exclusive_min=True),
)


def _odd(v: int) -> str | None:
    return None if v % 2 == 1 else "must be odd"


SUPERRES_SCHEMA = (
    Field("coarse_grid", "grid"),
    Field("fine_grid", "grid"),
    Field("upscale_factor", "int", required=False, min=2),
    Field("models", "list", item="str", nonempty=True),
    Field("wind_speeds", "list", item="number", min=0, exclusive_min=True, nonempty=True),
    Field("reynolds_numbers", "list", item="number", min=0, nonempty=True),
    Field("duration", "number", min=0, exclusive_min=True),
    Field("vtk_interval", "int", min=1),
    Field("patch_neighborhood", "int", required=False, min=1, check=_odd),
    Field("max_concurrent", "int", required=False, min=1),
)

SCHEMAS = {
    "sweep": SWEEP_SCHEMA,
    "shapeopt": SHAPEOPT_SCHEMA,
    "superres": SUPERRES_SCHEMA,
}


def _cross_checks(kind: str, data: dict, problems: list[str]) -> None:
    if kind == "sweep" and not data.get("models") and not data.get("meshes_dir"):
        problems.append("models: need at least one model or a meshes_dir")
    if kind == "superres":
        try:
            dims = [
                tuple(int(n) for n in str(data[k]).split("x")) for k in ("coarse_grid", "fine_grid")
            ]
        except (KeyError, ValueError):
            return
        factor = data.get("upscale_factor", 2)
        if _is_int(factor) and tuple(c * factor for c in dims[0]) != dims[1]:
            problems.append(f"fine_grid: must be coarse_grid times upscale_factor ({factor})")


def validate(kind: str, data: Any) -> list[str]:
    """Return a list of problems, empty when the config is valid."""
    if not isinstance(data, dict):
        return ["top level must be a mapping of keys to values"]
    problems: list[str] = []
    _validate_mapping(data, SCHEMAS[kind], "", problems)
    if not problems:
        _cross_checks(kind, data, problems)
    return problems


def load_config(kind: str, path: str | Path) -> dict:
    """Read a YAML file and validate it against the schema for `kind`."""
    if kind not in SCHEMAS:
        raise ValueError(f"unknown config kind {kind!r}, expected one of {sorted(SCHEMAS)}")
    try:
        with Path(path).open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(path, [f"not valid YAML: {e}"]) from e
    problems = validate(kind, data)
    if problems:
        raise ConfigError(path, problems)
    return data


def load_sweep_config(path: str | Path) -> dict:
    return load_config("sweep", path)


def load_shapeopt_config(path: str | Path) -> dict:
    return load_config("shapeopt", path)


def load_superres_config(path: str | Path) -> dict:
    return load_config("superres", path)


def guess_kind(data: Any) -> str | None:
    """Pick the config kind from its distinguishing keys."""
    if not isinstance(data, dict):
        return None
    if "base_models" in data or "ffd_grid" in data:
        return "shapeopt"
    if "coarse_grid" in data or "vtk_interval" in data:
        return "superres"
    if "reynolds_numbers" in data:
        return "sweep"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a pipeline YAML config")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--kind", choices=sorted(SCHEMAS), help="skip the guess from the keys")
    args = ap.parse_args(argv)

    status = 0
    for path in args.files:
        kind = args.kind
        if kind is None:
            try:
                kind = guess_kind(yaml.safe_load(path.read_text()))
            except (OSError, yaml.YAMLError) as e:
                print(f"{path}: {e}", file=sys.stderr)
                status = 1
                continue
            if kind is None:
                print(f"{path}: cannot tell the config kind, pass --kind", file=sys.stderr)
                status = 1
                continue
        try:
            load_config(kind, path)
        except (ConfigError, OSError) as e:
            print(e, file=sys.stderr)
            status = 1
        else:
            print(f"{path}: ok ({kind})")
    return status


if __name__ == "__main__":
    sys.exit(main())
