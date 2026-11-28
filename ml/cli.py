"""
Single entry point for the ML scripts.

Each subcommand runs the script below with the remaining arguments, so
`python ml/cli.py voxelize body.obj -o body.voxbin` is the same as
`python ml/voxelize.py body.obj -o body.voxbin`.

Usage:
    python ml/cli.py                      # list subcommands
    python ml/cli.py check-config ml/sweep_config.yaml
    python ml/cli.py train-shapeopt --data ml/shapeopt/dataset/shapeopt_data.sobin
    python ml/cli.py test
"""

import subprocess
import sys
from pathlib import Path

ML = Path(__file__).resolve().parent
ROOT = ML.parent

COMMANDS: dict[str, tuple[str, str]] = {
    "check-config": ("pipeline_config.py", "validate a pipeline YAML config"),
    "data-gen": ("data_gen.py", "run the Cd sweep on Modal"),
    "evaluate": ("evaluate.py", "evaluate the Cd/Cl surrogate"),
    "voxelize": ("voxelize.py", "voxelize an OBJ into a VOXB grid"),
    "preprocess": ("preprocess.py", "clean and validate an OBJ"),
    "convert": ("mesh_convert.py", "convert STL or OBJ to a clean triangle OBJ"),
    "study": ("study.py", "run solver cases and collect their manifests"),
    "gci": ("gci.py", "grid convergence index from study results"),
    "deploy-weights": ("deploy_weights.py", "copy trained weights where they are used"),
    "gen-shapes": ("shapeopt/gen_shapes.py", "generate FFD shape data"),
    "train-shapeopt": ("shapeopt/train_shapeopt.py", "train the shape surrogate"),
    "optimize": ("shapeopt/optimize.py", "optimize a shape through the surrogate"),
    "gen-pairs": ("superres/gen_pairs.py", "generate super-resolution pairs"),
    "train-superres": ("superres/train_superres.py", "train the super-resolution net"),
    "evaluate-superres": ("superres/evaluate_superres.py", "evaluate the super-resolution net"),
    "export-superres": ("superres/export_weights.py", "export super-resolution weights"),
    "superres-baselines": (
        "superres/evaluate_baselines.py",
        "compare the super-resolution net with interpolation",
    ),
}


def usage() -> str:
    width = max(len(name) for name in [*COMMANDS, "test"])
    lines = ["usage: lattice-ml <command> [args...]", "", "commands:"]
    lines += [f"  {name:<{width}}  {desc}" for name, (_, desc) in COMMANDS.items()]
    lines.append(f"  {'test':<{width}}  run the pytest suite")
    lines += ["", "Run a command with --help for its own options."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    name, rest = argv[0], argv[1:]
    if name == "test":
        cmd = [sys.executable, "-m", "pytest", str(ROOT / "tests"), *rest]
    elif name in COMMANDS:
        cmd = [sys.executable, str(ML / COMMANDS[name][0]), *rest]
    else:
        print(f"lattice-ml: unknown command {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
