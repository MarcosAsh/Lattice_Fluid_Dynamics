"""
Copy trained ML weights to the locations that actually consume them.

After training, weights sit wherever the training script dropped them. Two
consumers need them in fixed spots:

  - The headless simulation (and the Modal render worker) loads the
    super-resolution weights from ``simulation/assets/`` via the cwd-relative
    default path ``assets/sr_model.bin``. ``build_simulation`` clones that
    directory into the Modal volume, so committing the file is what deploys it.

  - The website ``/optimize`` page fetches the shape-optimization surrogate
    from ``website/public/models/shapeopt_model.bin`` and ``shapeopt_norm.bin``.

This script copies both sets into place, verifies the LTWS header on model
files, and prints what changed. It does not commit -- review the diff and
commit/push yourself, since that is what triggers the Modal rebuild and the
website deploy.

Usage:
    python ml/deploy_weights.py                 # deploy everything found
    python ml/deploy_weights.py --superres      # only the SR weights
    python ml/deploy_weights.py --shapeopt      # only the shapeopt weights
    python ml/deploy_weights.py --sr-dir build/ # custom source directory
"""

import argparse
import hashlib
import shutil
import struct
from pathlib import Path

# Repo root is two levels up from this file (ml/deploy_weights.py).
REPO = Path(__file__).resolve().parent.parent

LTWS_MAGIC = 0x4C545753  # 'SWTL' little-endian, the model.bin header


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


def _has_ltws_header(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(4)
    return len(head) == 4 and struct.unpack("<I", head)[0] == LTWS_MAGIC


def _deploy(label: str, src: Path, dst: Path, check_header: bool) -> bool:
    """Copy src to dst if it exists. Returns True if a file was deployed."""
    if not src.exists():
        print(f"  [skip] {label}: source not found ({_rel(src)})")
        return False
    if check_header and not _has_ltws_header(src):
        print(f"  [FAIL] {label}: {_rel(src)} is not an LTWS model file "
              f"(bad magic) -- refusing to deploy")
        return False

    if dst.exists() and dst.read_bytes() == src.read_bytes():
        print(f"  [ok]   {label}: already current ({_rel(dst)})")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"  [copy] {label}: {_rel(src)} -> {_rel(dst)} "
          f"({src.stat().st_size:,} B, md5={_md5(dst)})")
    return True


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def deploy_superres(sr_dir: Path) -> bool:
    print("Super-resolution -> simulation/assets (Modal render):")
    dst = REPO / "simulation" / "assets"
    changed = _deploy("sr_model", sr_dir / "sr_model.bin",
                      dst / "sr_model.bin", check_header=True)
    changed |= _deploy("sr_model_norm", sr_dir / "sr_model_norm.bin",
                       dst / "sr_model_norm.bin", check_header=False)
    return changed


def deploy_shapeopt(shapeopt_dir: Path) -> bool:
    print("Shape optimization -> website/public/models (website /optimize):")
    dst = REPO / "website" / "public" / "models"
    changed = _deploy("shapeopt_model", shapeopt_dir / "shapeopt_model.bin",
                      dst / "shapeopt_model.bin", check_header=True)
    # Training writes shapeopt_normalizer.bin; the website fetches
    # shapeopt_norm.bin. Bridge the name here.
    norm_src = shapeopt_dir / "shapeopt_normalizer.bin"
    if not norm_src.exists():
        norm_src = shapeopt_dir / "shapeopt_norm.bin"
    changed |= _deploy("shapeopt_norm", norm_src,
                       dst / "shapeopt_norm.bin", check_header=False)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--superres", action="store_true",
                        help="deploy only the super-resolution weights")
    parser.add_argument("--shapeopt", action="store_true",
                        help="deploy only the shape-optimization weights")
    parser.add_argument("--sr-dir", default=str(REPO / "assets"),
                        help="directory holding sr_model.bin + sr_model_norm.bin")
    parser.add_argument("--shapeopt-dir", default=str(REPO / "ml" / "shapeopt"),
                        help="directory holding shapeopt_model.bin + normalizer")
    args = parser.parse_args()

    # Default (neither flag) deploys both.
    do_sr = args.superres or not (args.superres or args.shapeopt)
    do_shape = args.shapeopt or not (args.superres or args.shapeopt)

    changed = False
    if do_sr:
        changed |= deploy_superres(Path(args.sr_dir))
    if do_shape:
        changed |= deploy_shapeopt(Path(args.shapeopt_dir))

    print()
    if changed:
        print("Weights updated. Next steps:")
        print("  - simulation/assets: commit + push so build_simulation "
              "rebuilds the Modal binary with the new weights")
        print("  - website/public/models: commit + redeploy the website")
    else:
        print("Nothing to deploy (sources missing or already current).")


if __name__ == "__main__":
    main()
