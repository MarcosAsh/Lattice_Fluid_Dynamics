"""
Export the anyobj MLP to JSON for browser inference.

Bundles weights, the z-score normalizer, and precomputed descriptors for
the bundled reference geometries so the browser needs no mesh processing.

Usage:
    python export_web.py    # anyobj_model_best.npz -> website/public/models
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import DESCRIPTOR_NAMES  # noqa: E402
from geometry import encode  # noqa: E402
from train import MODEL_OBJS  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "website" / "public" / "models" / "anyobj.json"


def rounded(arr):
    return np.round(np.asarray(arr, np.float64), 7).tolist()


def main():
    w = np.load(HERE / "anyobj_model_best.npz")
    norm = np.load(HERE / "anyobj_normalizer.npz")

    layers = []
    i = 0
    while f"w{i}" in w:
        wi = w[f"w{i}"]
        layers.append({"in": wi.shape[0], "out": wi.shape[1],
                       "w": rounded(wi.ravel()), "b": rounded(w[f"b{i}"])})
        i += 1

    descriptors = {name: rounded(encode(str(path))["descriptors"])
                   for name, path in MODEL_OBJS.items()}

    payload = {
        "layers": layers,
        "xMean": rounded(norm["x_mean"]),
        "xStd": rounded(norm["x_std"]),
        "yMean": rounded(norm["y_mean"]),
        "yStd": rounded(norm["y_std"]),
        "featureNames": list(DESCRIPTOR_NAMES) + ["log10_reynolds"],
        "targetNames": ["cd", "cl"],
        "descriptors": descriptors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(layers)} layers, "
          f"geometries: {', '.join(descriptors)})")


if __name__ == "__main__":
    main()
