"""
Export PyTorch super-resolution weights to LTWS binary format
for loading in the GLSL compute shader.

Usage:
    python export_weights.py --checkpoint sr_model_best.pt --output sr_model.bin
    python export_weights.py --checkpoint sr_model_best.pt --output sr_model.bin --normalizer sr_normalizer.npz
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

LTWS_MAGIC = 0x4C545753
LTWS_VERSION = 1


def export_ltws(state_dict: dict, output_path: str):
    """
    Export PyTorch state dict to LTWS binary format.

    Format:
        Header: magic (u32) | version (u32) | param_count (u32)
        Per param: ndim (u32) | shape (u32 x ndim) | data (f32 x numel)

    Linear layers store weight as (out_features, in_features) and bias
    as (out_features,). The GLSL shader reads weights row-major.
    """
    # Ordered list of parameter tensors
    params = []
    for name in sorted(state_dict.keys()):
        t = state_dict[name].cpu().float().numpy()
        params.append((name, t))

    with open(output_path, "wb") as f:
        f.write(struct.pack("<III", LTWS_MAGIC, LTWS_VERSION, len(params)))

        for name, arr in params:
            ndim = len(arr.shape)
            f.write(struct.pack("<I", ndim))
            for d in arr.shape:
                f.write(struct.pack("<I", d))
            f.write(arr.tobytes())

    total_params = sum(arr.size for _, arr in params)
    print(f"Exported {len(params)} tensors ({total_params:,} parameters)")
    print(f"  File: {output_path} ({Path(output_path).stat().st_size:,} bytes)")

    for name, arr in params:
        print(f"  {name}: {arr.shape}")


def export_normalizer(npz_path: str, output_path: str):
    """
    Export normalizer (mean, std) to a binary file.

    Format: float32[4] mean, float32[4] std (4 velocity channels)
    """
    data = np.load(npz_path)
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)

    with open(output_path, "wb") as f:
        f.write(mean.tobytes())
        f.write(std.tobytes())

    print(f"Exported normalizer: mean={mean}, std={std}")
    print(f"  File: {output_path}")


def compute_weight_offsets(state_dict: dict):
    """
    Compute flat byte offsets for each parameter in the packed weights
    buffer, for setting GLSL uniform offsets.
    """
    offset = 0
    print("\nGLSL weight offsets (float index):")
    for name in sorted(state_dict.keys()):
        t = state_dict[name]
        print(f"  {name}: offset={offset}, size={t.numel()}")
        offset += t.numel()
    print(f"  Total floats: {offset}")


def main():
    parser = argparse.ArgumentParser(
        description="Export super-resolution weights to LTWS format"
    )
    parser.add_argument("--checkpoint", required=True, help="PyTorch .pt file")
    parser.add_argument("--output", required=True, help="Output .bin file")
    parser.add_argument("--normalizer", help="Normalizer .npz file")
    parser.add_argument("--norm-output", help="Normalizer output .bin file")
    args = parser.parse_args()

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    export_ltws(state_dict, args.output)
    compute_weight_offsets(state_dict)

    if args.normalizer:
        norm_out = args.norm_output or args.output.replace(".bin", "_norm.bin")
        export_normalizer(args.normalizer, norm_out)


if __name__ == "__main__":
    main()
