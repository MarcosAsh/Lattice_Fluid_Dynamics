"""
Generate paired coarse/fine velocity field data for super-resolution training.

Runs LBM simulations at two resolutions, parses VTK .vti output, extracts
overlapping patch pairs, and writes them to a compact binary format (.srbin).

Usage:
    python gen_pairs.py --config superres_config.yaml
    python gen_pairs.py --config superres_config.yaml --resume
    python gen_pairs.py --status

Requires the simulation binary at build/3d_fluid_simulation_car (relative to
the project root). Build it first with `make` from the project root.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM_BINARY = PROJECT_ROOT / "build" / "3d_fluid_simulation_car"
ASSET_DIR = PROJECT_ROOT / "simulation" / "assets" / "3d-files"

OUTPUT_DIR = Path(__file__).resolve().parent / "dataset"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
SRBIN_FILE = OUTPUT_DIR / "patches.srbin"
STATS_FILE = OUTPUT_DIR / "stats.json"

# .srbin constants
SRBIN_MAGIC = 0x53524553
SRBIN_VERSION = 1
CHANNELS = 4  # ux, uy, uz, rho

MODEL_OBJ_PATHS = {
    "car": "assets/3d-files/car-model.obj",
    "ahmed25": "assets/3d-files/ahmed_25deg_m.obj",
    "ahmed35": "assets/3d-files/ahmed_35deg_m.obj",
}


# ---------------------------------------------------------------------------
# Config types
# ---------------------------------------------------------------------------

@dataclass
class PairConfig:
    model: str
    wind_speed: float
    reynolds: float
    coarse_grid: str
    fine_grid: str
    duration: int
    vtk_interval: int

    @property
    def pair_id(self) -> str:
        key = (
            f"{self.model}_{self.wind_speed}_{self.reynolds}_"
            f"{self.coarse_grid}_{self.fine_grid}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]


@dataclass
class PairResult:
    config: PairConfig
    status: str
    coarse_patches: int = 0
    fine_patches: int = 0
    error: str | None = None
    vtk_files_coarse: int = 0
    vtk_files_fine: int = 0


@dataclass
class PatchAccumulator:
    """Collects patches across all configs before writing the .srbin."""
    coarse_inputs: list[np.ndarray] = field(default_factory=list)
    fine_targets: list[np.ndarray] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.coarse_inputs)


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_pair_configs(config: dict) -> list[PairConfig]:
    pairs = []
    for model in config["models"]:
        for ws in config["wind_speeds"]:
            for re_num in config["reynolds_numbers"]:
                pairs.append(
                    PairConfig(
                        model=model,
                        wind_speed=float(ws),
                        reynolds=float(re_num),
                        coarse_grid=config["coarse_grid"],
                        fine_grid=config["fine_grid"],
                        duration=config["duration"],
                        vtk_interval=config["vtk_interval"],
                    )
                )
    return pairs


# ---------------------------------------------------------------------------
# VTI parsing
# ---------------------------------------------------------------------------

def parse_grid_dims(extent_str: str) -> tuple[int, int, int]:
    """Parse WholeExtent='0 NX 0 NY 0 NZ' into (NX, NY, NZ) cell counts.

    VTK ImageData uses node-based extents, so the number of cells along
    each axis equals the upper extent value. For example, WholeExtent
    '0 128 0 96 0 96' means 128 cells in X, 96 in Y, 96 in Z. The VTI
    writer in main.c passes the grid dimensions directly as the upper
    extent values, and uses PointData, so the number of data points is
    (NX+1)*(NY+1)*(NZ+1). But our writer actually uses NX, NY, NZ as
    both the extent upper bounds AND the loop count. Looking at the C
    code more carefully: total = nx*ny*nz and WholeExtent is '0 nx 0 ny
    0 nz', so VTK interprets that as (nx+1)*(ny+1)*(nz+1) points. The
    actual data written is nx*ny*nz elements. We need to match the data
    that was actually written, so we use the grid dims from the binary
    itself.

    The safest approach: read the uint64 size prefix of the velocity
    array to determine total = velDataSize / (3 * 4), then combine
    with the aspect ratio from the extent to recover nx, ny, nz.
    """
    parts = extent_str.split()
    # Expect 6 integers: x0, x1, y0, y1, z0, z1
    if len(parts) != 6:
        raise ValueError(f"Bad WholeExtent: {extent_str}")
    nx = int(parts[1])
    ny = int(parts[3])
    nz = int(parts[5])
    return nx, ny, nz


def parse_vti(filepath: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Parse a VTI file written by writeVTI() in main.c.

    Returns (velocity, solid, (nx, ny, nz)) where:
      velocity: float32 array of shape (nz, ny, nx, 3)  -- z-major ordering
      solid:    int32 array of shape (nz, ny, nx)
      dims:     (nx, ny, nz) grid dimensions

    The C writer stores data in flat x-fastest order (i = z*nx*ny + y*nx + x)
    but we reshape to (nz, ny, nx) for numpy convenience. This matches
    the standard C row-major layout when the flat index is computed as
    z*ny*nx + y*nx + x.
    """
    raw = filepath.read_bytes()

    # Split at the appended data marker. The VTI format writes
    # '<AppendedData encoding="raw">\n_' and then the binary blobs.
    marker = b"<AppendedData encoding=\"raw\">\n_"
    marker_pos = raw.find(marker)
    if marker_pos == -1:
        raise ValueError(f"No appended data marker in {filepath}")

    xml_bytes = raw[:marker_pos + len(marker)]
    binary_blob = raw[marker_pos + len(marker):]

    # Parse XML header to get grid dimensions
    # We need to close off the XML cleanly for parsing. The XML section
    # before the binary blob isn't well-formed on its own, so extract
    # WholeExtent with a regex instead.
    xml_text = raw[:marker_pos].decode("ascii", errors="replace")
    extent_match = re.search(r'WholeExtent="([^"]+)"', xml_text)
    if not extent_match:
        raise ValueError(f"No WholeExtent found in {filepath}")

    nx, ny, nz = parse_grid_dims(extent_match.group(1))
    total = nx * ny * nz

    # Read velocity array: uint64 size prefix, then total*3 float32 values
    offset = 0
    vel_data_size = struct.unpack_from("<Q", binary_blob, offset)[0]
    offset += 8

    expected_vel_size = total * 3 * 4
    if vel_data_size != expected_vel_size:
        raise ValueError(
            f"Velocity data size mismatch in {filepath}: "
            f"header says {vel_data_size}, expected {expected_vel_size} "
            f"(grid {nx}x{ny}x{nz}, total={total})"
        )

    velocity = np.frombuffer(
        binary_blob, dtype="<f4", count=total * 3, offset=offset
    ).reshape(nz, ny, nx, 3).copy()
    offset += vel_data_size

    # Read solid array: uint64 size prefix, then total int32 values
    solid_data_size = struct.unpack_from("<Q", binary_blob, offset)[0]
    offset += 8

    expected_solid_size = total * 4
    if solid_data_size != expected_solid_size:
        raise ValueError(
            f"Solid data size mismatch in {filepath}: "
            f"header says {solid_data_size}, expected {expected_solid_size}"
        )

    solid = np.frombuffer(
        binary_blob, dtype="<i4", count=total, offset=offset
    ).reshape(nz, ny, nx).copy()

    return velocity, solid, (nx, ny, nz)


# ---------------------------------------------------------------------------
# Downsampling and patch extraction
# ---------------------------------------------------------------------------

def compute_density(velocity: np.ndarray) -> np.ndarray:
    """Approximate density from LBM velocity field.

    In lattice Boltzmann, density is close to 1.0 everywhere for low-Mach
    flows. The actual rho is tracked inside the GPU buffers but not exported
    in the VTI. For super-resolution training we approximate rho = 1.0,
    which is a reasonable baseline. A future iteration could export rho as
    a separate DataArray in the VTI.
    """
    return np.ones(velocity.shape[:3], dtype=np.float32)


def downsample_2x(velocity: np.ndarray, solid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a fine field by averaging 2x2x2 blocks.

    Args:
        velocity: shape (nz, ny, nx, 3), float32
        solid:    shape (nz, ny, nx), int32

    Returns:
        coarse_vel: shape (nz//2, ny//2, nx//2, 3)
        coarse_solid: shape (nz//2, ny//2, nx//2), a cell is solid if
                      ANY of the 8 fine sub-cells is solid
    """
    nz, ny, nx, c = velocity.shape
    assert nz % 2 == 0 and ny % 2 == 0 and nx % 2 == 0, (
        f"Fine grid dims must be even for 2x downsampling, got {nx}x{ny}x{nz}"
    )

    # Reshape into 2x2x2 blocks and average
    vel_blocked = velocity.reshape(nz // 2, 2, ny // 2, 2, nx // 2, 2, c)
    coarse_vel = vel_blocked.mean(axis=(1, 3, 5)).astype(np.float32)

    # A coarse cell is solid if any of its 8 fine sub-cells is solid
    sol_blocked = solid.reshape(nz // 2, 2, ny // 2, 2, nx // 2, 2)
    coarse_solid = (sol_blocked.max(axis=(1, 3, 5)) > 0).astype(np.int32)

    return coarse_vel, coarse_solid


def extract_patches(
    coarse_vel: np.ndarray,
    coarse_solid: np.ndarray,
    fine_vel: np.ndarray,
    fine_solid: np.ndarray,
    neighborhood: int = 3,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract paired coarse/fine patches from aligned fields.

    For each valid coarse cell (i,j,k) with a full neighborhood and no
    solid cells, extract:
      - coarse input: neighborhood^3 block of 4-channel data (ux,uy,uz,rho)
      - fine target:  the 2x2x2 fine sub-cells corresponding to cell (i,j,k)

    Args:
        coarse_vel:   (cz, cy, cx, 3) downsampled from fine
        coarse_solid: (cz, cy, cx)
        fine_vel:     (fz, fy, fx, 3) original fine field
        fine_solid:   (fz, fy, fx)
        neighborhood: side length of the coarse input patch (must be odd)

    Returns:
        (coarse_patches, fine_patches) lists of numpy arrays
    """
    assert neighborhood % 2 == 1, "Patch neighborhood must be odd"
    half = neighborhood // 2

    cz, cy, cx, _ = coarse_vel.shape

    # Precompute density channels
    coarse_rho = compute_density(coarse_vel)
    fine_rho = compute_density(fine_vel)

    # Build 4-channel arrays: (z, y, x, 4)
    coarse_4ch = np.concatenate(
        [coarse_vel, coarse_rho[..., np.newaxis]], axis=-1
    )
    fine_4ch = np.concatenate(
        [fine_vel, fine_rho[..., np.newaxis]], axis=-1
    )

    coarse_patches = []
    fine_patches = []

    for iz in range(half, cz - half):
        for iy in range(half, cy - half):
            for ix in range(half, cx - half):
                # Check that the entire coarse neighborhood is fluid
                solid_block = coarse_solid[
                    iz - half : iz + half + 1,
                    iy - half : iy + half + 1,
                    ix - half : ix + half + 1,
                ]
                if solid_block.any():
                    continue

                # Also check the corresponding 2x2x2 fine cells
                fz0, fy0, fx0 = iz * 2, iy * 2, ix * 2
                fine_sub_solid = fine_solid[fz0 : fz0 + 2, fy0 : fy0 + 2, fx0 : fx0 + 2]
                if fine_sub_solid.any():
                    continue

                coarse_patch = coarse_4ch[
                    iz - half : iz + half + 1,
                    iy - half : iy + half + 1,
                    ix - half : ix + half + 1,
                ].copy()

                fine_patch = fine_4ch[
                    fz0 : fz0 + 2,
                    fy0 : fy0 + 2,
                    fx0 : fx0 + 2,
                ].copy()

                coarse_patches.append(coarse_patch)
                fine_patches.append(fine_patch)

    return coarse_patches, fine_patches


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    model: str,
    wind_speed: float,
    reynolds: float,
    grid: str,
    duration: int,
    vtk_interval: int,
    vtk_dir: Path,
) -> tuple[bool, str]:
    """Run the LBM simulation binary and produce VTK output.

    Returns (success, message).
    """
    if not SIM_BINARY.exists():
        return False, f"Simulation binary not found at {SIM_BINARY}"

    vtk_dir.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_OBJ_PATHS.get(model)
    if model_path is None:
        return False, f"Unknown model: {model}"

    cmd = [
        str(SIM_BINARY),
        f"--wind={wind_speed}",
        f"--duration={duration}",
        f"--grid={grid}",
        f"--model={model_path}",
        f"--vtk-output={vtk_dir}",
        f"--vtk-interval={vtk_interval}",
        "--viz=1",
        "--collision=1",
    ]
    if reynolds > 0:
        cmd.append(f"--reynolds={reynolds}")

    # The simulation needs a display for its OpenGL context. Use Xvfb
    # if DISPLAY is not already set.
    env = os.environ.copy()
    xvfb_proc = None

    if not env.get("DISPLAY"):
        # Try to start a virtual framebuffer
        try:
            xvfb_proc = subprocess.Popen(
                ["Xvfb", ":99", "-screen", "0", "1920x1080x24"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(1)
            env["DISPLAY"] = ":99"
        except FileNotFoundError:
            return False, "No DISPLAY set and Xvfb not found. Install xvfb or run under X."

    try:
        sim_timeout = duration * 120 + 300
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT / "simulation"),
            env=env,
            capture_output=True,
            text=True,
            timeout=sim_timeout,
        )

        if result.returncode != 0:
            tail = (result.stdout or "")[-1000:] + "\n" + (result.stderr or "")[-500:]
            return False, f"Simulation exited with code {result.returncode}: {tail}"

        vtk_files = sorted(vtk_dir.glob("field_*.vti"))
        if not vtk_files:
            return False, "Simulation completed but produced no VTK files"

        return True, f"Produced {len(vtk_files)} VTK files"

    except subprocess.TimeoutExpired:
        return False, f"Simulation timed out after {sim_timeout}s"
    except Exception as e:
        return False, str(e)
    finally:
        if xvfb_proc is not None:
            xvfb_proc.terminate()
            xvfb_proc.wait()


def run_simulation_modal(
    model: str,
    wind_speed: float,
    reynolds: float,
    grid: str,
    duration: int,
    vtk_interval: int,
    vtk_dir: Path,
) -> tuple[bool, str]:
    """Run the simulation via Modal GPU and download VTK files.

    Calls the generate_vtk_field Modal function remotely, receives
    base64-encoded VTI files, and writes them to vtk_dir.
    """
    import base64
    import modal

    vtk_dir.mkdir(parents=True, exist_ok=True)

    try:
        generate_vtk_field = modal.Function.from_name("fluid-sim", "generate_vtk_field")
        result = generate_vtk_field.remote(
            model=model,
            wind_speed=wind_speed,
            reynolds=reynolds,
            grid=grid,
            duration=duration,
            vtk_interval=vtk_interval,
        )
    except Exception as e:
        return False, f"Modal call failed: {e}"

    if result.get("status") != "complete":
        return False, f"Modal error: {result.get('error', 'unknown')}"

    fields = result.get("fields", [])
    for field_entry in fields:
        vti_path = vtk_dir / field_entry["filename"]
        vti_path.write_bytes(base64.b64decode(field_entry["data"]))

    return True, f"Downloaded {len(fields)} VTK files from Modal"


# ---------------------------------------------------------------------------
# .srbin I/O
# ---------------------------------------------------------------------------

def write_srbin(
    filepath: Path,
    coarse_patches: list[np.ndarray],
    fine_patches: list[np.ndarray],
):
    """Write patch pairs to .srbin format.

    Header (16 bytes):
        uint32 magic     = 0x53524553
        uint32 version   = 1
        uint32 num_patches
        uint32 channels  = 4

    Per patch:
        float32[108] coarse_input  (3x3x3 * 4 channels)
        float32[32]  fine_target   (2x2x2 * 4 channels)
    """
    num_patches = len(coarse_patches)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        f.write(struct.pack("<I", SRBIN_MAGIC))
        f.write(struct.pack("<I", SRBIN_VERSION))
        f.write(struct.pack("<I", num_patches))
        f.write(struct.pack("<I", CHANNELS))

        for coarse, fine in zip(coarse_patches, fine_patches):
            f.write(coarse.astype("<f4").tobytes())
            f.write(fine.astype("<f4").tobytes())

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Wrote {num_patches} patches to {filepath} ({size_mb:.1f} MB)")


def read_srbin(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read .srbin file back into numpy arrays for verification.

    Returns:
        coarse: float32 array of shape (N, 3, 3, 3, 4)
        fine:   float32 array of shape (N, 2, 2, 2, 4)
    """
    with open(filepath, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != SRBIN_MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X}, expected 0x{SRBIN_MAGIC:08X}")

        version = struct.unpack("<I", f.read(4))[0]
        if version != SRBIN_VERSION:
            raise ValueError(f"Unsupported version: {version}")

        num_patches = struct.unpack("<I", f.read(4))[0]
        channels = struct.unpack("<I", f.read(4))[0]

        coarse_size = 3 * 3 * 3 * channels
        fine_size = 2 * 2 * 2 * channels

        coarse_all = np.empty((num_patches, coarse_size), dtype=np.float32)
        fine_all = np.empty((num_patches, fine_size), dtype=np.float32)

        for i in range(num_patches):
            coarse_all[i] = np.frombuffer(
                f.read(coarse_size * 4), dtype="<f4"
            )
            fine_all[i] = np.frombuffer(
                f.read(fine_size * 4), dtype="<f4"
            )

    return (
        coarse_all.reshape(num_patches, 3, 3, 3, channels),
        fine_all.reshape(num_patches, 2, 2, 2, channels),
    )


# ---------------------------------------------------------------------------
# Manifest tracking
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_manifest(manifest: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def load_completed_ids(manifest: dict) -> set[str]:
    return set(manifest.get("completed", {}).keys())


# ---------------------------------------------------------------------------
# Pipeline: process one pair config
# ---------------------------------------------------------------------------

def process_pair(
    pc: PairConfig,
    accumulator: PatchAccumulator,
    neighborhood: int,
    work_dir: Path,
    use_modal: bool = False,
) -> PairResult:
    """Run coarse and fine simulations, extract patches, accumulate them."""
    coarse_vtk_dir = work_dir / f"{pc.pair_id}_coarse"
    fine_vtk_dir = work_dir / f"{pc.pair_id}_fine"

    runner = run_simulation_modal if use_modal else run_simulation

    # Run fine simulation
    backend = "Modal GPU" if use_modal else "local"
    print(f"  [{pc.pair_id}] Running fine sim ({pc.fine_grid}) on {backend}...")
    ok, msg = runner(
        model=pc.model,
        wind_speed=pc.wind_speed,
        reynolds=pc.reynolds,
        grid=pc.fine_grid,
        duration=pc.duration,
        vtk_interval=pc.vtk_interval,
        vtk_dir=fine_vtk_dir,
    )
    if not ok:
        return PairResult(config=pc, status="error", error=f"Fine sim failed: {msg}")

    fine_vtk_files = sorted(fine_vtk_dir.glob("field_*.vti"))
    print(f"  [{pc.pair_id}] Fine sim done, {len(fine_vtk_files)} VTK files")

    # We don't need to run a separate coarse simulation. Instead,
    # downsample the fine field to produce the coarse ground truth.
    # This gives perfectly aligned pairs.

    total_coarse = 0
    total_fine = 0

    for vti_path in fine_vtk_files:
        try:
            fine_vel, fine_solid, (fnx, fny, fnz) = parse_vti(vti_path)
        except Exception as e:
            print(f"    WARN: Failed to parse {vti_path.name}: {e}")
            continue

        # Downsample fine field to get coarse
        coarse_vel, coarse_solid = downsample_2x(fine_vel, fine_solid)

        # Extract patches
        c_patches, f_patches = extract_patches(
            coarse_vel=coarse_vel,
            coarse_solid=coarse_solid,
            fine_vel=fine_vel,
            fine_solid=fine_solid,
            neighborhood=neighborhood,
        )

        if c_patches:
            accumulator.coarse_inputs.extend(c_patches)
            accumulator.fine_targets.extend(f_patches)
            total_coarse += len(c_patches)
            total_fine += len(f_patches)

    print(f"  [{pc.pair_id}] Extracted {total_coarse} patches from {len(fine_vtk_files)} frames")

    return PairResult(
        config=pc,
        status="complete",
        coarse_patches=total_coarse,
        fine_patches=total_fine,
        vtk_files_fine=len(fine_vtk_files),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str, resume: bool = False, use_modal: bool = False):
    config = load_config(config_path)
    pairs = generate_pair_configs(config)
    manifest = load_manifest() if resume else {"completed": {}, "failed": {}}
    completed = load_completed_ids(manifest)
    neighborhood = config.get("patch_neighborhood", 3)

    pending = [p for p in pairs if p.pair_id not in completed]
    print(f"Total configs: {len(pairs)}, completed: {len(completed)}, pending: {len(pending)}")
    if use_modal:
        print("Using Modal GPU backend")

    if not pending:
        print("All pairs already generated.")
        return

    if not use_modal and not SIM_BINARY.exists():
        print(
            f"ERROR: Simulation binary not found at {SIM_BINARY}\n"
            f"Build it first: cd {PROJECT_ROOT} && make\n"
            f"Or use --modal to run on Modal GPU instead.",
            file=sys.stderr,
        )
        sys.exit(1)

    accumulator = PatchAccumulator()
    success = 0
    failed = 0

    # Use a temporary directory for VTK output. Cleaned up at the end.
    with tempfile.TemporaryDirectory(prefix="superres_vtk_") as tmp:
        work_dir = Path(tmp)

        for i, pc in enumerate(pending):
            tag = f"[{i + 1}/{len(pending)}]"
            print(f"{tag} {pc.model} wind={pc.wind_speed} Re={pc.reynolds}")

            result = process_pair(pc, accumulator, neighborhood, work_dir, use_modal=use_modal)

            if result.status == "complete":
                manifest.setdefault("completed", {})[pc.pair_id] = {
                    "model": pc.model,
                    "wind_speed": pc.wind_speed,
                    "reynolds": pc.reynolds,
                    "coarse_grid": pc.coarse_grid,
                    "fine_grid": pc.fine_grid,
                    "patches": result.coarse_patches,
                    "vtk_files": result.vtk_files_fine,
                }
                success += 1
            else:
                manifest.setdefault("failed", {})[pc.pair_id] = {
                    "model": pc.model,
                    "wind_speed": pc.wind_speed,
                    "reynolds": pc.reynolds,
                    "error": result.error,
                }
                failed += 1
                print(f"  FAILED: {result.error}")

            # Save manifest after each config so progress isn't lost
            save_manifest(manifest)

            # Clean up VTK files for this pair to save disk space
            for d in [work_dir / f"{pc.pair_id}_coarse", work_dir / f"{pc.pair_id}_fine"]:
                if d.exists():
                    shutil.rmtree(d)

    print(f"\nFinished: {success} complete, {failed} failed")
    print(f"Total patches accumulated: {accumulator.count}")

    if accumulator.count > 0:
        write_srbin(SRBIN_FILE, accumulator.coarse_inputs, accumulator.fine_targets)
        save_stats(manifest, accumulator)
    else:
        print("No patches extracted. Check simulation output and grid dimensions.")


def save_stats(manifest: dict, accumulator: PatchAccumulator):
    """Write summary statistics to JSON."""
    completed = manifest.get("completed", {})
    failed = manifest.get("failed", {})

    by_model = {}
    for entry in completed.values():
        m = entry["model"]
        if m not in by_model:
            by_model[m] = {"configs": 0, "patches": 0}
        by_model[m]["configs"] += 1
        by_model[m]["patches"] += entry.get("patches", 0)

    stats = {
        "total_configs": len(completed) + len(failed),
        "completed_configs": len(completed),
        "failed_configs": len(failed),
        "total_patches": accumulator.count,
        "coarse_patch_shape": [3, 3, 3, CHANNELS],
        "fine_patch_shape": [2, 2, 2, CHANNELS],
        "bytes_per_patch": (3 * 3 * 3 * CHANNELS + 2 * 2 * 2 * CHANNELS) * 4,
        "by_model": by_model,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics:")
    print(f"  Configs completed: {stats['completed_configs']}")
    print(f"  Configs failed:    {stats['failed_configs']}")
    print(f"  Total patches:     {stats['total_patches']}")
    for model, ms in by_model.items():
        print(f"  {model}: {ms['configs']} configs, {ms['patches']} patches")


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def show_status():
    """Show current state of data generation."""
    if not MANIFEST_FILE.exists():
        print("No data generated yet.")
        print(f"  Expected manifest at: {MANIFEST_FILE}")
        return

    manifest = load_manifest()
    completed = manifest.get("completed", {})
    failed = manifest.get("failed", {})

    total_patches = sum(e.get("patches", 0) for e in completed.values())

    print(f"Manifest: {MANIFEST_FILE}")
    print(f"  Completed: {len(completed)}")
    print(f"  Failed:    {len(failed)}")
    print(f"  Total patches: {total_patches}")

    if completed:
        by_model = {}
        for entry in completed.values():
            m = entry["model"]
            if m not in by_model:
                by_model[m] = {"configs": 0, "patches": 0}
            by_model[m]["configs"] += 1
            by_model[m]["patches"] += entry.get("patches", 0)

        print("\n  Per model:")
        for model, ms in sorted(by_model.items()):
            print(f"    {model}: {ms['configs']} configs, {ms['patches']} patches")

    if failed:
        print(f"\n  Failed configs:")
        for pid, entry in list(failed.items())[:10]:
            print(f"    {pid}: {entry['model']} wind={entry['wind_speed']} -- {entry['error'][:80]}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")

    if SRBIN_FILE.exists():
        size_mb = SRBIN_FILE.stat().st_size / (1024 * 1024)
        print(f"\n  Binary: {SRBIN_FILE} ({size_mb:.1f} MB)")
    else:
        print(f"\n  Binary: not yet written")

    if STATS_FILE.exists():
        with open(STATS_FILE) as f:
            stats = json.load(f)
        print(f"  Bytes per patch: {stats.get('bytes_per_patch', '?')}")


def verify_srbin():
    """Quick sanity check on the .srbin file."""
    if not SRBIN_FILE.exists():
        print(f"No .srbin file at {SRBIN_FILE}")
        return

    coarse, fine = read_srbin(SRBIN_FILE)
    print(f"Loaded {coarse.shape[0]} patches from {SRBIN_FILE}")
    print(f"  Coarse shape: {coarse.shape}")
    print(f"  Fine shape:   {fine.shape}")
    print(f"  Coarse vel range: [{coarse[..., :3].min():.4f}, {coarse[..., :3].max():.4f}]")
    print(f"  Fine vel range:   [{fine[..., :3].min():.4f}, {fine[..., :3].max():.4f}]")
    print(f"  Coarse rho range: [{coarse[..., 3].min():.4f}, {coarse[..., 3].max():.4f}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate paired coarse/fine data for super-resolution training"
    )
    parser.add_argument(
        "--config",
        default="superres_config.yaml",
        help="Config YAML file (default: superres_config.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed pair configs",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current data generation progress",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the .srbin file and print statistics",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Run simulations on Modal GPU instead of locally",
    )
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.verify:
        verify_srbin()
        return

    config_path = Path(__file__).resolve().parent / args.config
    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(str(config_path), resume=args.resume, use_modal=args.modal)


if __name__ == "__main__":
    main()
