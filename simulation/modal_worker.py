"""
Modal GPU worker for fluid sim.
Deploy: modal deploy modal_worker.py
Test: modal run modal_worker.py --duration=5 --model=ahmed25
"""

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import modal

# Structured JSON logging
log = logging.getLogger("fluid-sim")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter(
        json.dumps(
            {
                "ts": "%(asctime)s",
                "level": "%(levelname)s",
                "msg": "%(message)s",
            }
        )
    )
)
log.addHandler(_handler)

app = modal.App("fluid-sim")

GRID = "128x96x96"
S3_BUCKET = "fluid-sim-renders"
S3_REGION = "eu-west-2"


def _s3_client():
    """Create a boto3 S3 client from env credentials."""
    import boto3

    return boto3.client(
        "s3",
        region_name=os.environ.get("S3_REGION", S3_REGION),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get(
            "AWS_SECRET_ACCESS_KEY"
        ),
    )

image = (
    modal.Image.debian_slim("3.11")
    .apt_install(
        "build-essential",
        "cmake",
        "pkg-config",
        "git",
        "libsdl2-dev",
        "libsdl2-ttf-dev",
        "libegl1-mesa-dev",
        "libegl-dev",
        "libgles2-mesa-dev",
        "libglvnd-dev",
        "xvfb",
        "ffmpeg",
    )
    .pip_install("requests", "fastapi[standard]", "boto3")
    .env(
        {
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "DISPLAY": ":99",
        }
    )
)

build_cache = modal.Volume.from_name(
    "fluid-sim-build-cache", create_if_missing=True
)

REPO_URL = (
    "https://github.com/MarcosAsh/Lattice_Fluid_Dynamics.git"
)


@app.function(
    image=image,
    volumes={"/cache": build_cache},
    timeout=1800,
)
def build_simulation() -> str:
    repo_dir = Path("/cache/source")
    source_dir = repo_dir / "simulation"
    build_dir = Path("/cache/build")
    executable = build_dir / "3d_fluid_simulation_car"

    if repo_dir.exists():
        log.info("pulling latest source")
        pull = subprocess.run(
            ["git", "pull"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        if pull.returncode != 0:
            log.warning(
                "git pull failed, re-cloning",
                extra={"stderr": pull.stderr[:300]},
            )
            shutil.rmtree(repo_dir)
            # Fall through to clone below

    if not repo_dir.exists():
        log.info("cloning repo", extra={"url": REPO_URL})
        result = subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                REPO_URL,
                str(repo_dir),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise Exception(f"Clone failed: {result.stderr}")

    for f in [
        "CMakeCache.txt",
        "cmake_install.cmake",
        "Makefile",
    ]:
        p = source_dir / f
        if p.exists():
            p.unlink()
    cmake_dir = source_dir / "CMakeFiles"
    if cmake_dir.exists():
        shutil.rmtree(cmake_dir)

    log.info("running cmake")
    build_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "cmake",
            str(source_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    log.info("cmake output: " + result.stdout[-500:])
    if result.returncode != 0:
        raise Exception(f"CMake failed: {result.stderr}")

    log.info("running make")
    result = subprocess.run(
        ["make", "-j4"],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Exception(f"Make failed: {result.stderr}")

    if not executable.exists():
        files = list(build_dir.iterdir())
        raise Exception(
            f"No executable. Got: {[f.name for f in files]}"
        )

    build_cache.commit()
    log.info("build complete", extra={"path": str(executable)})
    return str(executable)


def _get_code_version() -> str:
    """Get the git short hash from the cached source."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd="/cache/source",
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _cache_key(
    model: str,
    wind_speed: float,
    viz_mode: int,
    collision_mode: int,
    reynolds: float,
    duration: int,
    grid: str,
) -> str:
    """Hash render params + code version to check for cached results."""
    import hashlib

    version = _get_code_version()
    key = (
        f"{model}_{wind_speed}_{viz_mode}_"
        f"{collision_mode}_{reynolds}_{duration}_{grid}_{version}"
    )
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _check_cache(cache_id: str) -> dict | None:
    """Check S3 for a cached render result."""
    try:
        bucket = os.environ.get("S3_BUCKET", S3_BUCKET)
        s3 = _s3_client()
        meta_key = f"cache/{cache_id}.json"
        resp = s3.get_object(Bucket=bucket, Key=meta_key)
        return json.loads(resp["Body"].read())
    except Exception:
        return None


def _save_cache(cache_id: str, result: dict):
    """Save render result metadata to S3 for caching."""
    try:
        bucket = os.environ.get("S3_BUCKET", S3_BUCKET)
        s3 = _s3_client()
        meta_key = f"cache/{cache_id}.json"
        s3.put_object(
            Bucket=bucket,
            Key=meta_key,
            Body=json.dumps(result),
            ContentType="application/json",
        )
    except Exception as e:
        log.warning(
            "cache save failed", extra={"error": str(e)}
        )


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": build_cache},
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=1800,
)
def render_simulation(
    job_id: str,
    wind_speed: float = 1.0,
    viz_mode: int = 1,
    collision_mode: int = 1,
    duration: int = 10,
    model: str = "car",
    obj_data: str | None = None,
    reynolds: float = 0,
    superres: bool = False,
    grid: str | None = None,
) -> dict:
    timings = {}
    t0 = time.monotonic()

    work_dir = Path(tempfile.mkdtemp(prefix="render_"))
    frames_dir = work_dir / "frames"
    frames_dir.mkdir()
    xvfb = None

    ctx = {
        "job_id": job_id,
        "model": model,
        "wind_speed": wind_speed,
        "reynolds": reynolds,
        "duration": duration,
    }

    # Check cache for non-custom models
    if model != "custom":
        cache_id = _cache_key(
            model, wind_speed, viz_mode,
            collision_mode, reynolds,
            duration, GRID,
        )
        cached = _check_cache(cache_id)
        if cached:
            log.info("cache hit", extra={**ctx, "cache_id": cache_id})
            return cached

    log.info("render start", extra=ctx)

    try:
        executable = Path(
            "/cache/build/3d_fluid_simulation_car"
        )
        source_dir = Path("/cache/source/simulation")

        # Build check
        t_build = time.monotonic()
        if not executable.exists():
            log.info("binary missing, building", extra=ctx)
            build_simulation.local()
        else:
            help_out = subprocess.run(
                [str(executable), "--help"],
                capture_output=True,
                text=True,
            )
            help_text = (help_out.stdout or "") + (
                help_out.stderr or ""
            )
            if "--model" not in help_text:
                log.info("binary outdated, rebuilding", extra=ctx)
                build_simulation.local()
        timings["build_check"] = time.monotonic() - t_build

        if not executable.exists():
            log.error("build failed", extra=ctx)
            return {
                "status": "error",
                "error": "Build failed",
                "error_type": "build",
            }

        # Xvfb for software GL rendering
        t_xvfb = time.monotonic()
        log.info("starting xvfb", extra=ctx)
        xvfb = subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1920x1080x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        timings["xvfb_start"] = time.monotonic() - t_xvfb

        # Model selection
        model_paths = {
            "car": "assets/3d-files/car-model.obj",
            "ahmed25": "assets/3d-files/ahmed_25deg_m.obj",
            "ahmed35": "assets/3d-files/ahmed_35deg_m.obj",
        }

        if model == "custom" and obj_data:
            import base64

            custom_obj = work_dir / "custom_model.obj"
            custom_obj.write_bytes(
                base64.b64decode(obj_data)
            )
            model_path = str(custom_obj)
            log.info(
                "custom obj loaded",
                extra={
                    **ctx,
                    "size_kb": custom_obj.stat().st_size // 1024,
                },
            )
        else:
            model_path = model_paths.get(
                model, model_paths["car"]
            )

        env = os.environ.copy()
        env["DISPLAY"] = ":99"

        sim_grid = grid or GRID
        cmd = [
            "stdbuf", "-oL",
            str(executable),
            f"--wind={wind_speed}",
            f"--viz={viz_mode}",
            f"--collision={collision_mode}",
            f"--duration={duration}",
            f"--output={frames_dir}",
            f"--grid={sim_grid}",
        ]
        cmd.append(f"--model={model_path}")
        if reynolds > 0:
            cmd.append(f"--reynolds={reynolds}")
        if superres:
            cmd.append("--superres")

        log.info("simulation starting", extra=ctx)
        t_sim = time.monotonic()
        sim_timeout = duration * 120 + 300
        proc = subprocess.Popen(
            cmd,
            cwd=str(source_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Read stdout line-by-line for live Cd/Cl streaming
        stdout_lines: list[str] = []
        _live_cd: list[float] = []
        _live_cl: list[float] = []
        _live_update_interval = 3  # save progress every N new Cd samples
        _last_live_count = 0

        try:
            deadline = time.monotonic() + sim_timeout
            while True:
                if time.monotonic() > deadline:
                    proc.kill()
                    proc.wait()
                    stdout = "\n".join(stdout_lines)
                    stderr = proc.stderr.read() if proc.stderr else ""
                    log.error(
                        "render timed out",
                        extra={
                            **ctx,
                            "stdout_tail": (stdout or "")[-2000:],
                            "stderr_tail": (stderr or "")[-500:],
                        },
                    )
                    return {
                        "status": "error",
                        "error": (
                            f"Timed out\nstdout:\n"
                            f"{(stdout or '')[-2000:]}"
                        ),
                        "error_type": "timeout",
                    }

                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    stdout_lines.append(line.rstrip("\n"))

                    # Parse Cd/Cl as they arrive
                    if "Cd=" in line:
                        try:
                            val = float(line.split("Cd=")[1].split()[0])
                            if math.isfinite(val):
                                _live_cd.append(val)
                        except (IndexError, ValueError):
                            pass
                    if "Cl=" in line:
                        try:
                            val = float(line.split("Cl=")[1].split()[0])
                            if math.isfinite(val):
                                _live_cl.append(val)
                        except (IndexError, ValueError):
                            pass

                    # Periodically save progress for live polling
                    if (job_id and len(_live_cd) > 0 and
                            len(_live_cd) - _last_live_count >= _live_update_interval):
                        _last_live_count = len(_live_cd)
                        _save_job_result(job_id, {
                            "status": "rendering",
                            "cd_series": _live_cd[:],
                            "cl_series": _live_cl[:],
                        })

            stderr = proc.stderr.read() if proc.stderr else ""
            stdout = "\n".join(stdout_lines)
        except Exception:
            proc.kill()
            proc.wait()
            stdout = "\n".join(stdout_lines)
            stderr = proc.stderr.read() if proc.stderr else ""

        sim_rc = proc.returncode
        timings["simulation"] = time.monotonic() - t_sim

        # Parse Cd/Cl
        cd_values = []
        cl_values = []
        effective_re = None
        grid_size = None
        char_length = None
        strouhal = None
        sample_interval = None
        t_star = None
        flow_throughs = None
        cfl = None
        cd_pressure_values = []
        cd_friction_values = []

        def _parse_float(line: str, prefix: str, suffix: str | None = None) -> float | None:
            try:
                raw = line.split(prefix)[1]
                if suffix:
                    raw = raw.split(suffix)[0]
                else:
                    raw = raw.split()[0]
                val = float(raw.strip())
                if not math.isfinite(val):
                    log.warning("non-finite value", extra={"prefix": prefix, "raw": raw})
                    return None
                return val
            except (IndexError, ValueError) as e:
                log.warning("parse failed", extra={"prefix": prefix, "line": line.strip(), "error": str(e)})
                return None

        for line in stdout.split("\n"):
            if "Cd=" in line:
                val = _parse_float(line, "Cd=")
                if val is not None:
                    cd_values.append(val)
            if "Cl=" in line:
                val = _parse_float(line, "Cl=")
                if val is not None:
                    cl_values.append(val)
            if "Effective Re" in line:
                val = _parse_float(line, "Re =")
                if val is not None:
                    effective_re = val
            if "Char length:" in line:
                val = _parse_float(line, "Char length:", "lattice")
                if val is not None:
                    char_length = val
            if "LBM Grid:" in line:
                try:
                    grid_size = line.split("Grid:")[1].split("(")[0].strip()
                except (IndexError, ValueError) as e:
                    log.warning("grid parse failed", extra={"line": line.strip(), "error": str(e)})
            if "St=" in line and "f=" in line:
                val = _parse_float(line, "St=")
                if val is not None:
                    strouhal = val
            if "Sample interval:" in line:
                val = _parse_float(line, "Sample interval:", "lattice")
                if val is not None:
                    sample_interval = int(val)
            if "t*=" in line:
                val = _parse_float(line, "t*=")
                if val is not None:
                    t_star = val
            if "flow-throughs=" in line:
                val = _parse_float(line, "flow-throughs=")
                if val is not None:
                    flow_throughs = val
            if "CFL:" in line:
                val = _parse_float(line, "CFL:")
                if val is not None:
                    cfl = val
            if "Cd_pressure=" in line:
                val = _parse_float(line, "Cd_pressure=")
                if val is not None:
                    cd_pressure_values.append(val)
            if "Cd_friction=" in line:
                val = _parse_float(line, "Cd_friction=")
                if val is not None:
                    cd_friction_values.append(val)

        cd_value = None
        if cd_values:
            cd_value = sum(cd_values[-5:]) / len(
                cd_values[-5:]
            )
        cl_value = None
        if cl_values:
            cl_value = sum(cl_values[-5:]) / len(
                cl_values[-5:]
            )

        if sim_rc != 0:
            log.error(
                "simulation crashed",
                extra={
                    **ctx,
                    "rc": sim_rc,
                    "stdout_tail": stdout[-2000:],
                    "stderr": stderr[-500:],
                },
            )
            return {
                "status": "error",
                "error": f"Crashed (rc={sim_rc}): {stdout[-2000:]}\n{stderr[-300:]}",
                "error_type": "simulation",
            }

        frames = sorted(frames_dir.glob("frame_*.ppm"))
        log.info(
            "simulation done",
            extra={
                **ctx,
                "frames": len(frames),
                "cd_value": cd_value,
                "cl_value": cl_value,
                "effective_re": effective_re,
            },
        )

        if len(frames) == 0:
            log.error("no frames produced", extra=ctx)
            return {
                "status": "error",
                "error": "No frames rendered",
                "error_type": "simulation",
            }

        # Encode video
        t_enc = time.monotonic()
        output_video = work_dir / "output.mp4"
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "60",
            "-i",
            str(frames_dir / "frame_%05d.ppm"),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            str(output_video),
        ]
        ffmpeg_result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        timings["encode"] = time.monotonic() - t_enc

        if not output_video.exists():
            log.error(
                "ffmpeg failed",
                extra={
                    **ctx,
                    "stderr": ffmpeg_result.stderr[-300:],
                },
            )
            return {
                "status": "error",
                "error": (
                    "FFmpeg failed: "
                    + ffmpeg_result.stderr[-300:]
                ),
                "error_type": "encode",
            }

        video_mb = output_video.stat().st_size / 1024 / 1024

        # Upload
        t_upload = time.monotonic()
        video_url = upload_video(output_video, job_id)
        timings["upload"] = time.monotonic() - t_upload

        timings["total"] = time.monotonic() - t0
        log.info(
            "render complete",
            extra={
                **ctx,
                "cd_value": cd_value,
                "effective_re": effective_re,
                "video_mb": round(video_mb, 1),
                "timings": timings,
            },
        )

        result = {
            "status": "complete",
            "video_url": video_url,
            "model": model,
            "wind_speed": wind_speed,
            "cd_value": cd_value,
            "cl_value": cl_value,
            "cd_series": cd_values,
            "cl_series": cl_values,
            "effective_re": effective_re,
            "grid_size": grid_size,
            "char_length": char_length,
            "strouhal": strouhal,
            "sample_interval": sample_interval,
            "t_star": t_star,
            "flow_throughs": flow_throughs,
            "cfl": cfl,
            "cd_pressure_series": cd_pressure_values,
            "cd_friction_series": cd_friction_values,
            "timings": {
                k: round(v, 2)
                for k, v in timings.items()
            },
        }

        # Cache for future identical requests
        if model != "custom":
            cid = _cache_key(
                model, wind_speed, viz_mode,
                collision_mode, reynolds,
                duration, GRID,
            )
            _save_cache(cid, result)

        return result

    except subprocess.TimeoutExpired:
        # Handled inside the Popen block above; this catches
        # any other TimeoutExpired (shouldn't happen).
        return {
            "status": "error",
            "error": "Timed out (outer)",
            "error_type": "timeout",
        }
    except Exception as e:
        import traceback

        log.error(
            "render exception",
            extra={**ctx, "error": str(e)},
        )
        return {
            "status": "error",
            "error": f"{e}\n{traceback.format_exc()[-500:]}",
            "error_type": "exception",
        }
    finally:
        if xvfb:
            xvfb.terminate()
        shutil.rmtree(work_dir, ignore_errors=True)


def upload_video(video_path: Path, job_id: str) -> str:
    bucket = os.environ.get("S3_BUCKET", S3_BUCKET)
    region = os.environ.get("S3_REGION", S3_REGION)
    s3 = _s3_client()

    key = f"renders/{job_id}.mp4"
    log.info(
        "uploading to s3",
        extra={"bucket": bucket, "key": key},
    )
    s3.upload_file(
        str(video_path),
        bucket,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )

    url = (
        f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    )
    return url


def _save_job_result(job_id: str, result: dict):
    """Write render result to S3 so the status endpoint can find it."""
    try:
        bucket = os.environ.get("S3_BUCKET", S3_BUCKET)
        s3 = _s3_client()
        s3.put_object(
            Bucket=bucket,
            Key=f"jobs/{job_id}.json",
            Body=json.dumps(result),
            ContentType="application/json",
        )
    except Exception as e:
        log.warning("job result save failed", extra={"error": str(e)})


def _get_job_result(job_id: str) -> dict | None:
    """Check S3 for a completed job result."""
    try:
        bucket = os.environ.get("S3_BUCKET", S3_BUCKET)
        s3 = _s3_client()
        resp = s3.get_object(Bucket=bucket, Key=f"jobs/{job_id}.json")
        return json.loads(resp["Body"].read())
    except Exception:
        return None


@app.function(image=image, timeout=30)
@modal.fastapi_endpoint(method="GET")
def health() -> dict:
    """Lightweight health check -- no GPU needed."""
    return {
        "status": "ok",
        "grid": GRID,
        "gpu": "A10G",
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": build_cache},
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=1800,
)
def _run_render(data: dict):
    """Background render: runs simulation, saves result to S3."""
    import uuid

    job_id = data.get("job_id", str(uuid.uuid4()))

    # Mark job as started
    _save_job_result(job_id, {"status": "rendering"})

    result = render_simulation.local(
        job_id=job_id,
        wind_speed=float(data.get("wind_speed", 1.0)),
        viz_mode=int(data.get("viz_mode", 1)),
        collision_mode=int(
            data.get("collision_mode", 2)
        ),
        duration=int(data.get("duration", 10)),
        model=data.get("model", "car"),
        obj_data=data.get("obj_data"),
        reynolds=float(data.get("reynolds", 0)),
        superres=bool(data.get("superres", False)),
        grid=data.get("grid"),
    )

    _save_job_result(job_id, result)
    return result


@app.function(image=image, timeout=30, secrets=[modal.Secret.from_name("aws-secret")])
@modal.fastapi_endpoint(method="POST")
def render_endpoint(data: dict) -> dict:
    """Accept a render request and spawn it in the background."""
    import uuid

    job_id = data.get("job_id", f"job_{uuid.uuid4().hex[:12]}")
    data["job_id"] = job_id

    # Spawn the render asynchronously -- returns immediately
    _run_render.spawn(data)

    return {"status": "accepted", "job_id": job_id}


@app.function(image=image, timeout=30, secrets=[modal.Secret.from_name("aws-secret")])
@modal.fastapi_endpoint(method="GET")
def job_status(job_id: str) -> dict:
    """Poll for render job completion."""
    if not job_id:
        return {"status": "error", "error": "Missing job_id"}

    result = _get_job_result(job_id)
    if result is None:
        return {"status": "rendering"}

    return result


@app.function(
    image=image,
    volumes={"/cache": build_cache},
    timeout=300,
)
def force_rebuild() -> str:
    """Force a clean rebuild by clearing the cache."""
    source_dir = Path("/cache/source")
    build_dir = Path("/cache/build")

    if build_dir.exists():
        shutil.rmtree(build_dir)
    if source_dir.exists():
        shutil.rmtree(source_dir)

    build_cache.commit()
    return build_simulation.local()


@app.function(
    image=image.pip_install("numpy"),
    gpu="A10G",
    volumes={"/cache": build_cache},
    timeout=1800,
)
def generate_vtk_field(
    model: str = "ahmed25",
    wind_speed: float = 1.0,
    reynolds: float = 0,
    grid: str = "128x96x96",
    duration: int = 15,
    vtk_interval: int = 50,
) -> dict:
    """Run simulation with VTK output and return velocity fields as bytes.

    Used by the super-resolution data generation pipeline to collect
    paired coarse/fine velocity fields without needing a local binary.
    """
    import base64
    import numpy as np

    executable = Path("/cache/build/3d_fluid_simulation_car")
    source_dir = Path("/cache/source/simulation")

    if not executable.exists():
        log.info("binary missing, building")
        build_simulation.local()

    if not executable.exists():
        return {"status": "error", "error": "Build failed"}

    vtk_dir = Path(tempfile.mkdtemp(prefix="vtk_"))

    model_paths = {
        "car": "assets/3d-files/car-model.obj",
        "ahmed25": "assets/3d-files/ahmed_25deg_m.obj",
        "ahmed35": "assets/3d-files/ahmed_35deg_m.obj",
    }
    model_path = model_paths.get(model, model_paths["car"])

    # Start Xvfb
    xvfb = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1920x1080x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    env = os.environ.copy()
    env["DISPLAY"] = ":99"

    frames_dir = Path(tempfile.mkdtemp(prefix="frames_"))
    cmd = [
        "stdbuf", "-oL",
        str(executable),
        f"--wind={wind_speed}",
        f"--duration={duration}",
        f"--grid={grid}",
        f"--model={model_path}",
        f"--output={frames_dir}",
        f"--vtk-output={vtk_dir}",
        f"--vtk-interval={vtk_interval}",
        "--viz=1", "--collision=1",
    ]
    if reynolds > 0:
        cmd.append(f"--reynolds={reynolds}")

    try:
        result = subprocess.run(
            cmd, cwd=str(source_dir), env=env,
            capture_output=True, text=True,
            timeout=duration * 120 + 300,
        )

        vtk_files = sorted(vtk_dir.glob("field_*.vti"))
        log.info(f"vtk generation done: {len(vtk_files)} files, rc={result.returncode}")

        if not vtk_files:
            return {
                "status": "error",
                "error": f"No VTK files (rc={result.returncode})",
                "stdout_tail": (result.stdout or "")[-1000:],
            }

        # Read each VTI and return the raw binary data
        fields = []
        for vti in vtk_files:
            fields.append({
                "filename": vti.name,
                "data": base64.b64encode(vti.read_bytes()).decode(),
            })

        return {
            "status": "complete",
            "grid": grid,
            "model": model,
            "num_fields": len(fields),
            "fields": fields,
        }

    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "Timed out"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        xvfb.terminate()
        shutil.rmtree(frames_dir, ignore_errors=True)
        shutil.rmtree(vtk_dir, ignore_errors=True)


@app.function(image=image, timeout=30, secrets=[modal.Secret.from_name("aws-secret")])
@modal.fastapi_endpoint(method="POST")
def optimize_shape(data: dict) -> dict:
    """Run shape optimization: deform mesh via FFD, predict Cd with surrogate."""
    import uuid

    job_id = data.get("job_id", f"opt_{uuid.uuid4().hex[:8]}")
    base_model = data.get("base_model", "ahmed25")
    wind_speed = float(data.get("wind_speed", 1.0))
    reynolds = float(data.get("reynolds", 10000))
    ffd_displacements = data.get("ffd_displacements")

    if ffd_displacements is None:
        return {
            "status": "error",
            "error": "ffd_displacements required",
        }

    return {
        "status": "accepted",
        "job_id": job_id,
        "base_model": base_model,
        "wind_speed": wind_speed,
        "reynolds": reynolds,
        "num_params": len(ffd_displacements),
    }


@app.local_entrypoint()
def main(
    wind: float = 1.0,
    viz: int = 1,
    collision: int = 1,
    duration: int = 5,
    model: str = "car",
    build_only: bool = False,
    rebuild: bool = False,
):
    import uuid

    if rebuild:
        print("Force rebuilding...")
        path = force_rebuild.remote()
        print(f"Rebuilt: {path}")
        return

    if build_only:
        print("Building...")
        path = build_simulation.remote()
        print(f"Built: {path}")
        return

    job_id = str(uuid.uuid4())[:8]
    print(
        f"Job {job_id}: wind={wind}, viz={viz}, "
        f"collision={collision}, "
        f"model={model}, {duration}s"
    )

    result = render_simulation.remote(
        job_id=job_id,
        wind_speed=wind,
        viz_mode=viz,
        collision_mode=collision,
        duration=duration,
        model=model,
    )

    print(f"Status: {result['status']}")
    if result.get("video_url"):
        url = result["video_url"]
        if url.startswith("data:"):
            import base64

            b64_data = url.split(",")[1]
            with open(f"render_{job_id}.mp4", "wb") as f:
                f.write(base64.b64decode(b64_data))
            print(f"Saved: render_{job_id}.mp4")
        else:
            print(f"URL: {url}")
    if result.get("cd_value"):
        print(f"Cd: {result['cd_value']:.4f}")
    if result.get("cl_value"):
        print(f"Cl: {result['cl_value']:.4f}")
    if result.get("timings"):
        print(f"Timings: {result['timings']}")
    if result.get("error"):
        print(f"Error: {result['error']}")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/cache": build_cache},
    timeout=7200,
)
def run_tests(grid: str = "256x128x128", duration: int = 120):
    """Run LBM test suite + Ahmed body at larger grid on GPU.

    Usage: modal run modal_worker.py::run_tests --grid 256x128x128
    """
    import subprocess
    from pathlib import Path

    # Force clean rebuild to pick up latest changes
    import shutil
    build_dir = Path("/cache/build")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    source_dir = Path("/cache/source")
    if source_dir.exists():
        shutil.rmtree(source_dir)
    build_simulation.local()

    source_dir = Path("/cache/source/simulation")
    build_dir = Path("/cache/build")
    test_bin = build_dir / "test_lbm"
    sim_bin = build_dir / "3d_fluid_simulation_car"

    # Build test binary if missing
    if not test_bin.exists():
        log.info("building test binary")
        result = subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "test_lbm", "-j4"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Test build failed: {result.stderr}")
            return

    # Run unit tests -- unset DISPLAY so EGL uses GPU directly
    # instead of falling back to Xvfb + Mesa software GL.
    egl_env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    print("=" * 60)
    print("UNIT TESTS")
    print("=" * 60)
    result = subprocess.run(
        [str(test_bin)],
        cwd=str(source_dir),
        env=egl_env,
        capture_output=True, text=True,
        timeout=600,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Run Ahmed body simulation at requested grid size
    print("=" * 60)
    print(f"AHMED 25deg @ {grid} for {duration}s")
    print("=" * 60)

    # Start Xvfb
    xvfb = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1920x1080x24"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    import time
    time.sleep(2)

    # Unset DISPLAY so the sim uses EGL (direct GPU) instead of
    # Xvfb + Mesa software GL which has a 128MB SSBO limit.
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    cmd = [
        str(sim_bin),
        "--angle=25",
        f"--grid={grid}",
        f"--duration={duration}",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(source_dir),
        env=env,
        capture_output=True, text=True,
        timeout=duration * 60 + 300,
    )

    # Print key lines
    for line in result.stdout.splitlines():
        if any(kw in line for kw in [
            "Reynolds", "tau:", "Cd calc", "Projected", "Blockage",
            "Effective", "Drag Force", "avg=", "Cd_pressure",
            "WARNING", "capped", "Ground", "charLength",
        ]):
            print(line)

    xvfb.kill()
    print("\nDone.")
