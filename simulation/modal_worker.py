"""
Modal GPU worker for fluid sim.
Deploy: modal deploy modal_worker.py
Test: modal run modal_worker.py --duration=5 --model=ahmed25
"""

import json
import logging
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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("cmake")
    .apt_install(
        "build-essential",
        "pkg-config",
        "git",
        "libgl1-mesa-dev",
        "libglew-dev",
        "libglu1-mesa-dev",
        "libsdl2-dev",
        "libsdl2-ttf-dev",
        "xvfb",
        "x11-utils",
        "ffmpeg",
        "mesa-utils",
        "libosmesa6-dev",
    )
    .pip_install("requests", "fastapi[standard]", "boto3")
    .env(
        {
            "DISPLAY": ":99",
            "MESA_GL_VERSION_OVERRIDE": "4.3",
            "LIBGL_ALWAYS_SOFTWARE": "1",
        }
    )
)

build_cache = modal.Volume.from_name(
    "fluid-sim-build-cache", create_if_missing=True
)

REPO_URL = (
    "https://github.com/MarcosAsh/3dFluidDynamicsInC.git"
)


@app.function(
    image=image,
    volumes={"/cache": build_cache},
    timeout=1000,
)
def build_simulation() -> str:
    repo_dir = Path("/cache/source")
    source_dir = repo_dir / "simulation"
    build_dir = Path("/cache/build")
    executable = build_dir / "3d_fluid_simulation_car"

    if repo_dir.exists():
        log.info("pulling latest source")
        subprocess.run(
            ["git", "pull"],
            cwd=repo_dir,
            capture_output=True,
        )
    else:
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


@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": build_cache},
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=1000,
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

        help_check = subprocess.run(
            [str(executable), "--help"],
            capture_output=True,
            text=True,
        )
        help_text = (help_check.stdout or "") + (
            help_check.stderr or ""
        )
        supports_model = "--model" in help_text

        # Xvfb
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

        # Run simulation
        env = os.environ.copy()
        env["DISPLAY"] = ":99"

        cmd = [
            str(executable),
            f"--wind={wind_speed}",
            f"--viz={viz_mode}",
            f"--collision={collision_mode}",
            f"--duration={duration}",
            f"--output={frames_dir}",
            "--grid=64x32x32",
        ]
        if supports_model:
            cmd.append(f"--model={model_path}")
        if reynolds > 0:
            cmd.append(f"--reynolds={reynolds}")

        log.info("simulation starting", extra=ctx)
        t_sim = time.monotonic()
        result = subprocess.run(
            cmd,
            cwd=str(source_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=duration * 3 + 60,
        )
        timings["simulation"] = time.monotonic() - t_sim

        # Parse Cd/Cl
        cd_values = []
        cl_values = []
        effective_re = None
        grid_size = None

        for line in result.stdout.split("\n"):
            if "Cd=" in line:
                try:
                    s = line.split("Cd=")[1].split()[0]
                    cd_values.append(float(s))
                except Exception:
                    pass
            if "Cl=" in line:
                try:
                    s = line.split("Cl=")[1].split()[0]
                    cl_values.append(float(s))
                except Exception:
                    pass
            if "Effective Re" in line:
                try:
                    s = line.split("Re =")[1].split()[0]
                    effective_re = float(s)
                except Exception:
                    pass
            if "LBM Grid:" in line:
                try:
                    s = line.split("Grid:")[1].split("(")[0]
                    grid_size = s.strip()
                except Exception:
                    pass

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

        if result.returncode != 0:
            log.error(
                "simulation crashed",
                extra={
                    **ctx,
                    "stderr": result.stderr[-300:],
                },
            )
            return {
                "status": "error",
                "error": f"Crashed: {result.stderr[-300:]}",
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

        return {
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
            "timings": {
                k: round(v, 2) for k, v in timings.items()
            },
        }

    except subprocess.TimeoutExpired:
        log.error("render timed out", extra=ctx)
        return {
            "status": "error",
            "error": "Timed out",
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
    import boto3

    bucket = os.environ.get("S3_BUCKET", "fluid-sim-renders")
    region = os.environ.get("S3_REGION", "eu-west-2")

    s3 = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=os.environ.get(
            "AWS_ACCESS_KEY_ID"
        ),
        aws_secret_access_key=os.environ.get(
            "AWS_SECRET_ACCESS_KEY"
        ),
    )

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


@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": build_cache},
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=1000,
)
@modal.fastapi_endpoint(method="POST")
def render_endpoint(data: dict) -> dict:
    import uuid

    import requests

    job_id = data.get("job_id", str(uuid.uuid4()))

    result = render_simulation.local(
        job_id=job_id,
        wind_speed=float(data.get("wind_speed", 1.0)),
        viz_mode=int(data.get("viz_mode", 1)),
        collision_mode=int(
            data.get("collision_mode", 1)
        ),
        duration=int(data.get("duration", 10)),
        model=data.get("model", "car"),
        obj_data=data.get("obj_data"),
        reynolds=float(data.get("reynolds", 0)),
    )

    callback_url = data.get("callback_url")
    if callback_url:
        try:
            requests.post(
                callback_url,
                json={"job_id": job_id, **result},
                timeout=10,
            )
        except Exception as e:
            log.warning(
                "callback failed",
                extra={"job_id": job_id, "error": str(e)},
            )

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
