"""
Modal GPU worker for fluid sim.
Deploy: modal deploy modal_worker.py
Test: modal run modal_worker.py --duration=5 --model=ahmed25
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import modal

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

REPO_URL = "https://github.com/MarcosAsh/3dFluidDynamicsInC.git"


@app.function(image=image, volumes={"/cache": build_cache}, timeout=1000)
def build_simulation() -> str:
    repo_dir = Path("/cache/source")
    source_dir = repo_dir / "simulation"
    build_dir = Path("/cache/build")
    executable = build_dir / "3d_fluid_simulation_car"

    if repo_dir.exists():
        print("Pulling latest...")
        subprocess.run(["git", "pull"], cwd=repo_dir, capture_output=True)
    else:
        print(f"Cloning {REPO_URL}")
        result = subprocess.run(
            ["git", "clone", "--depth=1", REPO_URL, str(repo_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise Exception(f"Clone failed: {result.stderr}")

    # Clean any stale cmake files from the source dir
    for f in ["CMakeCache.txt", "cmake_install.cmake", "Makefile"]:
        cache_file = source_dir / f
        if cache_file.exists():
            cache_file.unlink()
            print(f"Removed {f}")
    cmake_files_dir = source_dir / "CMakeFiles"
    if cmake_files_dir.exists():
        shutil.rmtree(cmake_files_dir)
        print("Removed CMakeFiles/")

    print("Running cmake...")
    build_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["cmake", str(source_dir), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"CMake failed: {result.stderr}")

    print("Running make...")
    result = subprocess.run(
        ["make", "-j4"],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    print(result.stdout[-2000:])
    if result.returncode != 0:
        raise Exception(f"Make failed: {result.stderr}")

    if not executable.exists():
        files = list(build_dir.iterdir())
        raise Exception(f"No executable. Got: {[f.name for f in files]}")

    build_cache.commit()
    print(f"Done: {executable}")
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
    import time

    work_dir = Path(tempfile.mkdtemp(prefix="render_"))
    frames_dir = work_dir / "frames"
    frames_dir.mkdir()

    xvfb = None

    try:
        executable = Path("/cache/build/3d_fluid_simulation_car")
        source_dir = Path("/cache/source/simulation")

        def ensure_built():
            """Make sure we have a fresh binary with the latest CLI flags."""
            if not executable.exists():
                print("Need to build first...")
                build_simulation.local()
            else:
                # Check that the binary understands --model; if not, rebuild.
                help_out = subprocess.run(
                    [str(executable), "--help"],
                    capture_output=True,
                    text=True,
                )
                help_text = (help_out.stdout or "") + (help_out.stderr or "")
                if "--model" not in help_text:
                    print(
                        "Binary missing --model flag; rebuilding to refresh build cache..."
                    )
                    build_simulation.local()

        ensure_built()
        if not executable.exists():
            return {"status": "error", "error": "Build failed"}

        help_check = subprocess.run(
            [str(executable), "--help"],
            capture_output=True,
            text=True,
        )
        help_text = (help_check.stdout or "") + (help_check.stderr or "")
        supports_model_flag = "--model" in help_text

        print("Starting Xvfb...")
        xvfb = subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1920x1080x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)

        # Select model path
        model_paths = {
            "car": "assets/3d-files/car-model.obj",
            "ahmed25": "assets/3d-files/ahmed_25deg_m.obj",
            "ahmed35": "assets/3d-files/ahmed_35deg_m.obj",
        }

        if model == "custom" and obj_data:
            import base64

            custom_obj = work_dir / "custom_model.obj"
            custom_obj.write_bytes(base64.b64decode(obj_data))
            model_path = str(custom_obj)
            print(f"Custom OBJ: {custom_obj.stat().st_size / 1024:.0f} KB")
        else:
            model_path = model_paths.get(model, model_paths["car"])

        print(
            f"Running sim: wind={wind_speed}, viz={viz_mode}, model={model}, {duration}s"
        )

        env = os.environ.copy()
        env["DISPLAY"] = ":99"

        cmd = [
            str(executable),
            f"--wind={wind_speed}",
            f"--viz={viz_mode}",
            f"--collision={collision_mode}",
            f"--duration={duration}",
            f"--output={frames_dir}",
        ]

        if supports_model_flag:
            cmd.append(f"--model={model_path}")
        else:
            print(
                "Warning: binary does not support --model; using default compiled model."
            )

        if reynolds > 0:
            cmd.append(f"--reynolds={reynolds}")

        result = subprocess.run(
            cmd,
            cwd=str(source_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=duration * 3 + 60,
        )

        print(f"stdout: {result.stdout[-1000:]}")
        if result.stderr:
            print(f"stderr: {result.stderr[-500:]}")

        # Extract Cd and Cl values from stdout.
        # The simulation skips the first ~3s of output (warmup), so all
        # values here should already be past the startup transient.
        cd_values = []
        cl_values = []
        for line in result.stdout.split("\n"):
            if "Cd=" in line:
                try:
                    cd_str = line.split("Cd=")[1].split()[0]
                    cd_values.append(float(cd_str))
                except:
                    pass
            if "Cl=" in line:
                try:
                    cl_str = line.split("Cl=")[1].split()[0]
                    cl_values.append(float(cl_str))
                except:
                    pass

        # Get the last stable Cd/Cl values (average of last few)
        cd_value = None
        if cd_values:
            cd_value = sum(cd_values[-5:]) / len(cd_values[-5:])

        cl_value = None
        if cl_values:
            cl_value = sum(cl_values[-5:]) / len(cl_values[-5:])

        if result.returncode != 0:
            return {
                "status": "error",
                "error": f"Crashed: {result.stderr[-300:]}",
            }

        frames = sorted(frames_dir.glob("frame_*.ppm"))
        print(f"Got {len(frames)} frames")

        if len(frames) == 0:
            return {"status": "error", "error": "No frames rendered"}

        print("Encoding video...")
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
            ffmpeg_cmd, capture_output=True, text=True, timeout=120
        )

        if not output_video.exists():
            return {
                "status": "error",
                "error": f"FFmpeg failed: {ffmpeg_result.stderr[-300:]}",
            }

        video_size = output_video.stat().st_size
        print(f"Video size: {video_size / 1024 / 1024:.1f} MB")

        video_url = upload_video(output_video, job_id)

        return {
            "status": "complete",
            "video_url": video_url,
            "model": model,
            "wind_speed": wind_speed,
            "cd_value": cd_value,
            "cl_value": cl_value,
            "cd_series": cd_values,
            "cl_series": cl_values,
        }

    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "Timed out"}
    except Exception as e:
        import traceback

        return {
            "status": "error",
            "error": f"{e}\n{traceback.format_exc()[-500:]}",
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
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    key = f"renders/{job_id}.mp4"

    print(f"Uploading to s3://{bucket}/{key}")
    s3.upload_file(
        str(video_path), bucket, key, ExtraArgs={"ContentType": "video/mp4"}
    )

    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    print(f"Uploaded: {url}")
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
        collision_mode=int(data.get("collision_mode", 1)),
        duration=int(data.get("duration", 10)),
        model=data.get("model", "car"),
        obj_data=data.get("obj_data"),
        reynolds=float(data.get("reynolds", 0)),
    )

    callback_url = data.get("callback_url")
    if callback_url:
        try:
            requests.post(
                callback_url, json={"job_id": job_id, **result}, timeout=10
            )
        except Exception as e:
            print(f"Callback failed: {e}")

    return result


@app.function(image=image, volumes={"/cache": build_cache}, timeout=300)
def force_rebuild() -> str:
    """Force a clean rebuild by clearing the cache."""
    source_dir = Path("/cache/source")
    build_dir = Path("/cache/build")

    if build_dir.exists():
        shutil.rmtree(build_dir)
        print("Cleared build directory")

    if source_dir.exists():
        shutil.rmtree(source_dir)
        print("Cleared source directory")

    build_cache.commit()

    # Now rebuild
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
        f"Job {job_id}: wind={wind}, viz={viz}, collision={collision}, model={model}, {duration}s"
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
            print(f"Video: base64, {len(url) // 1024}KB")
            import base64

            b64_data = url.split(",")[1]
            with open(f"render_{job_id}.mp4", "wb") as f:
                f.write(base64.b64decode(b64_data))
            print(f"Saved: render_{job_id}.mp4")
        else:
            print(f"URL: {url}")
    if result.get("cd_value"):
        print(f"Drag Coefficient (Cd): {result['cd_value']:.4f}")
    if result.get("cl_value"):
        print(f"Lift Coefficient (Cl): {result['cl_value']:.4f}")
    if result.get("error"):
        print(f"Error: {result['error']}")
