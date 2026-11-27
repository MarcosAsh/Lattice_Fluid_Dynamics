# Building

## Requirements

- C11 compiler (GCC, Clang, or MSVC)
- CMake 3.16+
- OpenGL 4.3+ capable GPU and drivers
- SDL2
- EGL (optional, for headless GPU rendering)

The GL loader (glad) is vendored in `simulation/lib/glad`.

## Linux

### Arch

```bash
sudo pacman -S base-devel cmake sdl2-compat mesa
```

### Ubuntu / Debian

```bash
sudo apt install build-essential cmake pkg-config \
    libsdl2-dev \
    libgl1-mesa-dev libegl1-mesa-dev
```

### Fedora

```bash
sudo dnf install cmake gcc SDL2-devel \
    mesa-libGL-devel mesa-libEGL-devel
```

### Build

```bash
cd simulation
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run

```bash
cd simulation
./build/3d_fluid_simulation_car --wind=1.0 --duration=10
```

On Wayland, you might need:

```bash
SDL_VIDEODRIVER=x11 ./build/3d_fluid_simulation_car
```

## macOS

Not supported natively. macOS stops at OpenGL 4.1 and the solver needs
4.3 compute shaders. Use a Linux VM or container with a GPU, or the
website with a Modal backend.

## Windows

Not covered by CI, so treat these steps as a starting point.

Install Visual Studio 2022 with C++ workload, then use vcpkg:

```powershell
vcpkg install sdl2
cd simulation
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

Or use MSYS2/MinGW:

```bash
pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-SDL2
cd simulation
mkdir -p build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
mingw32-make -j4
```

## Website

```bash
cd website
npm install
npm run dev
```

Set `MODAL_RENDER_ENDPOINT` in `.env.local` to connect to the
Modal backend. Without it, the site runs in demo mode.

## Tests

```bash
cmake -B build -S simulation -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure        # everything
ctest --test-dir build --output-on-failure -L cpu # no GL context needed
```

The `gl` tests need an OpenGL 4.3 context. Without a GPU, run them
under Mesa:

```bash
MESA_GL_VERSION_OVERRIDE=4.3 LIBGL_ALWAYS_SOFTWARE=1 USE_EGL=0 \
    xvfb-run -a ctest --test-dir build --output-on-failure
```

`simulation/CMakePresets.json` has `release`, `debug`, `asan` and `ci`
presets. Run them from `simulation/`, for example
`cmake --preset asan && cmake --build --preset asan`.

Build options:

| Option | Default | Effect |
|---|---|---|
| `LATTICE_WARNINGS` | `ON` | Compiler warnings |
| `LATTICE_WERROR` | `OFF` | Warnings fail the build |
| `LATTICE_SANITIZE` | empty | Sanitizers, e.g. `address,undefined` |

Debug builds print OpenGL debug messages to stderr. Set
`LATTICE_GL_DEBUG=1` or `0` to force them on or off.

## Python tools

The scripts under `ml/` are a [uv](https://docs.astral.sh/uv/) project.

```bash
uv sync --extra torch-cpu --extra anyobj   # or --extra torch for CUDA
uv run pytest
uv run python ml/cli.py                    # list the subcommands
```

## CLI options

Run `./build/3d_fluid_simulation_car --help` for the full list. The most
used ones:

```
--wind=SPEED         Wind speed 0-5 (default: 1.0)
--viz=MODE           Visualization mode 0-9 (default: 1)
--collision=MODE     0=off, 1=AABB, 2=mesh, 3=voxel (default: 2)
--duration=SECS      Headless render duration (0=interactive)
--output=PATH        Output directory for frames
--model=PATH         Path to OBJ model file
--reynolds=RE        Target Reynolds number (0=derive from wind speed)
--grid=XxYxZ         Grid size (default: 128x64x64)
--manifest=PATH      Run manifest JSON
--converge=TOL       Stop once the windowed mean Cd has settled
--converge-window=N  Cd samples per window (default: 20)
--max-duration=SECS  Frame cap for --converge
--version            Print version and exit
```

Invalid numbers and unknown options exit with status 2.

### Run manifest

A run with `--output` writes `manifest.json` next to the frames
(`--manifest=PATH` picks another place). It records the git commit, every
resolved parameter, the lattice setup (grid, viscosity, tau, Mach,
Reynolds), the GL renderer, the runtime and the final drag results, so a
Cd value can be traced back to the exact run that produced it.

### Stopping on convergence

```bash
./build/3d_fluid_simulation_car --model=assets/3d-files/ahmed_25deg_m.obj \
    --converge=0.01 --converge-window=20 --max-duration=60 \
    --manifest=run.json
```

The run ends when the mean Cd of the last two windows differs by less
than `TOL` twice in a row, or at `--max-duration`. The manifest reports
whether it converged, the mean and its standard error (batch means).

## Running a study

`ml/study.py` runs a list of solver cases and collects their manifests
into one table. `ml/gci.py` takes three or more grids of the same case and
reports the observed order, the extrapolated value and the grid
convergence index (GCI).

A study file names the binary, the directory to run it from, the output
directory and the cases. `ml/examples/ahmed25_grid.yaml` runs the Ahmed
body on three grids. Paths are relative to the study file. `base` applies
to every case, `sweep` lists values to cross, and `cases` are crossed with
the sweep. Keys in `vary_with_grid` may change with the grid without
splitting a GCI group.

```bash
cmake -B build -S simulation -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
uv run python ml/cli.py study ml/examples/ahmed25_grid.yaml
uv run python ml/cli.py gci build/studies/ahmed25_grid/results.json
```

Each case gets a directory under `cases/` with its command line
(`cmd.txt`), log, manifest and exit status. Running the study again skips
cases that already have a manifest for the same parameters and retries
the failed ones. `--force` reruns everything and `--dry-run` prints the
commands. `results.csv` and `results.json` hold one row per case. The
summary warns about failed cases, mixed commits, dirty trees, mixed GPUs
and runs that the solver capped below the requested Reynolds number.

The solver caps Re where tau would drop below 0.52, and the cap is lower
on coarse grids. Pick an Re that the coarsest grid can reach, or the grids
run different flows.

On the cluster, `--backend slurm` writes one sbatch script per case (with
`--gres=shard:1`) and a `submit.sh`. `--submit` calls `sbatch` itself.
When the jobs are done, `study.py STUDY --collect` builds the table.

`gci` follows Celik et al. (2008) with a safety factor of 1.25. It gives
no order when the change between grids is within twice the runs' combined
standard error, when convergence is oscillatory or divergent, or when the
order falls outside 1 to 4. It prints the reason instead. Otherwise it
reports the fine-grid value with an uncertainty that combines the GCI and
twice the standard error. `--plot` needs the `plots` extra.

Meshes from the automotive datasets come as STL.
`uv run python ml/cli.py convert body.stl` writes `body.obj` with
duplicate vertices merged and degenerate triangles dropped.
