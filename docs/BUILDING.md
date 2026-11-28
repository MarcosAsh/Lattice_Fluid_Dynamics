# Building

## Requirements

- C17 compiler (GCC or Clang)
- CMake 3.22+
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
--collision-operator=OP  bgk, regularized, mrt, entropic or recursive
                     (default: mrt)
--strict-reynolds    Fail if the grid cannot reach --reynolds
--tau-floor=TAU      Developer: lowest tau before Re is capped (default 0.52)
--manifest=PATH      Run manifest JSON
--converge=TOL       Stop once the 95% CI of mean Cd is within TOL
--converge-window=N  Cd samples between checks (default: 20)
--max-duration=SECS  Frame cap for --converge
--checkpoint=PATH    Write the solver state at the end of the run
--checkpoint-interval=FRAMES  Also write it every FRAMES frames
--restart=PATH       Continue the run saved in a checkpoint
--surface-output=PATH  Time-averaged Cp and wall force on the body
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
    --converge=0.01 --max-duration=300 --manifest=run.json
```

Every run keeps its Cd samples, one every 20 frames, and puts an error bar
on the mean at the end, with or without `--converge`. The result is
printed after `Drag statistics:` and stored in the `convergence` section
of the manifest.

The error bar comes from four steps.

1. Cut the start-up transient with MSER-5 (White 1997). The samples are
   averaged in groups of five, and the cut goes where the standard error
   of the remaining samples is smallest. The cut is at most half the run.
   If the smallest value lies in the second half, the run is marked
   `transient may not be over`. A force oscillation that is still dying
   out often causes this.
2. Estimate the standard error of the mean from overlapping batch means
   (Meketon and Schmeiser 1984). The batch size grows with the square root
   of the sample count (Flegal and Jones 2010). The lugsail correction
   (Vats and Flegal 2022) offsets the low bias of short batches on
   correlated samples. Batch sizes are multiples of 6, so a force that
   alternates from one sample to the next cancels inside each batch.
3. Build a 95 percent confidence interval with a Student t quantile.
4. Test the kept samples for a shift in the mean with the Cramer-von Mises
   test of Heidelberger and Welch (1983). A straight line is also fitted,
   and its rise across the kept samples is reported as the drift
   (Scandurra et al. 2025).

The effective sample size is the number of independent samples that would
give the same standard error. It can be larger than the sample count when
neighbouring samples are anticorrelated. The lag-1 autocorrelation is
printed too. A value near -1 means the force alternates from sample to
sample, and the summary then reports a period-2 oscillation.

With `--converge=TOL` the run stops at the first check where all of these
hold:

- at least 200 samples are left after the cut
- the effective sample size is at least 50
- the mean passes the stationarity test
- the 95 percent half-width is less than `TOL` times the mean
- the drift is less than `TOL` times the mean

A check runs every `--converge-window` samples (default 20).
`--max-duration` caps the run. A run that reaches the cap prints why it
did not converge.

`TOL` used to bound the change between the means of two windows. It is now
the relative half-width, so `--converge=0.01` asks for the mean Cd to
within 1 percent at 95 percent confidence.

The manifest records the method name and version, the sample counts
(total, cut, kept, and NaN or Inf samples dropped), the mean, standard
deviation, standard error, 95 percent half-width, effective sample size,
lag-1 autocorrelation, drift and the stationarity result. Counts are in Cd
samples and Cd has no units. Values that could not be computed are `null`,
and `reason` says why.

On synthetic series the interval covered the true mean in 94 to 98 percent
of trials, for white noise, strongly correlated noise and a period-2
oscillation. About 94 percent of runs stopped by the rule were still
covered. The rule never stopped on a drift of six times `TOL` across 200
samples. A drift close to `TOL` can slip through, because a short series
cannot tell it from noise.

References:

- K. P. White, "An effective truncation heuristic for bias reduction in
  simulation output", Simulation 69(6), 1997.
- M. S. Meketon and B. Schmeiser, "Overlapping batch means: something for
  nothing?", Winter Simulation Conference, 1984.
- J. M. Flegal and G. L. Jones, "Batch means and spectral variance
  estimators in Markov chain Monte Carlo", Ann. Statist. 38(2), 2010.
- D. Vats and J. M. Flegal, "Lugsail lag windows for estimating
  time-average covariance matrices", Biometrika 109(3), 2022.
- P. Heidelberger and P. D. Welch, "Simulation run length control in the
  presence of an initial transient", Oper. Res. 31(6), 1983.
- L. Scandurra, P. Alexias and E. de Villiers, "A framework for initial
  transient detection and statistical assessment of convergence in CFD
  simulations", arXiv:2511.22618, 2025.

### Checkpoint and restart

```bash
./build/3d_fluid_simulation_car --model=assets/3d-files/ahmed_25deg_m.obj \
    --duration=60 --checkpoint=run.ckpt --checkpoint-interval=1000
./build/3d_fluid_simulation_car --model=assets/3d-files/ahmed_25deg_m.obj \
    --duration=120 --restart=run.ckpt --checkpoint=run.ckpt
```

`--checkpoint=PATH` writes the solver state when the run ends.
`--checkpoint-interval=FRAMES` also writes it every FRAMES frames, over
the same file. The file goes to `PATH.tmp` first and is then renamed, so
a crash never leaves half a checkpoint at PATH.

`--restart=PATH` continues the saved run. Give it the same model, grid
and physics options as the first run. `--duration` counts from the start
of the first run, so the second command above runs for 60 more seconds.
On the same GPU and driver, a restarted run ends with the same
distributions, drag samples and statistics as a run that never stopped.
The integration test checks this byte for byte.

A checkpoint holds:

- f - w for every cell
- the step and frame counters, which also fix the inlet ramp and the
  phase of the inlet turbulence
- the open force averaging window and the stability guard counters
- every Cd and Cl sample, the running sums and the convergence state,
  so the drag statistics carry on
- the frame where Cd sampling starts

Particles are reseeded. `--vtk-stats` averages start again from zero.
The post-collision buffer and the velocity buffer are not saved, because
the first step after a restart rewrites both before anything reads them.

The header stores the grid, lattice, storage convention (f - w), float
size, collision operator, tau, inlet velocity, Smagorinsky constant,
sponge, periodic flags, inlet turbulence and body force. It also stores
FNV-1a hashes of the solid mask and of the Bouzidi wall distances, the
model path, the git commit and the step. A restart is refused with exit
status 2 when any of these differ, and the message names the field. Only
a different model path with matching hashes is let through, with a
warning. The hashes are exact, so a build that computes the wall
distances with other rounding cannot restart the file.

The file ends with a CRC-32 of everything before it. A truncated file or
a bad checksum is refused before any state is loaded. A checkpoint needs
about 76 bytes per cell, 2.5 GB for 512x256x256. It is read and written
in chunks of at most 64 MB.

Layout, in the byte order of the writing machine, which a marker in the
header records:

| Part | Content |
|---|---|
| Header | 1024 bytes: magic `LATCKPT`, format version, byte order marker, the fields above |
| Section | 4-byte tag, 4 zero bytes, 8-byte length, data |
| `LBMS` | step counter, force window state, guard counters |
| `FBUF` | f - w, 19 floats per cell, x fastest |
| `FRCE` | force sums of the open window, one per workgroup |
| `RUNS` | frame counters and drag statistics of the wind tunnel loop |
| `CDSR`, `CLSR` | Cd and Cl samples |
| End | tag `END!`, CRC-32 of all bytes before the CRC |

The manifest records the checkpoint read under `restarted_from` (path,
commit, step and frame) and the one written under `checkpoint`.

A checkpoint only continues a run on the same grid. Interpolating one
onto a finer grid is not supported.

### Surface pressure and shear

```bash
./build/3d_fluid_simulation_car --model=assets/3d-files/ahmed_25deg_m.obj \
    --duration=60 --surface-output=out/surface.vtk
```

`--surface-output=PATH` averages the momentum exchange and the density of
every boundary cell over the steps that make up the Cd samples. It reads
the same buffers as the drag shader and changes neither the flow nor the
drag. Three files are written when the run ends:

- PATH: the mesh triangles with `cp`, `cp_wall`, `traction`, `shear`,
  `normal`, `cells` and `samples` per triangle, as legacy VTK PolyData.
  ParaView reads it.
- `STEM_cells.vtk`: one point per boundary cell with `cp`, `cp_wall`,
  `cf`, `rho`, `area`, `links`, `samples`, `force`, `traction`, `shear`
  and `normal`.
- `STEM_plane.csv`: the boundary cells within half a cell of the body's
  symmetry plane, with `theta_deg`, the angle from upstream, positive
  toward the lift side.

STEM is PATH without `.vtk`. Positions are in lattice cells from the grid
corner, forces in lattice units, and `samples` counts the averaged steps.
The manifest has a `surface` section with the cell force sum, the mesh
integral and the mean drag readback over the same steps. A restart starts
the surface averages again.

How far each value can be trusted:

- `force` is exact for the scheme. It is the time-averaged momentum
  exchange of the Bouzidi links of one fluid cell, and the cells add up
  to the reported drag to float round-off (5e-8 or better on the sphere
  checks). How the force splits between neighbouring cells depends on
  the staircase, so single cells scatter.
- `cp` is (p - p_inf) / (0.5 rho U^2) with p = rho / 3 at the centre of
  the boundary cell, which sits up to 1.7 cells off the wall. p_inf is
  the mean density over the plane x = 1. Single cells scatter by a few
  tenths of Cp next to the staircase. Averaged over 15 degree bins on a
  12 cell sphere it is within 0.125 of the reference curve, so it is the
  better pressure estimate.
- `cp_wall` is the pressure from the normal force. At rest a cell's
  links give a force of rho times their weight sum W, so a cell's wall
  pressure is (F.n) / (3 W.n). This takes out the staircase, which on a
  D = 8 sphere sees 13% more area than the mesh. On the mesh each
  triangle takes the ratio of the sums over its nearby cells, which
  scatters much less than the ratio per cell in `STEM_cells.vtk`. Cells
  whose links see less than 0.25 cells^2 get no per-cell value.
  `cp_wall` overshoots the wall pressure, by up to 0.22 on a 12 cell
  sphere at Re 20. A likely cause, not checked, is the normal viscous
  stress at the first fluid node, which is zero only at the wall itself.
- `shear` is the tangential force over the same link area, and `cf` its
  magnitude over 0.5 rho U^2. It needs a resolved boundary layer and is
  the least reliable value.
- `area` and `traction` (force / area) come from the mesh. Triangles are
  split until no edge is longer than 0.75 cells. Each cell spreads its
  force over the triangles within 2 cells, weighted by area times
  (1 - d/2)^2. The triangle forces add up to the cell forces, so the
  mesh integral of `traction` equals the drag. The areas add up to the
  mesh area. `traction` keeps the staircase bias, so use `cp_wall` for
  pressure.

On the mesh, every value is smoothed over about two cells. Without a mesh
(the AABB fallback) PATH holds the cells and the normals come from the
links. A sphere needs about 12 cells across for a Cp curve of the right
shape, see `docs/validation/REPORT.md`.

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

The solver caps Re where tau would drop below 0.52 (or `--tau-floor`),
and the cap is lower on coarse grids. Pick an Re that the coarsest grid can reach, or the grids
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
