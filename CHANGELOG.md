# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [2.0.0] - Unreleased

Drag values from 2.0.0 differ from 1.5.0. Several solver bugs are fixed
and Cd now uses the mesh frontal area. Do not mix results from the two
versions. See [docs/validation/REPORT.md](docs/validation/REPORT.md).

### Added

- Validation suite: `simulation/validation/lbm_validate`,
  `run_validation.py`, `ctest -L validation`, and the report in
  `docs/validation/`.
- `--collision-operator=bgk|regularized|mrt|entropic|recursive`. `--mrt`
  and `--entropic` remain as aliases. `recursive` is recursive
  regularized on D3Q19.
- `--tau-floor=TAU`, a developer flag to run below the default floor of
  0.52.
- Distributions are stored as f - w, which removes most float32
  round-off in mass and restores second-order Poiseuille convergence.
- `--strict-reynolds` fails when the grid cannot reach `--reynolds`.
- Run manifest JSON with the git commit, parameters, lattice setup, GPU
  and drag results. Written with `--output` or `--manifest=PATH`.
- A 95% confidence interval on mean Cd for every run, with transient
  cut, batch means, stationarity test and drift.
- `--converge-window` and `--max-duration`.
- `--checkpoint`, `--checkpoint-interval` and `--restart` save a run and
  continue it. On the same GPU and driver a restarted run matches an
  uninterrupted one bit for bit.
- `--surface-output` writes time-averaged Cp, wall pressure and shear on
  the body as VTK, plus a CSV along the symmetry plane.
- `--version`.
- Entropic collision and inlet turbulence (`--turbulence`).
- Outlet sponge layer, on by default, width set with `--sponge=CELLS`.
- `--vtk-stats` for time-averaged turbulence statistics, and density in
  VTK dumps.
- Large grids split across several storage buffers.
- Periodic x boundaries in the LBM API.
- `ml/study.py` study runner, local or Slurm, and `ml/gci.py` grid
  convergence index.
- STL to OBJ converter (`ml/cli.py convert`) and the `lattice-ml` CLI.
- Any-mesh drag surrogate with trained weights, OBJ voxelizer and
  preprocessing pipeline.
- The OBJ loader triangulates polygon faces.
- pytest suite and CTest for the C tests.
- CMake presets (`release`, `debug`, `asan`, `ci`) and the
  `LATTICE_WARNINGS`, `LATTICE_WERROR` and `LATTICE_SANITIZE` options.
- OpenGL debug output in debug builds, set with `LATTICE_GL_DEBUG`.
- Website: side-by-side run comparison, getting started tutorial,
  not-found and error pages, Playwright smoke test, reduced motion
  support.
- Quickstart and maintainer path in `CONTRIBUTING.md`.

### Changed

- Command line parsing is strict. Invalid numbers and unknown options
  exit with status 2.
- `--converge=TOL` now means the 95% confidence interval half-width of the
  mean Cd, relative to the mean. It used to bound the change between two
  window means. `--converge=0.01` asks for Cd within 1%.
- Cd is signed. It used to be computed from `abs(fx)`.
- Lift is taken along the shorter cross axis of the body, z for the Ahmed
  meshes and y for the car. The Ahmed meshes used to report side force as
  lift. Side force is printed as `Cside`.
- The Cd reference area comes from the mesh silhouette. It used to come
  from the voxels. Ahmed body Cd drops by about 11% from this alone.
- A NaN in the flow exits with status 3. Cells reset by the stability
  guard during Cd sampling print a warning per window and exit with
  status 4.
- The flow starts at rest to match the inlet ramp.
- The solver warns when it caps the requested Reynolds number, and the
  manifest records both the requested and the reached value.
- Boundary forces are averaged over every step of the sampling window.
- No ground plane by default.
- Headless runs render on the GPU through EGL.
- The Python tools are a uv project and need Python 3.12 or later. Use
  `uv sync` and `uv run`.
- The build needs CMake 3.22 and a C17 compiler, and uses imported CMake
  targets.
- `docs/BENCHMARK.md` and `docs/grid_convergence/REPORT.md` are marked
  superseded. Their numbers predate the solver fixes and should not be
  cited.
- Website: Next.js, React, ESLint, Playwright, Tailwind and Vitest
  updated, Node 26 in CI, stricter TypeScript, docs and home pages
  rendered on the server, render API input validated.

### Fixed

- Boundary force was summed as an integer with a 1e-4 quantum.
  Poiseuille wall force was off by up to 20%.
- Deep GPU queues dropped work and gave Cd 0.000. The queue is now
  bounded.
- Entropic alpha lost precision in float32 and raised the viscosity by
  16.7%.
- Smagorinsky used 18 in place of 18 sqrt(2).
- Bouzidi wall distances on meshes were culled with the wrong box.
- Periodic z read outside the slab when the grid was split.
- Headless runs without `--output` never ended.
- Turbulence components were correlated.
- Unchecked `fread`, `fwrite` and allocations, a dead triangle bounds
  check and a cell count overflow.
- Modal render cache served stale results after a push.
- Website: NaN surrogate estimate, unsafe saved state hydration, locked
  controls in demo mode.

### Removed

- SDL_ttf is no longer needed.
- The dead CPU solver code.
- The unused Arial font from the website.

## [1.5.0] - 2026-04-03

Super-resolution velocity upscaling, FFD shape optimisation and physics
fixes.

## [1.0.0] - 2026-03-15

Zou-He boundaries, regularized collision and a training data pipeline.

## [0.5.0] - 2026-03-12

First public release, with the C solver and the Next.js site in one
repository.

[2.0.0]: https://github.com/MarcosAsh/Lattice_Fluid_Dynamics/compare/v1.5.0...v2.0.0
[1.5.0]: https://github.com/MarcosAsh/Lattice_Fluid_Dynamics/compare/v1.0.0...v1.5.0
[1.0.0]: https://github.com/MarcosAsh/Lattice_Fluid_Dynamics/compare/v0.5.0...v1.0.0
[0.5.0]: https://github.com/MarcosAsh/Lattice_Fluid_Dynamics/releases/tag/v0.5.0
