# Contributing

Thanks for your interest in Lattice. Here's how to get involved.

## Getting started

1. Fork the repo and clone your fork
2. Build the simulation and website (see the README for deps)
3. Make your changes on a branch
4. Open a pull request against `master`

## What to work on

Check the [issues](https://github.com/MarcosAsh/Lattice_Fluid_Dynamics/issues) for things that need attention. If you want to work on something bigger, open an issue first so we can talk through the approach before you write a bunch of code.

Some areas that could use help:

- **Simulation accuracy** -- improving the LBM solver, better boundary conditions, turbulence modeling
- **Performance** -- profiling and optimizing the compute shaders, reducing memory bandwidth
- **New visualization modes** -- pressure fields, streamlines, temperature mapping
- **Website UX** -- mobile experience, better loading states, accessibility
- **Documentation** -- the white paper could always be clearer, and inline code comments are sparse

## Project structure

The repo has two main parts:

- `simulation/` is pure C with OpenGL compute shaders. Build with CMake.
- `website/` is a Next.js app (TypeScript, Tailwind). Run with `npm run dev`.
- `simulation/modal_worker.py` bridges the two -- it builds and runs the simulation on a cloud GPU.

## Code style

**C code:** Formatted with `clang-format` (see `.clang-format`), `make fmt` runs it. Keep functions short, name things clearly, and comment anything non-obvious in the shader code.

**Python:** Linted and formatted with `ruff`, configured in `pyproject.toml`.

**TypeScript:** The project uses ESLint with the Next.js config. Run `npm run lint` before submitting. We use the Catppuccin Mocha color palette throughout the frontend -- stick with the `ctp-*` color tokens defined in `globals.css`.

## Running tests

CI builds and tests the simulation, the Python tools and the website on
every push and pull request. To run the same checks locally:

```bash
# Simulation
cmake -B build -S simulation -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# Python
uv sync --extra torch-cpu --extra anyobj
uv run ruff check . && uv run ruff format --check .
uv run pytest

# Website
cd website
npm ci
npm run lint
npm run build
npm test
```

`make test` runs all three. See `docs/BUILDING.md` for running the GL
tests without a GPU.

## Pull requests

- Keep PRs focused. One feature or fix per PR.
- Write a short description of what you changed and why.
- If you're touching the LBM solver or shaders, include before/after screenshots or videos if possible -- it makes review much easier.
- Don't worry about squashing commits, I'll handle that on merge if needed.

## Questions?

Open an issue or start a discussion. Happy to help you get oriented in the codebase.
