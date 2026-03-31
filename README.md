<p align="center">
  <img src="website/public/logo.png" alt="Lattice" height="80">
</p>

# Lattice

GPU-accelerated wind tunnel that simulates fluid flow around 3D objects using Lattice Boltzmann Methods and OpenGL compute shaders. Written in C with a Next.js web frontend for remote rendering.

The simulation runs a D3Q19 LBM grid on the GPU, pushes hundreds of thousands of particles through the velocity field, and computes drag coefficients in real time. You can run it locally with a desktop window or trigger headless renders on a cloud GPU through the website.

## What's in here

```
simulation/   C simulation engine (LBM + OpenGL compute shaders)
website/      Next.js frontend that talks to a Modal GPU worker
```

### Simulation (`simulation/`)

The core fluid solver. Particles are advected through a 3D velocity field computed by the LBM kernel, with collision detection against arbitrary OBJ meshes. Everything runs on the GPU via OpenGL 4.3 compute shaders.

- `src/` -- simulation loop, LBM solver, particle system, rendering
- `shaders/` -- compute shaders for LBM collision/streaming and particle updates
- `lib/` -- headers for fluid grid, particles, OpenGL helpers
- `assets/` -- OBJ models (car, Ahmed bodies) and fonts
- `obj-file-loader/` -- lightweight Wavefront OBJ parser

### Website (`website/`)

A Next.js app that lets you configure simulation parameters and kick off GPU renders via Modal. Styled with the Catppuccin Mocha palette.

Features:
- Configure wind speed, visualization mode, collision mode, and model selection
- Upload custom OBJ files for testing your own geometry
- Drag coefficient (Cd) readout with comparison table across runs
- Shareable URLs via hash-encoded parameters
- Demo mode when no GPU backend is configured

## Building the simulation

### Dependencies

You need CMake, a C11 compiler, and OpenGL 4.3+ capable GPU drivers.

**Debian/Ubuntu:**
```bash
sudo apt-get install build-essential cmake pkg-config \
  libsdl2-dev libsdl2-ttf-dev libglew-dev mesa-common-dev libgl1-mesa-dev
```

**Arch:**
```bash
sudo pacman -S base-devel cmake sdl2 sdl2_ttf glew mesa
```

### Build and run

```bash
cmake -B build -S simulation -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/3d_fluid_simulation_car
```

Run from the repo root so the binary can find `simulation/assets/`.

## Running the website

```bash
cd website
npm install
npm run dev
```

The site works in demo mode without any backend. To enable GPU rendering, set `MODAL_RENDER_ENDPOINT` in `website/.env.local`:

```
MODAL_RENDER_ENDPOINT=https://your-modal-endpoint.modal.run
```

### Deploying the Modal worker

```bash
cd simulation
pip install modal
modal deploy modal_worker.py
```

This gives you a URL to set as `MODAL_RENDER_ENDPOINT`. You'll also need an `aws-secret` in Modal with `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` for S3 video uploads.

## Controls (desktop simulation)

| Key | Action |
|---|---|
| Mouse drag | Orbit camera |
| Scroll wheel | Zoom |
| W/S | Tilt up/down |
| A/D | Rotate around target |
| Q/E | Zoom in/out (step) |
| R | Reset camera |
| Up/Down | Wind speed |
| V | Cycle viz mode |
| 3-9 | Select viz mode |
| Left/Right | Color sensitivity |
| 0/1/2 | Collision off/AABB/per-triangle |
| Esc | Quit |

## Visualization modes

0. **Depth** -- distance from camera
1. **Velocity magnitude** -- blue (slow) to red (fast)
2. **Velocity direction** -- RGB mapped to XYZ velocity components
3. **Particle lifetime** -- age of each particle
4. **Turbulence** -- laminar vs turbulent regions
5. **Flow progress** -- rainbow gradient by X position
6. **Vorticity** -- lateral motion indicator

## Citing

If you use Lattice in academic work, you can cite it with the BibTeX below (or click "Cite this repository" on GitHub):

```bibtex
@software{ashton_lattice,
  author  = {Ashton, Marcos},
  title   = {Lattice: {GPU}-accelerated {Lattice Boltzmann} fluid dynamics},
  url     = {https://github.com/MarcosAsh/Lattice_Fluid_Dynamics},
  license = {MIT}
}
```

## License

MIT
