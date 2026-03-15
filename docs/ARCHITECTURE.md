# Architecture

High-level overview of how the pieces fit together.

## Simulation (C + OpenGL 4.3)

The core is a Lattice Boltzmann Method (LBM) solver running entirely
on the GPU via OpenGL compute shaders.

```
main.c          Entry point, SDL window, main loop, CLI parsing
lbm.c / lbm.h  LBM grid creation, solid voxelization, Cd/Cl computation
```

### Compute shaders (simulation/shaders/)

| Shader              | Purpose                                      |
|---------------------|----------------------------------------------|
| lbm_collide.comp    | D3Q19 collision (BGK or regularized), BCs    |
| lbm_stream.comp     | Streaming step (pull from neighbors)         |
| lbm_force.comp      | Momentum exchange for drag/lift forces       |
| particle_lbm.comp   | Particle advection using LBM velocity field  |
| particle.comp       | Legacy particle update                       |

### Data flow per frame

```
for each substep (5x per frame):
    lbm_collide.comp   f -> f_new   (boundary conditions + collision)
    lbm_stream.comp    f_new -> f   (propagate distributions)

particle_lbm.comp      sample velocity field, move particles
lbm_force.comp         compute drag/lift (every 60 frames)
render particles       draw as GL_POINTS with color modes
```

### Solid voxelization

OBJ meshes are loaded via `obj-file-loader/`, transformed (scale,
center, rotate), then rasterized to the LBM grid using ray casting
in `LBM_SetSolidMesh()`. Each grid cell is marked solid or fluid.

### Drag coefficient

Computed via momentum exchange (Mei-Luo-Shyy method) in the force
shader. For each fluid cell adjacent to a solid cell, the momentum
transferred through bounce-back is accumulated with atomic adds.
The CPU reads back the force buffer and normalizes by dynamic
pressure and frontal area.

## Website (Next.js + React)

```
website/src/
  app/
    page.tsx              Main page, state management, render trigger
    api/render/route.ts   Backend route, rate limiting, Modal proxy
    docs/page.tsx         White paper / documentation
  components/
    ControlPanel.tsx      Parameter sliders and model selection
    StatusDisplay.tsx      Job progress bar
    VideoPlayer.tsx        S3 video playback
    ResultsPanel.tsx       Cd/Cl display, CSV export, run history
```

### Render flow

1. User sets params in ControlPanel, clicks Render
2. `page.tsx` POSTs to `/api/render`
3. Route handler forwards to Modal endpoint
4. Modal runs the simulation, encodes video, uploads to S3
5. Response comes back with video URL + Cd/Cl values
6. VideoPlayer shows the video, ResultsPanel shows coefficients

## Infrastructure

### Modal (GPU compute)

`simulation/modal_worker.py` defines the Modal app:

- **build_simulation()** -- clones repo, runs cmake + make
- **render_simulation()** -- starts Xvfb, runs the binary headless,
  encodes frames to MP4 with ffmpeg, uploads to S3
- **render_endpoint()** -- HTTP POST wrapper for render_simulation

Uses a T4 GPU, Debian Slim image with OpenGL (software Mesa),
and a persistent build cache volume.

### Vercel

Hosts the Next.js frontend. Deploys automatically from master.
The `MODAL_RENDER_ENDPOINT` env var connects it to Modal.

### S3

Rendered videos are stored in the `fluid-sim-renders` bucket
(eu-west-2). URLs follow the pattern:
`https://fluid-sim-renders.s3.eu-west-2.amazonaws.com/renders/{job_id}.mp4`
