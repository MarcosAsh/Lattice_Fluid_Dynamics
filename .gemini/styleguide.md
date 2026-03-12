# Lattice Code Review Guidelines

This is a GPU-accelerated wind tunnel simulator. The simulation engine is written in C with OpenGL 4.3 compute shaders implementing a D3Q19 Lattice Boltzmann Method. The web frontend is a Next.js app (TypeScript, Tailwind) that triggers headless GPU renders via Modal.

## Project structure

```
simulation/   C engine -- LBM solver, particle system, OBJ loader, OpenGL rendering
website/      Next.js frontend -- parameter controls, video playback, results display
```

These two halves talk through a Modal GPU worker (`simulation/modal_worker.py`) and a Next.js API route (`website/src/app/api/render/route.ts`).

## C simulation code

### Correctness over cleverness

The LBM solver and compute shaders are the core of this project. Any change to collision, streaming, force computation, or boundary handling needs to be physically justified. Reviewers should check:

- **Unit consistency.** The simulation uses lattice units internally. Any value crossing the boundary between world coordinates and lattice coordinates (drag force, reference area, velocity) must be scaled correctly. We already had a bug where Cd was off by orders of magnitude because the reference area was in world units but the force was in lattice units.
- **Numerical stability.** LBM is sensitive to the relaxation parameter tau and inlet velocity. Changes that affect these should demonstrate they don't blow up after a few hundred timesteps.
- **Memory management.** Every `malloc` needs a `free`, every `glGenBuffers` needs a `glDeleteBuffers`. We've had shader source leaks before. Check cleanup paths, especially error/early-return paths.

### OpenGL and compute shaders

- Shader code lives in `simulation/shaders/`. Changes to buffer bindings in shaders must match the `glBindBufferBase` calls in C code.
- Dispatch sizes must cover the full grid: `(sizeX + 7) / 8` etc. Off-by-one here means cells at the grid boundary get skipped.
- Always call `glMemoryBarrier` after dispatching a compute shader that writes to a buffer that will be read by the next dispatch.

### Style

- C11, no C++ features.
- Functions should be short and do one thing. The main loop is already too long, don't make it worse.
- Use `float` for simulation values, not `double`. GPU buffers are single precision.
- Printf debugging is fine for development but should not be left in hot paths (the render loop runs at 60fps).

## Website code

### React and Next.js

- This is a Next.js App Router project. No Pages Router patterns.
- Components go in `website/src/components/`, API routes in `website/src/app/api/`.
- State that can be derived should be derived, not stored separately. We got bitten by the React strict mode lint rule about calling setState synchronously in effects -- prefer deriving values in the render body or using callbacks inside intervals/event handlers.
- Don't reach for `useEffect` when a value can be computed directly from props or existing state.

### Styling

- The entire frontend uses the Catppuccin Mocha color palette. All colors should use the `ctp-*` tokens defined in `globals.css`. Do not use raw hex colors or default Tailwind colors.
- Accent color is `ctp-mauve`. Interactive elements (buttons, focus rings, progress bars) use it.
- Background hierarchy: `ctp-crust` (deepest) < `ctp-mantle` (panels) < `ctp-base` (page) < `ctp-surface0/1` (elevated).

### API route

- The render API route at `/api/render` is the only bridge to the GPU backend. It must validate input (especially custom OBJ size) and handle the case where `MODAL_RENDER_ENDPOINT` is not set (demo mode, return 503).
- Never trust client input. The OBJ data is base64-encoded and can be up to 5MB. Validate size before forwarding.

## Modal worker

- `simulation/modal_worker.py` runs on Modal with a T4 GPU.
- The worker clones the repo, builds the C simulation, runs it headless via Xvfb, encodes the frames to MP4 with ffmpeg, and uploads to S3.
- Changes to CLI flags in the C code must be reflected in the worker's command construction. The worker already checks `--help` output to detect supported flags.
- Stdout parsing (for Cd, Cl values) is fragile -- it splits on `Cd=` and `Cl=`. If the printf format changes in C, the parser breaks silently. Keep these in sync.

## Code review principles

When reviewing pull requests, follow these rules:

1. **Question necessity.** Don't assume every change is needed. If a PR adds complexity without clear benefit, say so. A three-line fix is better than a ten-line abstraction.

2. **Check the physics.** For simulation changes, verify the math. LBM has well-known reference results (Ahmed body Cd around 0.25-0.35 at 25 degrees). If a change would produce nonsensical values, flag it.

3. **Spot fragile patterns.** Stdout parsing, string splitting, implicit ordering dependencies between GPU dispatches -- these are all places where things break silently. Point them out.

4. **No dead code.** Don't accept commented-out code, unused imports, or functions added "for future use." If it's not used by this PR, it shouldn't be in this PR.

5. **No AI slop.** Watch for overly verbose code, unnecessary abstractions, hallucinated APIs, or generic boilerplate that doesn't match the project's style. If the code reads like it was generated without understanding the codebase, call it out.

6. **Match existing patterns.** Before suggesting a new pattern, check if there's an established way to do it in the repo. The frontend has consistent patterns for panels, sliders, selects, and status display. The C code has consistent patterns for shader loading, buffer management, and uniform passing. Follow them.

7. **Error messages matter.** If a user-facing error is added, it should say what went wrong, what was expected, and how to fix it. "Render failed" is not good enough.
