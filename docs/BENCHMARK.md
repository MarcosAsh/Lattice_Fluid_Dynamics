# Benchmark Results

GPU profiling and Cd measurements across grid sizes,
models, and wind speeds. Generated automatically.

## System Info

- GPU: 
- Date: 2026-03-15 22:06 UTC

## Results

Duration: 15s per run

| Model | Wind | Grid | Re | Cd | Solid Cells | Memory (MB) | Time (s) |
|-------|------|------|----|----|-------------|-------------|----------|
| ahmed25 | 0.5 | 128x64x64 | 80 | 19.253 | 3300 | 87.6 | 43568 |
| ahmed25 | 1.0 | 128x64x64 | 160 | 11.587 | 3300 | 87.6 | 49034 |
| ahmed25 | 1.5 | 128x64x64 | 240 | 0.000 | 3300 | 87.6 | 51363 |
| ahmed25 | 2.0 | 128x64x64 | 320 | 0.000 | 3300 | 87.6 | 48050 |
| ahmed25 | 3.0 | 128x64x64 | 480 | 0.000 | 3300 | 87.6 | 55991 |
| ahmed35 | 0.5 | 128x64x64 | 80 | 19.190 | 3276 | 87.8 | 68045 |
| ahmed35 | 1.0 | 128x64x64 | 160 | 11.548 | 3276 | 87.8 | 70171 |
| ahmed35 | 1.5 | 128x64x64 | 240 | 0.000 | 3276 | 87.8 | 68004 |
| ahmed35 | 2.0 | 128x64x64 | 320 | 0.000 | 3276 | 87.8 | 60817 |
| ahmed35 | 3.0 | 128x64x64 | 480 | 0.000 | 3276 | 87.8 | 68448 |
| car | 0.5 | 128x64x64 | 80 | nan | 1396 | 87.3 | 159913 |
| car | 1.0 | 128x64x64 | 160 | 4.103 | 1396 | 87.3 | 53731 |
| car | 1.5 | 128x64x64 | 240 | 0.000 | 1396 | 87.3 | 58192 |
| car | 2.0 | 128x64x64 | 320 | 0.000 | 1396 | 87.3 | 67906 |
| car | 3.0 | 128x64x64 | 480 | 0.000 | 1396 | 87.3 | 60243 |

## Cd vs grid (Ahmed 25 deg, wind=1.0)

Grid-convergence data from Modal runs on 2026-04-12, duration=8s.
Cd_corr is the blockage-corrected value (Pope & Harper 1966).

| Grid          | Requested Re | Effective Re | Cd      | Notes                          |
|---------------|--------------|--------------|---------|--------------------------------|
| 128x96x96     | 200          | ~125 clamped | 7.24    | tau=0.52 stability clamp hit   |
| 128x96x96     | 500          | ~125 clamped | 7.22    | clamp makes --reynolds no-op   |
| 192x128x128   | 200          | 200          | ~4-5    | not converged, 7 samples       |

On the 128 grid the Ahmed body is ~16 cells across X, which forces
tau to the stability floor and caps effective Re around 125. On the
192 grid the body spans 57.6 cells, so viscosity can be set low
enough to reach the requested Re without clamping. Cd drops with Re
as expected in the viscous-dominated regime.

See issue #155 for the full grid convergence study and #156 for
Ahmed validation.

## Ahmed Body Comparison

Published experimental Cd (Re > 10^5):
- Ahmed 25 deg: 0.25 - 0.30
- Ahmed 35 deg: 0.35 - 0.40

Our simulation runs at Re ~ 160 (limited by grid
resolution and stability). At this Re, viscous drag
dominates and Cd is much higher than the high-Re
experimental values. This is expected physics, not a
bug. Reaching experimental Cd requires either larger
grids (256+ cells in each dimension) or a subgrid
turbulence model like Smagorinsky.

## Performance Notes

The simulation uses OpenGL 4.3 compute shaders for
all LBM operations. Each frame runs 5 LBM substeps
(collision + streaming). The 128x64x64 grid uses ~86 MB
of GPU memory for distribution functions, velocity field,
and solid mask.
