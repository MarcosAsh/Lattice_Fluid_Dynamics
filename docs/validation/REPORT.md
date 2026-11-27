# Validation of the GPU LBM wind tunnel

Hardware: NVIDIA RTX 5070 Ti, driver 615.78.08, headless EGL (all numbers
below unless marked laptop). Laptop: Intel Arc (Meteor Lake) iGPU, Mesa
26.0.8. Raw logs, results.json and plots are in this directory.

## Method

`simulation/validation/lbm_validate` runs each case through the same
shaders and `LBM_*` API as the wind tunnel and prints one RESULT line.
`simulation/validation/run_validation.py` runs the set, checks tolerances
and writes `results.md`, `results.json` and the plots.
`ctest -L validation` runs the quick set. Full set, from `simulation/`:

    uv run --with numpy --with matplotlib python validation/run_validation.py \
        --build build --out ../docs/validation/final

Lattice units throughout. U = 0.05 (Ma 0.087) unless stated. Cd = Fx /
(0.5 rho U^2 A) with A the exact frontal area (pi D^2/4 for spheres, mesh
silhouette for OBJ bodies). Forces are momentum exchange on Bouzidi links,
averaged over every step of a window.

## Bugs found and fixed

| Bug | Before | After |
|---|---|---|
| Force summed as int with a 1e-4 quantum | Poiseuille wall force off by up to 20% | within 5e-5 of the body force |
| Deep GPU queues dropped work | Cd = 0.000 in every sample of an Ahmed 35 run, test_lbm 83/90 and 89/90 | bit-identical repeats, 90/90 |
| Entropic alpha solved from absolute entropies in float32 | TGV nu +16.7% (laptop) | -1.1e-4 |
| Smagorinsky used 18 instead of 18 sqrt(2) | nu_t 0.71 of the standard model | 0.999 |
| Mesh Bouzidi q culled with the +x ray box | 4236 of 4704 links at q = 0.5 (D=16), sphere Cd 1.374 (D=8, Re 100, laptop) | 0 at q = 0.5, Cd 1.256 |
| Periodic z read outside the slab when f is split | 187808 resets, mass -42% (laptop) | identical to one slab |
| Flow started at U with the inlet ramping from 0 | Ahmed 64 grid Cd samples 0.37 to 8.90 (laptop) | 3.37 to 3.52 |
| Cd from the voxel area | Ahmed A = 140 cells^2 | mesh A = 155.6 cells^2 |
| Cl from y, the side axis of the Ahmed meshes | side force reported as lift | lift from z |
| Reset cells and NaN were silent, Cd used abs(fx) | | warning per window, exit 3 on NaN, 4 on resets, signed Cd |
| --mrt did nothing, BGK and regularized not selectable | | --collision-operator=bgk, regularized, mrt, entropic |
| Re cap at tau 0.52 was silent and the manifest lost the requested Re | | warning, --strict-reynolds, both Re in the manifest |

Baseline Ahmed 25 at 128x64x64, Re 200: Cd 2.921 (sd 0.042) with the
voxel area. With all fixes: 2.695. Rescaling the baseline to the mesh area
gives 2.628, so the area accounts for -11% and the solver fixes for +2.5%.

## Results

### Uniform flow (64x32x32, MRT, tau 0.52, 4000 steps)

Closed periodic box: relative mass drift -2.4e-7. Open tunnel: max |u - U|
1.5e-5, max |rho - 1| 6e-7, zero force.

### Taylor-Green vortex (final/tgv_viscosity.png)

nu_eff from the decay of kinetic energy. N = 64, tau 0.6, U0 0.02:
BGK -1.1e-4, regularized -2.4e-4, MRT -3.0e-4, entropic -1.1e-4. At
tau 0.503, U0 0.05: MRT -2.6e-3, entropic +7.7e-4. Smagorinsky eddy
viscosity against the analytic value for this field: 0.999 (Cs 0.1),
0.995 (Cs 0.2). The same run on 4 SSBO slabs is bit-identical to one.

### Poiseuille channel (final/poiseuille_convergence.png)

Body force, walls at distance q from the last node, tau 0.8. In all 18
runs the wall force equals the body force within 4.6e-5. L2 order from
H = 8 to 16 is 1.87 to 2.13 for BGK and MRT at q = 0.25, 0.5, 0.75.

### Sphere (sphere/sphere_cd.png, sphere/gci_*.png)

Domain 16D x 8D x 8D, periodic sides, sphere 6D from the inlet, blockage
1.2%, MRT, 60 D/U.

| Re | Reference (CGW) | D=8 | D=12 | D=16 | Extrapolated (gci) |
|---|---|---|---|---|---|
| 20 | 2.715 | 2.918 | 2.832 | 2.810 | 2.793, p = 2.82, GCI_fine 0.79% |
| 100 | 1.087 | 1.257 | 1.154 | 1.131 | 1.116, p = 3.21, GCI_fine 1.68% |

Clift, Grace and Weber (1978): Cd = 24/Re (1 + 0.1315 Re^(0.82-0.05 log Re))
at Re <= 20, and 24/Re (1 + 0.1935 Re^0.6305) for 20 < Re <= 260 (2.735 at
Re 20). Schiller-Naumann gives 2.61 and 1.09.

In a 32D x 16D x 16D domain at D = 8, Cd drops by 1.8% (Re 20) and 1.4%
(Re 100), so the 8D box adds about that much. Other checks at Re 100:
icosphere through the OBJ path, D=16: 1.1305 against 1.1308. Smagorinsky
Cs 0.1 raises Cd by 0.95%. BGK at D=12 gives 1.1546 against MRT 1.1540.

### Ahmed body (final/ahmed_cd_series.png)

Production binary, MRT + Smagorinsky 0.1, Re 200 based on length, no ground.

| Angle | Grid | tau | Cd (last half) | sd | Cl |
|---|---|---|---|---|---|
| 25 | 128x64x64 | 0.529 | 2.695 | 0.278 | 0.135 |
| 25 | 256x128x128 | 0.558 | 2.658 | 0.437 | 0.158 |
| 35 | 128x64x64 | 0.529 | 2.690 | 0.275 | 0.150 |
| 35 | 256x128x128 | 0.558 | 2.655 | 0.433 | 0.166 |

The wake sheds with St = 0.29 on body height, so samples vary by 10 to 16%.
Block means on the 256 grid rise from 2.61 to 2.73 and are not yet
stationary. 25 and 35 degrees differ by 0.2%.

## Reachable Reynolds number

tau is floored at 0.52 and U is 0.05, so Re_max = 7.5 L with L the body
length in cells. The Ahmed body spans about 0.3 Nx, so Re_max is about
2.25 Nx: 144 at Nx 64, 288 at 128, 576 at 256 and 1152 at 512. The experimental Ahmed Re is 4.29e6. These
grids can show low-Re trends. They cannot reproduce Cd 0.285 (25 deg) or
0.26 (35 deg).

## Known limits

- No ground plane by default, no wall model, laminar or Smagorinsky only.
- The Ahmed Cd here is a Re 200 laminar result and is not comparable to experiment.
- Sphere drag at D <= 8 is 8 to 16% high. Use D >= 16.
- float32 storage leaves a mass drift of about 2e-8 per step in a closed
  box and limits Poiseuille accuracy at H = 32. Storing f - w (Lehmann et
  al. 2022) would fix it and is not done.
- MRT viscosity error on TGV grows with N, 8e-4 at N = 128. Not explained.
- The cause of dropped work on NVIDIA is not identified. Bounding the
  queue removed it in every run made.
- Pressure and friction split uses the local equilibrium and is approximate.
- The sphere GCI assumes a sample standard error of sd/sqrt(windows/4).
- Not run: sphere at Re 300, Strouhal checks, wake and Cp profiles.
