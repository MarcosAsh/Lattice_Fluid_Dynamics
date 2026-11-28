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
- f - w is stored, see DDF shifting below. float32 still costs about
  8e-4 in TGV viscosity at tau 0.503 and N = 64, and 3e-3 at N = 128.
- MRT viscosity error on TGV grows with N, 8e-4 at N = 128. Float32
  round-off is the likely cause, see DDF shifting below.
- The cause of dropped work on NVIDIA is not identified. Bounding the
  queue removed it in every run made.
- Pressure and friction split uses the local equilibrium and is approximate.
- The sphere GCI assumes a sample standard error of sd/sqrt(windows/4).
- Not run: Strouhal checks, wake and Cp profiles. The sphere at Re 300
  and 1000 is in Higher Reynolds number below.

## Storing f - w (DDF shifting)

The f buffers now hold f_i - w_i (Lehmann et al. 2022, sec 2.2). The
equilibrium is computed as in their eq 27, with rho - 1 summed directly
from the stored values. Bounce-back, Bouzidi and streaming need no change
because they only mix directions of equal weight. Zou-He, the stability
guard, the sponge, the body force, the MRT moments and the entropic
operator now work on the shifted values. The entropic operator adds w
back before taking logs. Each boundary link adds 2 w_i e_i to the
momentum exchange sum. That constant is now summed separately in exact
units of 1/36 and added back on the CPU.

The checks found one more float32 problem. On the Arc iGPU the division
u = j / rho shifted the TGV viscosity by 5e-5 at tau 0.6, against a
float32 numpy port of the same arithmetic. The shader now computes
u = j - j drho / rho, so the division only touches the small term.

Numbers below are from the laptop. `ddf_shift/runs_before.txt` and
`runs_after.txt` give the `lbm_validate` options and RESULT line of every
run. `runs_tgv_rho_u_fit.txt` has the TGV and uniform reruns with the
rho u fit, prefixed base or ddf.

| Case | Before | After |
|---|---|---|
| Periodic TGV, BGK, tau 0.6, mass drift after 778 steps | 1.8e-5 | -1.5e-9 |
| Periodic TGV, BGK, tau 0.503, mass drift after 25938 steps | 7.6e-4 | -4.6e-7 |
| Uniform periodic box, MRT, 4000 steps | -1.0e-7 | 2.4e-7 |
| Poiseuille BGK, H = 32, L2 error | 1.10e-3 | 7.08e-4 |
| Poiseuille MRT, H = 32, L2 error | 2.33e-3 | 1.12e-3 |
| Poiseuille BGK, wall force error at H = 32 | -1.6e-4 | -1.3e-5 |
| Poiseuille MRT, wall force error at H = 32 | 1.5e-4 | 3.5e-5 |
| Poiseuille L2 order, H 16 to 32, BGK / MRT | 1.37 / 1.07 | 1.98 / 2.05 |

Uniform flow in a closed periodic box, 64x32x32, tau 0.52, 4000 steps,
relative mass drift. Every cell rounds the same way here, so any bias
adds up instead of averaging out.

| Operator | Before | After |
|---|---|---|
| BGK | -2.3e-7 | -4.2e-9 |
| regularized | 9.3e-7 | -4.1e-9 |
| MRT | -1.0e-7 | 2.4e-7 |
| entropic | -2.3e-7 | -4.2e-9 |

MRT still drifts because its inverse transform does not sum exactly to
the input mass in float32. It stays below the 1e-6 check.

Taylor-Green, N = 64, nu_eff / nu - 1. The before runs at tau 0.503
gained 4e-4 to 8e-4 of mass. That lowers u = j / rho and makes the decay
look faster, so the table also gives the fit to the energy of rho u
(second number), which mass drift does not touch. After the change the
two fits agree to 3e-6.

| Operator | tau 0.6 before | tau 0.6 after | tau 0.503 before | tau 0.503 after |
|---|---|---|---|---|
| BGK | -1.33e-4 | -1.30e-4 | -7.2e-4 / -2.24e-3 | -1.63e-3 |
| regularized | -2.59e-4 | -2.51e-4 | -6.3e-4 / -1.39e-3 | -1.02e-3 |
| MRT | -2.73e-4 | -1.62e-4 | -1.70e-3 / -2.93e-3 | -1.72e-3 |
| entropic | -1.35e-4 | -1.29e-4 | -7.2e-4 / -2.24e-3 | -1.63e-3 |

`ddf_shift/tgv_d2q9.py` runs the BGK case on the CPU (D3Q19 BGK on a
z-invariant field is D2Q9 BGK). In float64 it gives -1.12e-4 at tau 0.6
and -8.7e-4 at tau 0.503. In float32 with f - w stored it gives
-1.31e-4 and -1.64e-3, which the GPU now matches. At tau 0.6 the GPU is
within 2e-5 of the exact value. At tau 0.503 float32 storage still
costs about 8e-4 in nu, shifted or not. At N = 128 and tau 0.503 BGK
gives -3.95e-3 after (mass drift -1.8e-6) against -2.78e-3 / -8.9e-3
before (mass drift 3.0e-3), and float64 gives -8.8e-4. This round-off is the likely cause of the MRT
error growing with N that the limits list below calls unexplained.

Sphere Cd, MRT, laptop.

| Re | D | Before | After | Change |
|---|---|---|---|---|
| 20 | 8 | 2.91775 | 2.91763 | -0.004% |
| 20 | 12 | 2.83261 | 2.83238 | -0.008% |
| 100 | 8 | 1.25680 | 1.25672 | -0.006% |
| 100 | 12 | 1.15409 | 1.15391 | -0.016% |

`test_lbm` passes 135 of 135 checks, with four new ones. They cover the
shifted equilibrium, the MRT moments of w, mass in a periodic box, and
the force at rest, which must equal the sum of 2 w_i e_i over the
links. The integration test passes. `run_validation.py` now also checks
the TGV mass drift at tau 0.503 (below 1e-5) and the Poiseuille order
from H = 16 to 32 (above 1.7). Both fail on the old code, the order
check at q = 0.5 where it was run.

## Higher Reynolds number

Results files are in `high_re/`, with the harness output of the quick set
in `high_re/results.md`. 73 of 74 checks pass. The failure is the D = 8
sphere at Re 100, which `sphere/results.md` already lists. Every number
below is from the laptop.

### What limits each operator now

`lbm_validate` already sets tau directly, so the survey needed no flag.
The wind tunnel binary now takes `--tau-floor=TAU` (default 0.52) to
lower its floor. Two tests probe stability. The first is the 3D
Taylor-Green vortex of HiOCFD case C3.5 on 64^3 cells, U0 = 0.05 (new
case `lbm_validate tgv3d`). At Re 1600, 5000 and 50000, tau is 0.50095,
0.50031 and 0.50003. The second is the sphere at D = 8, 40 D/U, at Re
300 (tau 0.504) and Re 1000 (tau 0.5012). The TGV column gives the peak
of -dE/dt against the 512^3 spectral DNS (0.012475 at t = 8.96 with the
same 1 tc difference window), then the time of the peak. At Re 5000 and
50000 only stability counts, since the DNS is for Re 1600. "unstable"
means the stability guard reset cells.

| Operator | Cs | TGV Re 1600 | TGV Re 5000 | TGV Re 50000 | Sphere Re 300 | Sphere Re 1000 |
|---|---|---|---|---|---|---|
| BGK | 0 | unstable | unstable | unstable | 0.837 (sd 0.003) | unstable (148 resets by t 16) |
| regularized | 0 | -4.6%, t 9.1 | stable | unstable | 0.864 (sd 0.003) | 0.675 (sd 0.008) |
| MRT | 0 | -14.3%, t 8.6 | stable | stable | 0.855 (sd 0.002) | unstable (210 resets by t 20) |
| entropic | 0 | -6.0%, t 8.3 | stable | stable | 0.874 (sd 0.003) | unstable (2 resets by t 33) |
| recursive (RR) | 0 | -6.0%, t 8.8 | stable | stable | 0.847 (sd 0.003) | 0.643 (sd 0.004) |
| BGK | 0.1 | -3.9%, t 8.4 | stable | stable | 0.892 (sd 0.003) | 0.735 (sd 0.017) |
| regularized | 0.1 | -8.7%, t 8.6 | stable | stable | 0.909 (sd 0.003) | 0.721 (sd 0.004) |
| MRT | 0.1 | -5.6%, t 8.5 | stable | stable | 0.901 (sd 0.002) | 0.678 (sd 0.011) |
| entropic | 0.1 | -6.3%, t 8.3 | stable | stable | 0.921 (sd 0.003) | 0.793 (sd 0.025) |
| recursive (RR) | 0.1 | -9.6%, t 8.6 | stable | stable | 0.895 (sd 0.003) | 0.705 (sd 0.004) |

The Clift, Grace and Weber values are 0.653 at Re 300 and 0.471 at Re
1000. D = 8 is too coarse to match them (at Re 100 it is 16% high), so
the sphere columns test stability, not accuracy.

- Plain BGK fails all three TGV cases and the sphere at Re 1000.
- MRT without Smagorinsky survives every TGV case but fails the sphere
  at Re 1000. Its TGV dissipation peak is 14% low at Re 1600.
- Entropic survives the TGV but resets 2 cells on the sphere at Re 1000.
- Regularized fails the TGV at Re 50000.
- With Smagorinsky Cs = 0.1, every operator passes every case. The
  production default, MRT with Smagorinsky, runs at tau 0.50003 on the
  TGV and 0.5012 on the sphere, so the 0.52 floor is conservative for it.

Cost per step in a periodic 128x64x64 box, 2000 steps, in MLUPS: BGK
25.4, regularized 24.9, MRT 25.8, entropic 25.0, RR 25.6, MRT with
Smagorinsky 25.6, RR with Smagorinsky 25.7. The Arc iGPU is bandwidth
bound here, so the extra arithmetic of MRT, entropic and RR does not
show.

### Choice of operator

The cumulant operator needs D3Q27, 42% more memory and new versions of
every shader. Recursive regularization (Malaspinas 2015) works on D3Q19.
Coreixas et al. (2020) rank single relaxation time RR among the three
most stable static operators, and Coreixas et al. (2019, App. G and H)
give its D3Q19 form. It is local, so it fits the current collide shader.
Hybrid RR (Jacob et al. 2018) would also need neighbour velocities and a
blending weight from a paper the HAL server would not serve, so it was
not attempted.

RR is `--collision-operator=recursive` (`--op=rr` in `lbm_validate`).
After the usual density, velocity and non-equilibrium stress, the third
and fourth order non-equilibrium Hermite terms come from the stress by
Malaspinas eq 31. Post-collision f is the Hermite series of his eq 37 cut
to the six third order and three fourth order terms D3Q19 supports. Its
equilibrium is Coreixas eq H22-H28.

The first version used the other D3Q19 equilibrium in Coreixas, the
pruned tensor product of eq H15-H21, rebuilt from 19 raw moments (eq
G6). It passed the 2D TGV but blew up on the 3D TGV at Re 1600 by t = 2
and on the sphere at Re 300 by t = 2 (`high_re/survey_rr_product_form.txt`).
A single GPU step matched a numpy version of the same algebra to 2e-9,
so this was not a coding error. Zeroing the third and fourth order
non-equilibrium terms left it unstable (TGV Re 1600: 3356 resets), so
that equilibrium is the cause on this lattice. Coreixas warns that the
two D3Q19 equilibria differ at O(u^2) and should have different
stability. With the Gauss-Hermite form, RR passes every survey case.

On D3Q19 the fourth order Hermite terms leak into the diagonal of the
stress tensor (the lattice has no corner velocities). Off-diagonal stress
relaxes exactly by 1 - omega. The diagonal carries an extra term of order
u^2 times the non-equilibrium stress, 3e-5 in the unit test.

### RR against the existing operators

Same cases as the survey and the DDF shifting section, logged in
`high_re/evidence_runs.txt`.

Taylor-Green, nu_eff / nu - 1. RR gives -5.5e-4, -2.7e-4 and -2.3e-4 at
N = 32, 64 and 128 (tau 0.6). At N = 64 the others give -1.3e-4 (BGK,
entropic), -2.5e-4 (regularized) and -1.6e-4 (MRT). At tau 0.503 RR gives
-1.4e-3, between regularized (-1.0e-3) and BGK or MRT (-1.6e-3, -1.7e-3).

Poiseuille, tau 0.8, L2 error at H = 8, 16, 32. At q = 0.5 RR has the
lowest error of the three, and it converges at second order for every q.

| q | BGK | MRT | RR | RR order 8-16, 16-32 |
|---|---|---|---|---|
| 0.5 | 1.11e-2, 2.80e-3, 7.08e-4 | 1.84e-2, 4.61e-3, 1.12e-3 | 4.28e-3, 1.09e-3, 2.88e-4 | 1.98, 1.92 |
| 0.25 | | | 2.89e-2, 6.64e-3, 1.57e-3 | 2.13, 2.08 |
| 0.75 | | | 1.56e-2, 4.22e-3, 1.18e-3 | 1.89, 1.84 |

The wall force matches the body force within 6.7e-5 in all nine RR runs.

Sphere at low Re, MRT from the DDF shifting section:

| Re | D | MRT | RR | Difference | MRT change from D 8 to 12 |
|---|---|---|---|---|---|
| 20 | 8 | 2.91763 | 2.89951 | -0.62% | |
| 20 | 12 | 2.83238 | 2.82307 | -0.33% | -2.9% |
| 100 | 8 | 1.25672 | 1.25652 | -0.02% | |
| 100 | 12 | 1.15391 | 1.15846 | +0.39% | -8.2% |

The operators differ by far less than the grid error.

Sphere at D = 12 (192x96x96), 60 D/U, mean and spread of the last
quarter. Clift, Grace and Weber: 0.653 and 0.471. On this grid MRT is
6.2% high at Re 100.

| Operator | Re 300 (tau 0.506) | Re 1000 (tau 0.5018) |
|---|---|---|
| RR | 0.721 (+10.4%) | 0.531, sd 0.022 (+12.7%) |
| MRT + Smagorinsky 0.1 (default) | 0.749 (+14.8%) | 0.579, sd 0.040 (+22.8%) |
| regularized | 0.731 (+12.0%) | 0.589, sd 0.033 (+25.0%) |
| entropic | 0.732 (+12.1%) | unstable, 8 resets by t = 21 |

3D Taylor-Green at Re 1600, peak of -dE/dt against the DNS (0.012475)
and E at t = 10 against the DNS (0.07448):

| Operator | 64^3 peak | 64^3 E(10) | 128^3 peak | 128^3 E(10) |
|---|---|---|---|---|
| RR | -6.0% at t 8.8 | -5.1% | -6.6% at t 8.6 | +1.3% |
| MRT | -14.3% at t 8.6 | +3.8% | -1.5% at t 9.0 | +0.9% |

RR handles the coarse grid better, and MRT converges better. RR's peak
stays 6% low at 128^3. The diagonal stress term above is a likely cause,
not checked.

### Reachable Reynolds number

The lowest tau where each setup stayed clean (U = 0.05, Re = 0.15 L / (tau - 0.5) with L in cells):

| Setup | 3D TGV, 64^3 | Sphere, D = 8 | Ahmed 25, 128x64x64 |
|---|---|---|---|
| MRT + Smagorinsky 0.1 (default) | 0.50003 (Re 50000) | 0.50012 (Re 10000) | 0.50115 (Re 5000) |
| RR + Smagorinsky 0.1 | 0.50003 (Re 50000) | 0.50012 (Re 10000) | 0.50115 (Re 5000) |
| RR | 0.5000031 (Re 500000) | 0.5004 (Re 3000), fails at 0.50012 | not run |
| MRT | 0.5000031 (Re 500000) | fails at 0.5012 (Re 1000) | not run |

The Ahmed runs used the wind tunnel binary with `--reynolds=5000
--tau-floor=0.501`, 20 s, rc 0 (no reset or NaN exit). Cd over the last
half was 0.976 (sd 0.235) with MRT and 0.940 (sd 0.266) with RR, from
`high_re/ahmed_runs.txt`. Neither is a converged or validated value.

With tau 0.50012 and the Ahmed body at 0.3 Nx, the stable limit is about
Re 48000 at Nx 128, 96000 at 256 and 190000 at 512. Accuracy now
depends on resolution. At Re 300 and 1000 a 12 cell sphere is
10 to 25% high, against 6% at Re 100. These runs do not show what grid
gives 5% at Re 1000. D = 16 at 60 D/U would take about 30 minutes per
run on the laptop, scaled from the D = 12 timings, and was not run.

### Recommendation

- Tau floor: the default MRT with Smagorinsky stayed stable at tau
  0.50012 on the sphere and 0.50115 on the Ahmed body. A floor of 0.501
  for that setup is supported by these runs. The floor is not changed
  in the code, and one Ahmed run at 256 cells is still worth doing before
  changing it. MRT without Smagorinsky failed at 0.5012 on the sphere
  and passed at 0.504.
- Default operator: keep MRT with Smagorinsky for now. RR is the better
  choice without a subgrid model. It is the only operator that passes
  every survey case without one, and on the D = 12 sphere it is
  closest to the correlation (Re 300 +10% against +15%, Re 1000 +13%
  against +23%). But MRT converges better on the 128^3 TGV, plain RR
  failed at Re 10000 on D = 8, and RR has not been compared on the Ahmed
  body beyond one stability run. RR should become the default only if a
  D = 16 sphere and an Ahmed grid study confirm it.
