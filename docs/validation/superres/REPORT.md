# Super-resolution network: is it worth anything?

The 2x super-resolution network is not a result and should be dropped from the article and the JOSS paper. It cannot change drag, because only the particle renderer reads its output. As a picture of the flow it is worse than plain interpolation. On real coarse runs its velocity error is 3.4 times that of trilinear interpolation (pooled 95% CI 3.3 to 3.6). On its own training task it is 6.4 times worse. It is also worse than nearest neighbour, which is what the particles see without it. It costs 97 ms per frame on a 64x32x32 grid against 0.18 ms for a trilinear shader. A smoother display needs only trilinear sampling in `particle.comp`.

## 1. The shipped network

From `train_superres.py`, `superres.c` and `superres_upscale.comp`:

- Input: the 3x3x3 coarse neighbourhood of one cell, 4 channels (ux, uy, uz, rho), 108 values, normalised with a fixed mean and std.
- MLP 108 -> 128 -> 128 -> 128 -> 32 with ReLU, 51,104 parameters.
- Output: the 2x2x2 fine children of the centre cell. The MLP output is added to a skip term and de-normalised.

The weights in `simulation/assets/sr_model.bin` come from a trained model. The first file (c0907c4, April 2026) was a placeholder, Gaussian with std 0.010. The current file (ef764a7, 8 June 2026) has weights with std 0.06 to 0.07 and values up to 0.54, well outside the PyTorch initial bound of 0.088.

Training data:

- No coarse run was ever made. `gen_pairs.py` runs the fine grid and takes the 2x2x2 block mean as the coarse input. The training task is filtered fine to fine.
- `extract_patches` drops every patch that touches a solid cell. The net never saw a wall, and the shader runs it on every cell.
- The weight file has no metadata. The repo does not say which config, how many pairs or what loss produced it.
- The rho channel has mean exactly 1 and std exactly 1, which `train_superres.py` writes only when the channel is constant. ux has std 0.031 against a mean of 0.035 at inlet speed 0.05, which suggests start-up ramp frames were included. Both points are inferences.

The skip term is wrong. Both grids are cell centred over the same box, so the children of coarse cell i sit at -1/4 and +1/4 of a coarse cell. The shader's skip samples +1/4 and +3/4. The training skip samples -3/4 and +3/4. So the shader adds a correction learned against one skip to another. Tests in `tests/test_superres_baselines.py` pin both offsets. A numpy port with the shader skip matches the GPU output to 9.1e-7.

## 2. Effect on drag

There is none. `SR_Upscale` writes its own buffer, and only the particle shader reads it. Drag is momentum exchange on the distributions. VTK output, `--vtk-stats`, `--surface-output` and the streamline tracer read the coarse LBM buffers. Ahmed 25 at 64x32x32 and Re 140 gave Cd 3.51567 +/- 0.00907 with and without `--superres`, and the VTI dumps were byte identical.

## 3. Accuracy against interpolation

Method:

- Cases: sphere D = 8/16 cells at Re 20 and 50, Ahmed 25 at Re 140, car at Re 140. Coarse grid 64x32x32, fine 128x64x64, MRT, `--strict-reynolds`. Only the sphere is a new geometry. Ahmed 25 and the car were in the training set at other Re.
- Time averages come from `--vtk-stats` over every frame after the 300-frame ramp. Coarse runs last 30 s, fine runs 60 s, so both cover the same convective time.
- Real task: the coarse mean field goes in, and the fine mean field is the reference. Filtered task: the 2x2x2 block mean of the fine mean field goes in, as `gen_pairs.py` builds pairs.
- Methods: nearest, trilinear, tricubic (4-point Lagrange), Catmull-Rom, the net as the shader runs it, and the net with its training skip.
- Intervals: 95% block bootstrap over 8x8x8 fine-cell blocks, 2000 resamples, paired across methods.

Commands (from `simulation/`, then the repo root):

    docs/validation/superres/run_cases.sh BUILD RUNS
    uv run --no-sync python ml/cli.py superres-baselines \
        --case sphere_re20 RUNS/sphere_re20/coarse/field_001800.vti RUNS/sphere_re20/fine/field_003600.vti \
        --case sphere_re50 ... --case ahmed25_re140 ... --case car_re140 ... \
        --boot 2000 --slice ahmed25_re140 --out docs/validation/superres --plot

Velocity rel L2 over all fluid, real coarse run as input:

| Case | nearest | trilinear | tricubic | net (shader) | net (training skip) | net / trilinear |
|---|---|---|---|---|---|---|
| sphere Re 20 | 0.015 | 0.008 | 0.007 | 0.040 | 0.025 | 5.27 [5.03, 5.50] |
| sphere Re 50 | 0.019 | 0.012 | 0.012 | 0.042 | 0.024 | 3.56 [3.39, 3.72] |
| Ahmed 25 Re 140 | 0.025 | 0.016 | 0.016 | 0.053 | 0.030 | 3.37 [3.25, 3.48] |
| car Re 140 | 0.025 | 0.017 | 0.017 | 0.049 | 0.031 | 2.90 [2.70, 3.16] |
| pooled | 0.022 | 0.014 | 0.013 | 0.046 | 0.028 | 3.42 [3.28, 3.57] |

The same with the block mean of the fine field as input (the training task):

| Case | nearest | trilinear | tricubic | net (shader) | net (training skip) | net / trilinear |
|---|---|---|---|---|---|---|
| sphere Re 20 | 0.013 | 0.004 | 0.002 | 0.039 | 0.026 | 9.09 [8.13, 10.69] |
| sphere Re 50 | 0.015 | 0.006 | 0.004 | 0.040 | 0.024 | 6.61 [5.98, 7.63] |
| Ahmed 25 Re 140 | 0.020 | 0.008 | 0.005 | 0.050 | 0.027 | 5.96 [5.43, 6.79] |
| car Re 140 | 0.019 | 0.008 | 0.005 | 0.047 | 0.027 | 5.84 [5.32, 6.60] |
| pooled | 0.017 | 0.007 | 0.004 | 0.044 | 0.026 | 6.42 [6.05, 6.88] |

Pooled real task, other metrics (intervals are in `tables.md`):

| Metric | nearest | trilinear | tricubic | net (shader) | net (training skip) |
|---|---|---|---|---|---|
| velocity rel L2, near body | 0.107 | 0.066 | 0.064 | 0.228 | 0.139 |
| velocity rel L2, wake | 0.041 | 0.023 | 0.025 | 0.084 | 0.036 |
| gradient rel L2 | 0.297 | 0.238 | 0.233 | 0.568 | 0.450 |
| vorticity rel L2 | 0.274 | 0.234 | 0.230 | 0.477 | 0.365 |
| disturbance KE rel error | 0.059 | 0.025 | 0.053 | 0.093 | 0.029 |
| mean abs div / U, x1e3 (reference 0.078) | 0.848 | 0.159 | 0.189 | 2.209 | 1.814 |
| solid cells above 0.1 U | 0.030 | 0.069 | 0.032 | 0.221 | 0.231 |

Findings:

- The net loses to trilinear in every case, region and metric, on both tasks. Fixing the skip to match training would still leave it 2.1 times worse on real inputs and 3.8 times worse on the training task.
- On a uniform flow it adds a fixed sub-cell pattern of 0.4% RMS error. The interpolants return the flow exactly.
- Its divergence is 14 times trilinear's.
- It puts more than 0.1 U into 22% of the fine solid cells, against 7% for trilinear.
- On real inputs the interpolants barely differ (tricubic / trilinear = 0.988). The coarse run differs from the block mean of the fine run by 0.8 to 1.5%. That is the discretisation error of the coarse run, and an upsampler that sees only the coarse velocity cannot recover it.

`error_ratio.png` shows the net / trilinear ratios with intervals. `error_slice.png` shows the Ahmed 25 mid-plane.

## 4. Cost

`sr_bench.c` times `SR_Upscale` against a cell-centred trilinear shader (`trilinear2x.comp`). Wall clock with `glFinish`, Mesa Intel Arc (MTL) iGPU:

| Coarse grid | Net | Trilinear | One LBM step on the fine grid |
|---|---|---|---|
| 32x16x16 | 11.1 ms | 0.04 ms | 0.85 ms |
| 64x32x32 | 97 ms | 0.18 ms | 9.3 ms |
| 128x64x64 | 763 ms | 1.45 ms | 100 ms |
| 256x128x128 | 4.0 s | 13 ms | not run |

`main.c` calls it every frame, even before rendering starts. A 64x32x32 Ahmed run took 127.9 s with `--superres` and 14.0 s without.

## 5. What a version worth publishing would need

It would need to super-resolve the distribution functions, as Yang et al. (arXiv 2509.14721) do, so the moments and wall closure stay consistent. It would need training on real coarse runs paired with fine runs, with walls in the data, or solver-in-the-loop training against the fine solution. It would also need a far larger model and dataset than 51k parameters on three bodies, and a test against trilinear and tricubic on held-out cases, as here.

## Not verified

- The training data, config and validation loss of the shipped weights. The repo does not record them.
- Only one grid pair and four cases, all at Re 140 or below. The sphere is the only new geometry.
- GPU timings come from one laptop iGPU.
