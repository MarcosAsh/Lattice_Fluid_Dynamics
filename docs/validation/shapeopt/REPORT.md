# Shape optimiser, checked by the solver (laptop run)

This is a toy run. It shows the pipeline working end to end on a dataset the laptop can make in under an hour. It supports no claim about drag reduction.

## Setup

- Ahmed 25 deg, FFD lattice 3x2x2 with Y symmetry (18 free parameters), displacements up to 15% of the padded box.
- 80 LHS samples plus the baseline, seed 0, fixed in `laptop_config.yaml` before any run.
- 64x32x32, Re 100 on the baseline length, 30 s per run, default MRT with Smagorinsky, `--strict-reynolds`.
- Every shape at the baseline's scale (2.29885) and viscosity (tau 0.5288).
- Binary 88a8ecc, Intel Arc iGPU, Mesa 26.0.8.
- All 81 runs passed the geometry checks. Cd ranges from 3.49 to 4.59, with 95% half-widths of 0.002 to 0.004.
- The body is about 5 cells tall on 64x32x32 and 11 on 128x64x64. The validation report asks for about 16 for drag within a few percent.

## Results

See `verify.md`. The solver gives 4.2906 for the baseline. The single surrogate (seed 0) predicts 0.1289 and the 5-seed ensemble 2.9551.

- single (minimising the mean): the shape barely moved (volume x0.990). The solver measures a drop in Cd of 0.6% on 64x32x32 and 1.3% on 128x64x64, in the predicted direction. The verdict checks direction only, and the match is most likely chance.
- ensemble (minimising mean + 1 std): the surrogate predicted Cd up 0.57. The solver measured Cd down 0.91 on both grids, and CdA/A0 down 0.26 and 0.19. The verdict is wrong direction. The shape is 14% shorter, with 9% less volume and 23% more frontal area. In this dataset CdA/A0 follows volume (correlation 0.96), and 18 of the 81 random shapes have a lower CdA/A0.

## Why the surrogate fails

- Held-out Cd error is 0.107 to 0.169, against 0.154 for predicting the mean.
- The baseline sits alone at the centre of the normalised input space, and every LHS sample lies about 6 units out. Seeds 0 and 1 never fitted it, and the optimisers start there.
- `optimize` now warns about this.
- Possible fixes, none tried: normalise the targets, sample near the baseline, and use many more samples than parameters.

## What a real run needs

- At least 16 cells across the body height, so 192x96x96 or more.
- Several hundred samples, some near the baseline.
- A third grid per shape, so `gci` can give an order and an uncertainty on the change.
- A Reynolds number above 100 if the shapes are meant for vehicles.

Commands are in `run.sh`.
