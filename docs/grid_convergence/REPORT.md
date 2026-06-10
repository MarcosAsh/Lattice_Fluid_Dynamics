# Grid convergence study: Ahmed body 25 deg (#155)

Three passes run on Modal A100, 2026-06-10 (passes 1-2 on commit
6a7d5f3, pass 3 on 78e0e71 with force averaging).
Raw Cd series in `raw_*.json`, summary tables in `results_*.csv`,
figures in `convergence_ahmed25_re144*.png`.

## Pass 1: issue spec as written (30 s per grid, auto Re)

| Grid        | Cells | Cd (tail mean +/- std) | relStd | Re_eff | Flow-throughs |
|-------------|-------|------------------------|--------|--------|---------------|
| 32x16x16    | 8.2K  | 5.50 +/- 1.31          | 24%    | 72     | 34            |
| 64x32x32    | 65K   | 4.62 +/- 3.31          | 72%    | 144    | 18            |
| 128x64x64   | 524K  | 5.91 +/- 3.88          | 66%    | 200    | 9.7           |
| 256x128x128 | 4.2M  | 2.75 +/- 0.53          | 19%    | 200    | 5.6           |

Two protocol flaws make this pass unusable for extrapolation:

1. Re is derived from wind speed and body size in cells, so each grid
   simulated a different Reynolds number (72-200). Cd is strongly
   Re-dependent in this range.
2. Fixed wall duration means the fine grids got far fewer
   flow-throughs than the coarse ones.

## Pass 2: Re pinned at 144, duration scaled per grid

60 s / 90 s / 120 s for the three finest grids (~16-32 flow-throughs
each); the 32 grid was dropped (body is ~8 cells, unusable).

| Grid        | Cd (last-quarter mean) | relStd | Flow-throughs |
|-------------|------------------------|--------|---------------|
| 64x32x32    | 3.56 +/- 2.39          | 67%    | 32            |
| 128x64x64   | 5.44 +/- 3.62          | 66%    | 24            |
| 256x128x128 | 3.33 +/- 0.13          | 3.8%   | 16            |

## Pass 3: window-averaged force sampling

Same protocol as pass 2, with the solver change from 78e0e71: each Cd
sample is the mean boundary force over every LBM step in its sampling
window (~100 steps) instead of one instantaneous snapshot.

| Grid        | Cd (last-quarter mean) | relStd | lag-1 autocorr |
|-------------|------------------------|--------|----------------|
| 64x32x32    | 3.64 +/- 1.72          | 47%    | -0.95          |
| 128x64x64   | 5.00 +/- 3.45          | 69%    | ~0             |
| 256x128x128 | 3.33 +/- 0.11          | 3.4%   | -0.13          |

What the averaging revealed:

- The coarse-grid scatter is **not sampling noise**. If it were
  uncorrelated snapshot noise, a 100-step mean would have cut it
  ~10x; it barely moved. The fluctuation is physical (or numerical)
  unsteadiness with timescales at or beyond the sampling window.
- The 64 grid's averaged Cd alternates high/low between consecutive
  windows (lag-1 autocorrelation -0.95): a coherent oscillation with
  a period of ~2 sampling windows (~200 steps), aliased by the
  sampling. Worth a spectral look before trusting any Cd from this
  resolution.
- The 256-grid mean is **rock solid across passes and binaries**:
  3.3328 (snapshot sampling) vs 3.3333 (averaged sampling). Cd = 3.33
  +/- 0.11 at Re = 144 is the reference value this study produces.
- The relStd < 1% target is not reachable by better sampling alone at
  these grids; the residual 3-4% on the 256 grid is real wake
  unsteadiness and would need longer time-averaging windows (longer
  runs), not denser sampling.

## Findings

- **256x128x128 is the first resolution that produces a converged,
  low-noise Cd**: 3.33 +/- 0.13, with quarter-by-quarter variance
  still shrinking at the end of the run. This is the working
  reference value for Re=144.
- **Cd is not yet monotone in grid spacing**, so Richardson
  extrapolation is not valid on this data. The 64 and 128 grids have
  instantaneous Cd swinging between ~0 and ~15 (sample-to-sample sign
  flips around the mean), i.e. the momentum-exchange force on an
  under-resolved staircase body is dominated by discretization noise,
  not physical unsteadiness.
- **The 128 grid reads high in both passes** (5.4-5.9 vs 3.3-3.8 for
  its neighbours). Untangling that needs either longer averaging or
  force-accumulation in the solver (averaging the force over substeps
  before sampling), which would also be the path to the issue's
  relStd < 1% target.
- **Comparison with published data is Re-limited.** Ahmed et al.
  (1984) measured Cd = 0.285 at Re = 4.29M; at the Re ~ 144 reachable
  by the 64 grid, bluff-body Cd is genuinely an order of magnitude
  higher, so agreement with 0.285 is not expected at any of these
  resolutions. Matching published values requires running finer grids
  at their own (higher) Re cap and extrapolating in Re as well -- see
  #156.

## Recommendations

1. Use 256x128x128 (or finer) for any training-data generation where
   Cd accuracy matters (#154, #150); 128x64x64 Cd values carry ~60%
   noise.
2. Add time-averaged force accumulation to the force shader before
   re-attempting Richardson extrapolation and the relStd < 1% target.
3. Re-run this study at a fixed higher Re (e.g. 576, the 256-grid
   cap) on 128/256/512 grids once 512^3 fits in GPU memory via the
   existing slab chunking.

## Reproduction

```bash
modal run simulation/grid_convergence.py::sweep \
  --grids "64x32x32:60,128x64x64:90,256x128x128:120" \
  --reynolds 144 --tag re144
```
