| Case | Quantity | Measured | Reference | Tolerance | Result |
|---|---|---|---|---|---|
| uniform open tunnel | max |u - U| | 1.48e-05 | 0 | 1e-4 | pass |
| uniform periodic box | relative mass drift | -2.42e-07 | 0 | 1e-6 | pass |
| TGV bgk n=64 tau=0.6 | nu_eff/nu - 1 | -1.08e-04 | 0 | 1e-3 | pass |
| TGV reg n=64 tau=0.6 | nu_eff/nu - 1 | -2.36e-04 | 0 | 1e-3 | pass |
| TGV mrt n=64 tau=0.6 | nu_eff/nu - 1 | -2.97e-04 | 0 | 1e-3 | pass |
| TGV ent n=64 tau=0.6 | nu_eff/nu - 1 | -1.09e-04 | 0 | 1e-3 | pass |
| TGV mrt n=64 tau=0.503 | nu_eff/nu - 1 | -2.62e-03 | 0 | 1e-2 | pass |
| TGV ent n=64 tau=0.503 | nu_eff/nu - 1 | 7.68e-04 | 0 | 1e-2 | pass |
| TGV Smagorinsky Cs=0.1 | nu_t measured / predicted | 0.999 | 1 | 0.05 | pass |
| TGV Smagorinsky Cs=0.2 | nu_t measured / predicted | 0.995 | 1 | 0.05 | pass |
| TGV xz, 4 slabs vs 1 | nu_eff and L2 identical | 3.332983e-02 | 3.332983e-02 | bitwise | pass |
| Poiseuille bgk q=0.5 H=8 | wall force / body force - 1 | -9.3e-06 | 0 | 1e-3 | pass |
| Poiseuille bgk q=0.5 H=16 | wall force / body force - 1 | -8.6e-06 | 0 | 1e-3 | pass |
| Poiseuille bgk q=0.5 H=32 | wall force / body force - 1 | 2.9e-05 | 0 | 1e-3 | pass |
| Poiseuille bgk q=0.5 | L2 order, H 8 to 16 | 1.98 | 2 | > 1.7 | pass |
| Poiseuille bgk q=0.25 H=8 | wall force / body force - 1 | 4.2e-06 | 0 | 1e-3 | pass |
| Poiseuille bgk q=0.25 H=16 | wall force / body force - 1 | -9.8e-06 | 0 | 1e-3 | pass |
| Poiseuille bgk q=0.25 H=32 | wall force / body force - 1 | -6.7e-06 | 0 | 1e-3 | pass |
| Poiseuille bgk q=0.25 | L2 order, H 8 to 16 | 2.13 | 2 | > 1.7 | pass |
| Poiseuille bgk q=0.75 H=8 | wall force / body force - 1 | 9.2e-07 | 0 | 1e-3 | pass |
| Poiseuille bgk q=0.75 H=16 | wall force / body force - 1 | 3.8e-06 | 0 | 1e-3 | pass |
| Poiseuille bgk q=0.75 H=32 | wall force / body force - 1 | 3.2e-05 | 0 | 1e-3 | pass |
| Poiseuille bgk q=0.75 | L2 order, H 8 to 16 | 1.87 | 2 | > 1.7 | pass |
| Poiseuille mrt q=0.5 H=8 | wall force / body force - 1 | -6.9e-08 | 0 | 1e-3 | pass |
| Poiseuille mrt q=0.5 H=16 | wall force / body force - 1 | 1.0e-05 | 0 | 1e-3 | pass |
| Poiseuille mrt q=0.5 H=32 | wall force / body force - 1 | 4.7e-06 | 0 | 1e-3 | pass |
| Poiseuille mrt q=0.5 | L2 order, H 8 to 16 | 1.99 | 2 | > 1.7 | pass |
| Poiseuille mrt q=0.25 H=8 | wall force / body force - 1 | 1.7e-06 | 0 | 1e-3 | pass |
| Poiseuille mrt q=0.25 H=16 | wall force / body force - 1 | -8.6e-06 | 0 | 1e-3 | pass |
| Poiseuille mrt q=0.25 H=32 | wall force / body force - 1 | 4.6e-05 | 0 | 1e-3 | pass |
| Poiseuille mrt q=0.25 | L2 order, H 8 to 16 | 2.13 | 2 | > 1.7 | pass |
| Poiseuille mrt q=0.75 H=8 | wall force / body force - 1 | 1.3e-05 | 0 | 1e-3 | pass |
| Poiseuille mrt q=0.75 H=16 | wall force / body force - 1 | 5.1e-07 | 0 | 1e-3 | pass |
| Poiseuille mrt q=0.75 H=32 | wall force / body force - 1 | 1.8e-05 | 0 | 1e-3 | pass |
| Poiseuille mrt q=0.75 | L2 order, H 8 to 16 | 1.89 | 2 | > 1.7 | pass |
| mesh vs analytic sphere D=8 | links stuck at q=0.5 | 0 | 0 | 0 | pass |
| mesh vs analytic sphere D=8 | mean |q_mesh - q_exact| | 0.0010 | 0 | 0.01 | pass |
| mesh frontal area D=8 | A_mesh / (pi D^2/4) - 1 | 0.0034 | 0 | 0.01 | pass |
| mesh vs analytic sphere D=16 | links stuck at q=0.5 | 0 | 0 | 0 | pass |
| mesh vs analytic sphere D=16 | mean |q_mesh - q_exact| | 0.0022 | 0 | 0.01 | pass |
| mesh frontal area D=16 | A_mesh / (pi D^2/4) - 1 | 0.0019 | 0 | 0.01 | pass |
| Ahmed 25 128x64x64 | clean exit, Cd sample spread | rc=0 sd/mean=0.103 | rc=0 | < 0.05 | FAIL |
| Ahmed 25 256x128x128 | clean exit, Cd sample spread | rc=0 sd/mean=0.164 | rc=0 | < 0.05 | FAIL |
| Ahmed 35 128x64x64 | clean exit, Cd sample spread | rc=0 sd/mean=0.102 | rc=0 | < 0.05 | FAIL |
| Ahmed 35 256x128x128 | clean exit, Cd sample spread | rc=0 sd/mean=0.163 | rc=0 | < 0.05 | FAIL |
