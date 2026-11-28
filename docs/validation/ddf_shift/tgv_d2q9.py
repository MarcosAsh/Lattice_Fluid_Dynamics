"""Double precision D2Q9 BGK Taylor-Green run, the exact discrete answer.

D3Q19 BGK on a z-invariant field reduces to D2Q9 BGK, so this gives the
round-off free value of the lbm_validate tgv fit for --op=bgk. Usage:

    uv run --with numpy python tgv_d2q9.py N TAU U0 float64|float32 0|1

The last argument stores f - w (1) or f (0).
"""

import sys

import numpy as np


def main():
    n, tau, u0 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    dt = np.dtype(sys.argv[4])
    shifted = sys.argv[5] == "1"
    c = np.array([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)])
    w = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=dt)[:, None, None]
    cx, cy = c[:, 0, None, None], c[:, 1, None, None]
    nu = (tau - 0.5) / 3
    k = 2 * np.pi / n
    steps = int(1 / (4 * nu * k * k))
    x = (np.arange(n) + 0.5) * k
    a, b = np.meshgrid(x, x, indexing="ij")
    u = u0 * np.sin(a) * np.cos(b)
    v = -u0 * np.cos(a) * np.sin(b)
    p = 0.25 * u0 * u0 * (np.cos(2 * a) + np.cos(2 * b))

    def feq(rho, drho, ux, uy):
        eu = cx * ux + cy * uy
        u2 = ux * ux + uy * uy
        if shifted:
            return w * rho * (4.5 * eu * eu - 1.5 * u2 + 3 * eu) + w * drho
        return w * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * u2)

    f = feq(1 + 3 * p, 3 * p, u, v).astype(dt)
    om = dt.type(1 / tau)
    every = max(1, steps // 40)
    ts, ln_e = [], []
    for s in range(1, steps + 1):
        drho = f.sum(0) if shifted else f.sum(0) - 1
        rho = drho + 1
        ux = (cx * f).sum(0) / rho
        uy = (cy * f).sum(0) / rho
        if s % every == 0 and len(ts) < 64:
            e = (ux.astype(float) ** 2 + uy.astype(float) ** 2) * rho.astype(float) ** 2
            ts.append(s - 1)
            ln_e.append(np.log(e.sum()))
        f = f - om * (f - feq(rho, drho, ux, uy).astype(dt))
        for i in range(9):
            f[i] = np.roll(f[i], shift=(c[i, 0], c[i, 1]), axis=(0, 1))
    a0 = len(ts) // 2
    slope = np.polyfit(np.array(ts[a0:], float), np.array(ln_e[a0:]), 1)[0]
    print(f"nu_err_j {(-slope / (4 * k * k) - nu) / nu:.4e}")


if __name__ == "__main__":
    main()
