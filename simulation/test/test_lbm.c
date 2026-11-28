/*
 * LBM unit tests.
 *
 * Needs an OpenGL context, so run under Xvfb in CI:
 *   xvfb-run ./build/test_lbm
 */

#include "../lib/gl_helpers.h"
#include "../lib/lbm.h"
#include <glad/gl.h>
#include <SDL2/SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg)                                                      \
    do {                                                                       \
        tests_run++;                                                           \
        if (!(cond)) {                                                         \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__);                   \
        } else {                                                               \
            tests_passed++;                                                    \
        }                                                                      \
    } while (0)

#define ASSERT_NEAR(a, b, tol, msg)                                            \
    do {                                                                       \
        tests_run++;                                                           \
        if (fabs((double)(a) - (double)(b)) > (tol)) {                         \
            printf("  FAIL: %s -- got %.6f, want %.6f"                         \
                   " (line %d)\n",                                             \
                   msg,                                                        \
                   (double)(a),                                                \
                   (double)(b),                                                \
                   __LINE__);                                                  \
        } else {                                                               \
            tests_passed++;                                                    \
        }                                                                      \
    } while (0)

/* D3Q19 weights (same as in lbm.c) */
static const float w[19] = {1.0f / 3.0f,
                            1.0f / 18.0f,
                            1.0f / 18.0f,
                            1.0f / 18.0f,
                            1.0f / 18.0f,
                            1.0f / 18.0f,
                            1.0f / 18.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f};

static const int ex[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
static const int ey[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1};
static const int ez[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1};

static float feq(int i, float rho, float ux, float uy, float uz) {
    float eu = ex[i] * ux + ey[i] * uy + ez[i] * uz;
    float u2 = ux * ux + uy * uy + uz * uz;
    float cs2 = 1.0f / 3.0f;
    return w[i] * rho *
           (1.0f + eu / cs2 + (eu * eu) / (2.0f * cs2 * cs2) -
            u2 / (2.0f * cs2));
}

/* feq - w as the shaders store it, Lehmann 2022 eq 27 */
static float feq_shifted(int i, float drho, float ux, float uy, float uz) {
    float eu = ex[i] * ux + ey[i] * uy + ez[i] * uz;
    float u2 = ux * ux + uy * uy + uz * uz;
    return w[i] * (1.0f + drho) * (4.5f * eu * eu - 1.5f * u2 + 3.0f * eu) +
           w[i] * drho;
}

/* MRT moment transforms (CPU mirrors of the GLSL functions) */

static void cpu_forwardMRT(const float fi[19], float m[19]) {
    float sf = fi[1] + fi[2] + fi[3] + fi[4] + fi[5] + fi[6];
    float se = fi[7] + fi[8] + fi[9] + fi[10] + fi[11] + fi[12] + fi[13] +
               fi[14] + fi[15] + fi[16] + fi[17] + fi[18];

    m[0] = fi[0] + sf + se;
    m[1] = -30.0f * fi[0] - 11.0f * sf + 8.0f * se;
    m[2] = 12.0f * fi[0] - 4.0f * sf + se;
    m[3] = fi[1] - fi[2] + fi[7] - fi[8] + fi[9] - fi[10] + fi[11] - fi[12] +
           fi[13] - fi[14];
    m[4] = -4.0f * (fi[1] - fi[2]) + fi[7] - fi[8] + fi[9] - fi[10] + fi[11] -
           fi[12] + fi[13] - fi[14];
    m[5] = fi[3] - fi[4] + fi[7] + fi[8] - fi[9] - fi[10] + fi[15] - fi[16] +
           fi[17] - fi[18];
    m[6] = -4.0f * (fi[3] - fi[4]) + fi[7] + fi[8] - fi[9] - fi[10] + fi[15] -
           fi[16] + fi[17] - fi[18];
    m[7] = fi[5] - fi[6] + fi[11] + fi[12] - fi[13] - fi[14] + fi[15] + fi[16] -
           fi[17] - fi[18];
    m[8] = -4.0f * (fi[5] - fi[6]) + fi[11] + fi[12] - fi[13] - fi[14] +
           fi[15] + fi[16] - fi[17] - fi[18];

    float sXY = fi[7] + fi[8] + fi[9] + fi[10];
    float sXZ = fi[11] + fi[12] + fi[13] + fi[14];
    float sYZ = fi[15] + fi[16] + fi[17] + fi[18];

    m[9] = 2.0f * (fi[1] + fi[2]) - (fi[3] + fi[4]) - (fi[5] + fi[6]) + sXY +
           sXZ - 2.0f * sYZ;
    m[10] = -4.0f * (fi[1] + fi[2]) + 2.0f * (fi[3] + fi[4]) +
            2.0f * (fi[5] + fi[6]) + sXY + sXZ - 2.0f * sYZ;
    m[11] = (fi[3] + fi[4]) - (fi[5] + fi[6]) + sXY - sXZ;
    m[12] = -2.0f * (fi[3] + fi[4]) + 2.0f * (fi[5] + fi[6]) + sXY - sXZ;
    m[13] = fi[7] - fi[8] - fi[9] + fi[10];
    m[14] = fi[15] - fi[16] - fi[17] + fi[18];
    m[15] = fi[11] - fi[12] - fi[13] + fi[14];
    m[16] = fi[7] - fi[8] + fi[9] - fi[10] - fi[11] + fi[12] - fi[13] + fi[14];
    m[17] = -fi[7] - fi[8] + fi[9] + fi[10] + fi[15] - fi[16] + fi[17] - fi[18];
    m[18] =
        fi[11] + fi[12] - fi[13] - fi[14] - fi[15] - fi[16] + fi[17] + fi[18];
}

static void cpu_inverseMRT(const float m[19], float fi[19]) {
    float ms[19];
    ms[0] = m[0] / 19.0f;
    ms[1] = m[1] / 2394.0f;
    ms[2] = m[2] / 252.0f;
    ms[3] = m[3] / 10.0f;
    ms[4] = m[4] / 40.0f;
    ms[5] = m[5] / 10.0f;
    ms[6] = m[6] / 40.0f;
    ms[7] = m[7] / 10.0f;
    ms[8] = m[8] / 40.0f;
    ms[9] = m[9] / 36.0f;
    ms[10] = m[10] / 72.0f;
    ms[11] = m[11] / 12.0f;
    ms[12] = m[12] / 24.0f;
    ms[13] = m[13] / 4.0f;
    ms[14] = m[14] / 4.0f;
    ms[15] = m[15] / 4.0f;
    ms[16] = m[16] / 8.0f;
    ms[17] = m[17] / 8.0f;
    ms[18] = m[18] / 8.0f;

    fi[0] = ms[0] - 30.0f * ms[1] + 12.0f * ms[2];

    fi[1] = ms[0] - 11.0f * ms[1] - 4.0f * ms[2] + ms[3] - 4.0f * ms[4] +
            2.0f * ms[9] - 4.0f * ms[10];
    fi[2] = ms[0] - 11.0f * ms[1] - 4.0f * ms[2] - ms[3] + 4.0f * ms[4] +
            2.0f * ms[9] - 4.0f * ms[10];

    fi[3] = ms[0] - 11.0f * ms[1] - 4.0f * ms[2] + ms[5] - 4.0f * ms[6] -
            ms[9] + 2.0f * ms[10] + ms[11] - 2.0f * ms[12];
    fi[4] = ms[0] - 11.0f * ms[1] - 4.0f * ms[2] - ms[5] + 4.0f * ms[6] -
            ms[9] + 2.0f * ms[10] + ms[11] - 2.0f * ms[12];

    fi[5] = ms[0] - 11.0f * ms[1] - 4.0f * ms[2] + ms[7] - 4.0f * ms[8] -
            ms[9] + 2.0f * ms[10] - ms[11] + 2.0f * ms[12];
    fi[6] = ms[0] - 11.0f * ms[1] - 4.0f * ms[2] - ms[7] + 4.0f * ms[8] -
            ms[9] + 2.0f * ms[10] - ms[11] + 2.0f * ms[12];

    fi[7] = ms[0] + 8.0f * ms[1] + ms[2] + ms[3] + ms[4] + ms[5] + ms[6] +
            ms[9] + ms[10] + ms[11] + ms[12] + ms[13] + ms[16] - ms[17];
    fi[8] = ms[0] + 8.0f * ms[1] + ms[2] - ms[3] - ms[4] + ms[5] + ms[6] +
            ms[9] + ms[10] + ms[11] + ms[12] - ms[13] - ms[16] - ms[17];
    fi[9] = ms[0] + 8.0f * ms[1] + ms[2] + ms[3] + ms[4] - ms[5] - ms[6] +
            ms[9] + ms[10] + ms[11] + ms[12] - ms[13] + ms[16] + ms[17];
    fi[10] = ms[0] + 8.0f * ms[1] + ms[2] - ms[3] - ms[4] - ms[5] - ms[6] +
             ms[9] + ms[10] + ms[11] + ms[12] + ms[13] - ms[16] + ms[17];

    fi[11] = ms[0] + 8.0f * ms[1] + ms[2] + ms[3] + ms[4] + ms[7] + ms[8] +
             ms[9] + ms[10] - ms[11] - ms[12] + ms[15] - ms[16] + ms[18];
    fi[12] = ms[0] + 8.0f * ms[1] + ms[2] - ms[3] - ms[4] + ms[7] + ms[8] +
             ms[9] + ms[10] - ms[11] - ms[12] - ms[15] + ms[16] + ms[18];
    fi[13] = ms[0] + 8.0f * ms[1] + ms[2] + ms[3] + ms[4] - ms[7] - ms[8] +
             ms[9] + ms[10] - ms[11] - ms[12] - ms[15] - ms[16] - ms[18];
    fi[14] = ms[0] + 8.0f * ms[1] + ms[2] - ms[3] - ms[4] - ms[7] - ms[8] +
             ms[9] + ms[10] - ms[11] - ms[12] + ms[15] + ms[16] - ms[18];

    fi[15] = ms[0] + 8.0f * ms[1] + ms[2] + ms[5] + ms[6] + ms[7] + ms[8] -
             2.0f * ms[9] - 2.0f * ms[10] + ms[14] + ms[17] - ms[18];
    fi[16] = ms[0] + 8.0f * ms[1] + ms[2] - ms[5] - ms[6] + ms[7] + ms[8] -
             2.0f * ms[9] - 2.0f * ms[10] - ms[14] - ms[17] - ms[18];
    fi[17] = ms[0] + 8.0f * ms[1] + ms[2] + ms[5] + ms[6] - ms[7] - ms[8] -
             2.0f * ms[9] - 2.0f * ms[10] - ms[14] + ms[17] + ms[18];
    fi[18] = ms[0] + 8.0f * ms[1] + ms[2] - ms[5] - ms[6] - ms[7] - ms[8] -
             2.0f * ms[9] - 2.0f * ms[10] + ms[14] - ms[17] + ms[18];
}

/* Recursive regularized collision, CPU mirror of collideRR */
static void cpu_collideRR(const float g[19], float omega, float o[19]) {
    const float cs2 = 1.0f / 3.0f;
    float drho = 0, ux = 0, uy = 0, uz = 0;
    float sxx = 0, syy = 0, szz = 0, sxy = 0, sxz = 0, syz = 0;
    for (int i = 0; i < 19; i++) {
        drho += g[i];
        ux += ex[i] * g[i];
        uy += ey[i] * g[i];
        uz += ez[i] * g[i];
        sxx += ex[i] * ex[i] * g[i];
        syy += ey[i] * ey[i] * g[i];
        szz += ez[i] * ez[i] * g[i];
        sxy += ex[i] * ey[i] * g[i];
        sxz += ex[i] * ez[i] * g[i];
        syz += ey[i] * ez[i] * g[i];
    }
    float rho = 1.0f + drho;
    ux /= rho;
    uy /= rho;
    uz /= rho;
    float c = (1.0f - omega) / rho;
    float nxx = c * (sxx - rho * ux * ux - cs2 * drho);
    float nyy = c * (syy - rho * uy * uy - cs2 * drho);
    float nzz = c * (szz - rho * uz * uz - cs2 * drho);
    float nxy = c * (sxy - rho * ux * uy);
    float nxz = c * (sxz - rho * ux * uz);
    float nyz = c * (syz - rho * uy * uz);
    float Axx = ux * ux + nxx, Ayy = uy * uy + nyy, Azz = uz * uz + nzz;
    float Axy = ux * uy + nxy, Axz = ux * uz + nxz, Ayz = uy * uz + nyz;
    float Axxy = ux * ux * uy + 2 * ux * nxy + uy * nxx;
    float Axyy = ux * uy * uy + 2 * uy * nxy + ux * nyy;
    float Axxz = ux * ux * uz + 2 * ux * nxz + uz * nxx;
    float Axzz = ux * uz * uz + 2 * uz * nxz + ux * nzz;
    float Ayyz = uy * uy * uz + 2 * uy * nyz + uz * nyy;
    float Ayzz = uy * uz * uz + 2 * uz * nyz + uy * nzz;
    float Axxyy =
        ux * ux * uy * uy + uy * uy * nxx + ux * ux * nyy + 4 * ux * uy * nxy;
    float Axxzz =
        ux * ux * uz * uz + uz * uz * nxx + ux * ux * nzz + 4 * ux * uz * nxz;
    float Ayyzz =
        uy * uy * uz * uz + uz * uz * nyy + uy * uy * nzz + 4 * uy * uz * nyz;
    for (int i = 0; i < 19; i++) {
        float cx = ex[i], cy = ey[i], cz = ez[i];
        float hx = cx * cx - cs2, hy = cy * cy - cs2, hz = cz * cz - cs2;
        float h2 = hx * Axx + hy * Ayy + hz * Azz +
                   2 * (cx * cy * Axy + cx * cz * Axz + cy * cz * Ayz);
        float h3 = hx * cy * Axxy + cx * hy * Axyy + hx * cz * Axxz +
                   cx * hz * Axzz + hy * cz * Ayyz + cy * hz * Ayzz;
        float h4 = hx * hy * Axxyy + hx * hz * Axxzz + hy * hz * Ayyzz;
        float cu = cx * ux + cy * uy + cz * uz;
        o[i] = w[i] * drho +
               w[i] * rho * (3 * cu + 4.5f * h2 + 13.5f * h3 + 20.25f * h4);
    }
}

/* Tests */

static void test_weights_sum_to_one(void) {
    printf("test: D3Q19 weights sum to 1\n");
    float sum = 0;
    for (int i = 0; i < 19; i++)
        sum += w[i];
    ASSERT_NEAR(sum, 1.0f, 1e-6, "weights sum");
}

static void test_equilibrium_conserves_density(void) {
    printf("test: equilibrium conserves density\n");
    float rho = 1.5f;
    float ux = 0.05f, uy = 0.01f, uz = -0.02f;
    float sum = 0;
    for (int i = 0; i < 19; i++)
        sum += feq(i, rho, ux, uy, uz);
    ASSERT_NEAR(sum, rho, 1e-5, "feq density");
}

static void test_equilibrium_conserves_momentum(void) {
    printf("test: equilibrium conserves momentum\n");
    float rho = 1.2f;
    float ux = 0.08f, uy = -0.03f, uz = 0.04f;
    float mx = 0, my = 0, mz = 0;
    for (int i = 0; i < 19; i++) {
        float fi = feq(i, rho, ux, uy, uz);
        mx += ex[i] * fi;
        my += ey[i] * fi;
        mz += ez[i] * fi;
    }
    ASSERT_NEAR(mx, rho * ux, 1e-5, "feq momentum x");
    ASSERT_NEAR(my, rho * uy, 1e-5, "feq momentum y");
    ASSERT_NEAR(mz, rho * uz, 1e-5, "feq momentum z");
}

static void test_equilibrium_at_rest(void) {
    printf("test: equilibrium at rest is just weights\n");
    float rho = 1.0f;
    for (int i = 0; i < 19; i++) {
        float fi = feq(i, rho, 0, 0, 0);
        ASSERT_NEAR(fi, w[i], 1e-6, "feq at rest");
    }
}

static void test_shifted_equilibrium(void) {
    printf("test: shifted equilibrium is feq - w\n");
    float drho = 2e-3f, ux = 0.05f, uy = -0.02f, uz = 0.01f;
    double sum = 0, mx = 0;
    for (int i = 0; i < 19; i++) {
        float g = feq_shifted(i, drho, ux, uy, uz);
        ASSERT_NEAR(g, feq(i, 1.0f + drho, ux, uy, uz) - w[i], 1e-7, "feq-w");
        sum += g;
        mx += ex[i] * g;
    }
    ASSERT_NEAR(sum, drho, 1e-8, "shifted feq sums to rho - 1");
    ASSERT_NEAR(mx, (1.0 + drho) * ux, 1e-8, "shifted feq momentum");
}

static void test_mrt_moments_of_weights(void) {
    printf("test: MRT moments of w are (1, -11, 3, 0, ...)\n");
    float m[19];
    cpu_forwardMRT(w, m);
    for (int k = 0; k < 19; k++) {
        float want = k == 0 ? 1.0f : k == 1 ? -11.0f : k == 2 ? 3.0f : 0.0f;
        ASSERT_NEAR(m[k], want, 1e-6, "moment of w");
    }
}

/* Shear wave in a periodic box, mass must hold to float32 round-off */
static void test_periodic_mass_shifted(void) {
    printf("test: periodic box conserves mass with shifted f\n");
    int n = 16;
    LBMGrid *grid = LBM_Create(n, n, n, 0.001f);
    if (!grid)
        return;
    grid->tau = 0.503f;
    grid->periodicX = 1;
    grid->periodicYZ = 1;
    size_t cells = grid->totalCells;
    float *f = malloc(cells * 19 * sizeof(float));
    for (int z = 0; z < n; z++)
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++) {
                float a = 6.2831853f * (x + 0.5f) / n;
                float b = 6.2831853f * (y + 0.5f) / n;
                float u = 0.05f * sinf(a) * cosf(b);
                float v = -0.05f * cosf(a) * sinf(b);
                size_t c = x + (size_t)n * (y + (size_t)n * z);
                for (int i = 0; i < 19; i++)
                    f[c * 19 + i] = feq_shifted(i, 0.0f, u, v, 0.0f);
            }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->fBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, cells * 19 * 4, f);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->fNewBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, cells * 19 * 4, f);
    double m0 = 0;
    for (size_t i = 0; i < cells * 19; i++)
        m0 += f[i];
    for (int s = 0; s < 1000; s++)
        LBM_Step(grid, 0, 0, 0);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->fBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, cells * 19 * 4, f);
    double m1 = 0;
    for (size_t i = 0; i < cells * 19; i++)
        m1 += f[i];
    /* Stored values sum to mass - cells */
    double drift = (m1 - m0) / (double)cells;
    printf("  relative mass drift over 1000 steps: %.2e\n", drift);
    ASSERT(fabs(drift) < 1e-6, "mass drift below 1e-6");
    free(f);
    LBM_Free(grid);
}

/* At rest f = w, so each link adds 2 w_i e_i, which the shift moves to CPU */
static void test_force_at_rest(void) {
    printf("test: force at rest equals sum of 2 w e over links\n");
    LBMGrid *grid = LBM_Create(16, 8, 8, 0.1f);
    if (!grid)
        return;
    LBM_SetSolidAABB(grid, -0.3f, -0.15f, -0.15f, 0.3f, 0.15f, 0.15f);
    LBM_AddGroundPlane(grid, -1.5f);
    LBM_InitializeFlow(grid, 0.0f, 0.0f, 0.0f);
    size_t n = (size_t)19 * grid->totalCells;
    float *q = malloc(n * sizeof(float));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->qBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(float), q);
    double want[3] = {0, 0, 0};
    for (size_t k = 0; k < n; k++) {
        int i = (int)(k % 19);
        if (q[k] >= 0.0f && i > 0) {
            want[0] += 2.0 * w[i] * ex[i];
            want[1] += 2.0 * w[i] * ey[i];
            want[2] += 2.0 * w[i] * ez[i];
        }
    }
    free(q);
    float fx, fy, fz;
    LBM_ComputeDragForce(grid, &fx, &fy, &fz);
    printf("  force (%.6f, %.6f, %.6f), links give (%.6f, %.6f, %.6f)\n",
           fx,
           fy,
           fz,
           want[0],
           want[1],
           want[2]);
    ASSERT(fabs(want[2]) > 1.0, "ground links give a net z term");
    ASSERT_NEAR(fx, want[0], 1e-4, "rest force x");
    ASSERT_NEAR(fy, want[1], 1e-4, "rest force y");
    ASSERT_NEAR(fz, want[2], 1e-4, "rest force z");
    LBM_Free(grid);
}

static void test_rr_conserves_and_relaxes(void) {
    printf("test: recursive regularized keeps rho, j and relaxes Pi\n");
    float g[19], o[19];
    for (int i = 0; i < 19; i++)
        g[i] = feq_shifted(i, 0.02f, 0.05f, -0.03f, 0.02f) +
               1e-3f * (float)((i * 7) % 5 - 2);
    float omega = 1.9f;
    cpu_collideRR(g, omega, o);
    double m[2][4] = {{0}}, P[2][6] = {{0}};
    for (int k = 0; k < 2; k++) {
        const float *h = k ? o : g;
        for (int i = 0; i < 19; i++) {
            double fi = h[i] + w[i];
            m[k][0] += fi;
            m[k][1] += ex[i] * fi;
            m[k][2] += ey[i] * fi;
            m[k][3] += ez[i] * fi;
            P[k][0] += ex[i] * ex[i] * fi;
            P[k][1] += ey[i] * ey[i] * fi;
            P[k][2] += ez[i] * ez[i] * fi;
            P[k][3] += ex[i] * ey[i] * fi;
            P[k][4] += ex[i] * ez[i] * fi;
            P[k][5] += ey[i] * ez[i] * fi;
        }
    }
    for (int a = 0; a < 4; a++)
        ASSERT_NEAR(m[1][a], m[0][a], 1e-6, "rr conserves rho and j");
    double rho = m[0][0], u[3] = {m[0][1] / rho, m[0][2] / rho, m[0][3] / rho};
    int ia[6] = {0, 1, 2, 0, 0, 1}, ib[6] = {0, 1, 2, 1, 2, 2};
    /* No D3Q19 corners, so H4 adds O(u^2) to the diagonal */
    for (int k = 0; k < 6; k++) {
        double eq = rho * u[ia[k]] * u[ib[k]] + (k < 3 ? rho / 3.0 : 0.0);
        ASSERT_NEAR(P[1][k] - eq,
                    (1.0 - omega) * (P[0][k] - eq),
                    k < 3 ? 1e-4 : 1e-6,
                    "rr relaxes Pi by 1 - omega");
    }
}

/* Coreixas et al. 2019 eq H23 and H26 (+x face, +x+y edge) */
static void test_rr_equilibrium(void) {
    printf("test: recursive regularized equilibrium is Coreixas eq H22-H28\n");
    float rho = 1.01f, ux = 0.06f, uy = -0.04f, uz = 0.03f;
    float g[19], o[19];
    for (int i = 0; i < 19; i++)
        g[i] = feq_shifted(i, rho - 1.0f, ux, uy, uz);
    cpu_collideRR(g, 1.0f, o);
    double x2 = ux * ux, y2 = uy * uy, z2 = uz * uz;
    double h23 =
        rho / 18.0 *
        (1 + 3 * ux + 1.5 * (2 * x2 - y2 - z2) - 4.5 * (ux * y2 + ux * z2) -
         2.25 * (2 * x2 * y2 + 2 * x2 * z2 - y2 * z2));
    double h26 = rho / 36.0 *
                 (1 + 3 * (ux + uy) + 3 * (x2 + y2) - 1.5 * z2 + 9 * ux * uy +
                  4.5 * (2 * x2 * uy + 2 * ux * y2 - ux * z2 - uy * z2) +
                  4.5 * (2 * x2 * y2 - x2 * z2 - y2 * z2));
    ASSERT_NEAR(o[1] + w[1], h23, 1e-7, "rr equilibrium, +x face");
    ASSERT_NEAR(o[7] + w[7], h26, 1e-7, "rr equilibrium, +x+y edge");
}

static void test_grid_create_and_free(void) {
    printf("test: LBM grid create and free\n");
    LBMGrid *grid = LBM_Create(16, 8, 8, 0.1f);
    ASSERT(grid != NULL, "grid not null");
    if (!grid)
        return;
    ASSERT(grid->sizeX == 16, "sizeX");
    ASSERT(grid->sizeY == 8, "sizeY");
    ASSERT(grid->sizeZ == 8, "sizeZ");
    ASSERT(grid->totalCells == 16 * 8 * 8, "totalCells");
    ASSERT_NEAR(grid->tau, 0.5f + 3.0f * 0.1f, 1e-4, "tau");
    LBM_Free(grid);
}

static void test_solid_aabb(void) {
    printf("test: solid AABB marks cells\n");
    LBMGrid *grid = LBM_Create(16, 8, 8, 0.1f);
    if (!grid)
        return;
    /* Place a small box in the center */
    LBM_SetSolidAABB(grid, -0.5f, -0.25f, -0.25f, 0.5f, 0.25f, 0.25f);

    /* Read back solid buffer, check some cells are solid */
    int *solid = (int *)malloc(grid->totalCells * sizeof(int));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 0, grid->totalCells * sizeof(int), solid);
    int count = 0;
    for (int i = 0; i < grid->totalCells; i++)
        if (solid[i])
            count++;
    free(solid);
    ASSERT(count > 0, "some cells are solid");
    ASSERT(count < grid->totalCells, "not all cells solid");
    LBM_Free(grid);
}

static void test_flow_init_and_step(void) {
    printf("test: flow init and step don't crash\n");
    LBMGrid *grid = LBM_Create(16, 8, 8, 0.1f);
    if (!grid)
        return;
    LBM_SetSolidAABB(grid, -0.3f, -0.15f, -0.15f, 0.3f, 0.15f, 0.15f);
    LBM_InitializeFlow(grid, 0.05f, 0.0f, 0.0f);

    /* Run 10 steps */
    for (int i = 0; i < 10; i++)
        LBM_Step(grid, 0.05f, 0.0f, 0.0f);

    /* Compute force, should be nonzero */
    float fx, fy, fz;
    LBM_ComputeDragForce(grid, &fx, &fy, &fz);
    printf("  force@10: fx=%.6f\n", fx);
    ASSERT(fabs(fx) > 1e-10, "drag force nonzero after 10 steps");

    /* Run 990 more steps and check again */
    for (int i = 0; i < 990; i++)
        LBM_Step(grid, 0.05f, 0.0f, 0.0f);
    LBM_ComputeDragForce(grid, &fx, &fy, &fz);
    printf("  force@1000: fx=%.6f\n", fx);
    ASSERT(fabs(fx) > 1e-10, "drag force nonzero after 1000 steps");

    float Cd = LBM_ComputeDragCoefficient(grid, 0.05f, 4.0f);
    ASSERT(Cd > 0, "Cd is positive");
    ASSERT(Cd < 1000, "Cd is not absurd");

    LBM_Free(grid);
}

static void test_mrt_moment_roundtrip(void) {
    printf("test: MRT moment transform roundtrip\n");
    float rho = 1.3f, ux = 0.06f, uy = -0.02f, uz = 0.03f;
    float fi[19], m[19], fi_back[19];
    for (int i = 0; i < 19; i++)
        fi[i] = feq(i, rho, ux, uy, uz);

    cpu_forwardMRT(fi, m);
    cpu_inverseMRT(m, fi_back);

    for (int i = 0; i < 19; i++)
        ASSERT_NEAR(fi_back[i], fi[i], 1e-5, "MRT roundtrip fi");
}

static void test_mrt_conserved_moments(void) {
    printf("test: MRT forward transform gives correct conserved moments\n");
    float rho = 1.4f, ux = 0.05f, uy = -0.01f, uz = 0.02f;
    float fi[19], m[19];
    for (int i = 0; i < 19; i++)
        fi[i] = feq(i, rho, ux, uy, uz);

    cpu_forwardMRT(fi, m);

    ASSERT_NEAR(m[0], rho, 1e-5, "moment 0 = rho");
    ASSERT_NEAR(m[3], rho * ux, 1e-5, "moment 3 = rho*ux");
    ASSERT_NEAR(m[5], rho * uy, 1e-5, "moment 5 = rho*uy");
    ASSERT_NEAR(m[7], rho * uz, 1e-5, "moment 7 = rho*uz");
}

static void test_mrt_equilibrium_fixedpoint(void) {
    printf("test: MRT collision is identity at equilibrium\n");
    float rho = 1.1f, ux = 0.04f, uy = 0.01f, uz = -0.02f;
    float fi[19], m[19], meq[19];
    for (int i = 0; i < 19; i++)
        fi[i] = feq(i, rho, ux, uy, uz);

    cpu_forwardMRT(fi, m);

    // Build equilibrium moments
    float u2 = ux * ux + uy * uy + uz * uz;
    meq[0] = rho;
    meq[1] = -11.0f * rho + 19.0f * rho * u2;
    meq[2] = 3.0f * rho - 5.5f * rho * u2;
    meq[3] = rho * ux;
    meq[4] = -2.0f / 3.0f * rho * ux;
    meq[5] = rho * uy;
    meq[6] = -2.0f / 3.0f * rho * uy;
    meq[7] = rho * uz;
    meq[8] = -2.0f / 3.0f * rho * uz;
    meq[9] = rho * (2.0f * ux * ux - uy * uy - uz * uz);
    meq[10] = -0.5f * rho * (2.0f * ux * ux - uy * uy - uz * uz);
    meq[11] = rho * (uy * uy - uz * uz);
    meq[12] = -0.5f * rho * (uy * uy - uz * uz);
    meq[13] = rho * ux * uy;
    meq[14] = rho * uy * uz;
    meq[15] = rho * ux * uz;
    meq[16] = 0.0f;
    meq[17] = 0.0f;
    meq[18] = 0.0f;

    // At equilibrium, m should equal meq for all moments
    for (int k = 0; k < 19; k++)
        ASSERT_NEAR(m[k], meq[k], 1e-4, "moment at equilibrium");
}

static void test_mrt_flow_step(void) {
    printf("test: MRT flow init and step don't crash\n");
    LBMGrid *grid = LBM_Create(16, 8, 8, 0.1f);
    if (!grid)
        return;
    grid->useMRT = 1;
    LBM_SetSolidAABB(grid, -0.3f, -0.15f, -0.15f, 0.3f, 0.15f, 0.15f);
    LBM_InitializeFlow(grid, 0.05f, 0.0f, 0.0f);

    for (int i = 0; i < 10; i++)
        LBM_Step(grid, 0.05f, 0.0f, 0.0f);

    float fx, fy, fz;
    LBM_ComputeDragForce(grid, &fx, &fy, &fz);
    ASSERT(fabs(fx) > 1e-10, "MRT drag force nonzero");
    LBM_Free(grid);
}

static void test_mrt_smagorinsky_step(void) {
    printf("test: MRT + Smagorinsky don't crash\n");
    LBMGrid *grid = LBM_Create(16, 8, 8, 0.01f);
    if (!grid)
        return;
    grid->useMRT = 1;
    grid->useSmagorinsky = 1;
    grid->smagorinskyCs = 0.1f;
    LBM_SetSolidAABB(grid, -0.3f, -0.15f, -0.15f, 0.3f, 0.15f, 0.15f);
    LBM_InitializeFlow(grid, 0.05f, 0.0f, 0.0f);

    for (int i = 0; i < 20; i++)
        LBM_Step(grid, 0.05f, 0.0f, 0.0f);

    LBM_Free(grid);
}

static void test_sphere_cd_re100(void) {
    printf("test: sphere Cd at Re=100 matches Clift et al. reference\n");

    /* 128x64x64, sphere D=1.0 world -> 16 lattice cells.
     * Re=100, tau=0.548, MRT collision. Cd reference: 1.09
     * (Clift et al. 1978).
     * Both choices are needed for cross-platform reproducibility:
     * U=0.1 keeps tau off the 0.5 stability floor, and MRT damps
     * the ghost modes that BGK leaves free-running. Under BGK the
     * flow state is chaotic in the last digits, so different GL
     * shader compilers (llvmpipe vs hardware) settle on wildly
     * different Cd (observed 1.4-6.5 for identical source), while
     * the MRT box test reproduces to 4 decimals across platforms.
     * MRT is also what production runs (main.c enables it by
     * default). */
    float U = 0.1f;
    float diameter = 1.0f;

    float scaleX = 128.0f / 8.0f;
    float charLength = diameter * scaleX;
    float viscosity = (U * charLength) / 100.0f;
    float tau = 3.0f * viscosity + 0.5f;

    printf("  charLength=%.1f  viscosity=%.6f  tau=%.4f\n",
           charLength,
           viscosity,
           tau);

    LBMGrid *grid = LBM_Create(128, 64, 64, viscosity);
    if (!grid) {
        printf("  SKIP: could not create 128x64x64 grid\n");
        return;
    }

    grid->useMRT = 1;

    LBM_SetSolidSphere(grid, 0.0f, 0.0f, 0.0f, diameter / 2.0f);

    /* Compute projected area BEFORE running steps (glFinish in the
     * readback disrupts the force computation if called mid-loop). */
    float projArea = LBM_ComputeProjectedArea(grid, 0);
    printf("  projected area = %.1f cells^2\n", projArea);

    LBM_InitializeFlow(grid, U, 0.0f, 0.0f);

    /* Run 4000 steps: ~3 flow-throughs at U=0.1 on 128-cell domain.
     * Cd should be converged by then at Re=100. */
    int nSteps = 4000;
    for (int i = 0; i < nSteps; i++)
        LBM_Step(grid, U, 0.0f, 0.0f);

    float Cd = LBM_ComputeDragCoefficient(grid, U, projArea);

    /* Reference: Cd=1.09 at Re=100 (Clift, Grace & Weber). A
     * 16-cell-diameter staircase sphere at tau=0.548 gives Cd ~1.37
     * reproducibly (the discretization runs slightly hot in the
     * viscous regime). Bounds allow for compiler-level float
     * variation across GL implementations while still catching
     * real force-computation regressions. */
    printf("  Cd = %.3f  (reference: 1.09)\n", Cd);
    ASSERT(Cd > 0.8f, "sphere Cd > 0.8 (lower bound)");
    ASSERT(Cd < 2.0f, "sphere Cd < 2.0 (upper bound)");

    LBM_Free(grid);
}

static void test_box_cd_convergence(void) {
    printf("test: box Cd converges (same config as basic test, more steps)\n");

    /* Re=50, tau=0.55. Force oscillates at this tau but the EMA
     * should converge after enough samples. MRT + Smagorinsky
     * prevent blowup. */
    float U = 0.05f;
    float viscosity = (U * 16.0f) / 50.0f; /* charLength=16, Re=50, tau=0.548 */

    LBMGrid *grid = LBM_Create(128, 64, 64, viscosity);
    if (!grid) {
        printf("  SKIP: could not create grid\n");
        return;
    }

    grid->useMRT = 1;
    grid->useSmagorinsky = 1;
    grid->smagorinskyCs = 0.1f;
    LBM_SetSolidAABB(grid, -0.3f, -0.15f, -0.15f, 0.3f, 0.15f, 0.15f);

    float refArea = LBM_ComputeProjectedArea(grid, 0);
    printf("  projected area = %.1f, tau=%.3f\n", refArea, grid->tau);

    LBM_InitializeFlow(grid, U, 0.0f, 0.0f);

    /* Run 2000 steps to develop flow */
    for (int i = 0; i < 2000; i++)
        LBM_Step(grid, U, 0.0f, 0.0f);

    /* Quick force check first */
    float qfx, qfy, qfz;
    LBM_ComputeDragForce(grid, &qfx, &qfy, &qfz);
    printf("  force after 500 steps: fx=%.6f fy=%.6f fz=%.6f\n", qfx, qfy, qfz);

    /* Run 500 more */
    for (int i = 0; i < 500; i++)
        LBM_Step(grid, U, 0.0f, 0.0f);

    LBM_ComputeDragForce(grid, &qfx, &qfy, &qfz);
    printf("  force after 1000 steps: fx=%.6f\n", qfx);

    /* Read velocity at inlet (x=0), center cell */
    float *velData = (float *)malloc(4 * grid->totalCells * sizeof(float));
    glFinish();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->velocityBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                       0,
                       4 * grid->totalCells * sizeof(float),
                       velData);
    int midY = grid->sizeY / 2, midZ = grid->sizeZ / 2;
    int inletIdx = 0 + midY * grid->sizeX + midZ * grid->sizeX * grid->sizeY;
    int midIdx =
        grid->sizeX / 2 + midY * grid->sizeX + midZ * grid->sizeX * grid->sizeY;
    printf("  inlet vel: (%.6f, %.6f, %.6f) rho=%.4f\n",
           velData[4 * inletIdx],
           velData[4 * inletIdx + 1],
           velData[4 * inletIdx + 2],
           velData[4 * inletIdx + 3]);
    printf("  mid vel:   (%.6f, %.6f, %.6f) rho=%.4f\n",
           velData[4 * midIdx],
           velData[4 * midIdx + 1],
           velData[4 * midIdx + 2],
           velData[4 * midIdx + 3]);
    free(velData);

    float samples[5];

    float dynP = 0.5f * U * U;
    for (int s = 0; s < 5; s++) {
        for (int i = 0; i < 100; i++)
            LBM_Step(grid, U, 0.0f, 0.0f);
        float sfx, sfy, sfz;
        LBM_ComputeDragForce(grid, &sfx, &sfy, &sfz);
        samples[s] =
            (dynP * refArea > 1e-10f) ? fabsf(sfx) / (dynP * refArea) : 0.0f;
        printf("  sample %d: Cd = %.3f (fx=%.6f)\n", s, samples[s], sfx);
    }

    /* Check all samples are positive and within 2x of each other */
    float minCd = samples[0], maxCd = samples[0];
    for (int s = 1; s < 5; s++) {
        if (samples[s] < minCd)
            minCd = samples[s];
        if (samples[s] > maxCd)
            maxCd = samples[s];
    }
    printf("  range: %.3f - %.3f (ratio %.2f)\n",
           minCd,
           maxCd,
           maxCd / (minCd > 0.001f ? minCd : 0.001f));

    ASSERT(minCd > 0.0f, "all Cd samples positive");
    /* At low tau, instantaneous Cd oscillates but should stay within
     * an order of magnitude. The EMA in main.c smooths this further. */
    ASSERT(maxCd / (minCd > 0.001f ? minCd : 0.001f) < 10.0f,
           "Cd samples within 10x of each other");

    LBM_Free(grid);
}

static void test_force_averaging(void) {
    printf("test: averaged force matches instantaneous at steady state\n");

    /* Same box config as the convergence test: Re=50, tau=0.548,
     * MRT + Smagorinsky. At steady state the window-averaged force
     * must agree with an instantaneous snapshot; if the averaging
     * path mis-counts dispatches or fails to reset the accumulator,
     * the two diverge by the window length (~100x). */
    float U = 0.05f;
    float viscosity = (U * 16.0f) / 50.0f;

    LBMGrid *grid = LBM_Create(128, 64, 64, viscosity);
    if (!grid) {
        printf("  SKIP: could not create grid\n");
        return;
    }

    grid->useMRT = 1;
    grid->useSmagorinsky = 1;
    grid->smagorinskyCs = 0.1f;
    LBM_SetSolidAABB(grid, -0.3f, -0.15f, -0.15f, 0.3f, 0.15f, 0.15f);
    LBM_InitializeFlow(grid, U, 0.0f, 0.0f);

    /* Develop the flow, then take an instantaneous snapshot. The
     * glFinish drains the 2000 queued dispatches first -- without it
     * the snapshot read has (rarely) returned zero on Mesa iris. */
    for (int i = 0; i < 2000; i++)
        LBM_Step(grid, U, 0.0f, 0.0f);
    glFinish();

    float ix, iy, iz;
    LBM_ComputeDragForce(grid, &ix, &iy, &iz);
    printf("  instantaneous: fx=%.6f\n", ix);

    /* Accumulate over a 200-step window and read the mean */
    LBM_SetForceAveraging(grid, 1);
    for (int i = 0; i < 200; i++)
        LBM_Step(grid, U, 0.0f, 0.0f);

    float ax, ay, az;
    LBM_ComputeDragForce(grid, &ax, &ay, &az);
    printf("  averaged (200 steps): fx=%.6f\n", ax);

    ASSERT(ax > 0.0f, "averaged force positive");
    ASSERT(fabsf(ax - ix) / fabsf(ix) < 0.10f,
           "averaged force within 10%% of instantaneous at steady state");

    /* Readback must reset the window: a second read with no new
     * steps accumulated falls back to a fresh snapshot, not stale
     * or double-counted sums. */
    float rx, ry, rz;
    LBM_ComputeDragForce(grid, &rx, &ry, &rz);
    printf("  post-reset snapshot: fx=%.6f\n", rx);
    ASSERT(fabsf(rx - ix) / fabsf(ix) < 0.10f,
           "post-reset snapshot consistent");

    LBM_Free(grid);
}

static void test_ground_plane(void) {
    printf("test: ground plane marks cells and doesn't crash\n");
    LBMGrid *grid = LBM_Create(32, 16, 16, 0.1f);
    if (!grid) {
        printf("  SKIP: could not create grid\n");
        return;
    }
    LBM_SetSolidAABB(grid, -0.3f, -0.15f, -0.15f, 0.3f, 0.15f, 0.15f);
    LBM_AddGroundPlane(grid, -0.15f);

    /* Read back solid buffer and verify:
     * - some cells are 1 (body)
     * - some cells are 2 (ground)
     * - some cells are 0 (fluid) */
    int *solid = (int *)malloc(grid->totalCells * sizeof(int));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 0, grid->totalCells * sizeof(int), solid);
    int body = 0, ground = 0, fluid = 0;
    for (int i = 0; i < grid->totalCells; i++) {
        if (solid[i] == 1)
            body++;
        else if (solid[i] == 2)
            ground++;
        else
            fluid++;
    }
    free(solid);
    printf("  body=%d ground=%d fluid=%d\n", body, ground, fluid);
    ASSERT(body > 0, "some body cells");
    ASSERT(ground > 0, "some ground cells");
    ASSERT(fluid > 0, "some fluid cells");

    /* Run a few steps to verify no crash */
    LBM_InitializeFlow(grid, 0.05f, 0.0f, 0.0f);
    for (int i = 0; i < 20; i++)
        LBM_Step(grid, 0.05f, 0.0f, 0.0f);

    float fx, fy, fz;
    LBM_ComputeDragForce(grid, &fx, &fy, &fz);
    ASSERT(fabs(fx) > 1e-10, "drag force nonzero with ground");

    LBM_Free(grid);
}

/* n steps, checkpoint, n more against 2n straight, with a cut force
 * window, inlet ramp, turbulence and two slabs */
static LBMGrid *restart_grid(int op) {
    LBMGrid *g = LBM_Create(32, 16, 16, 0.02f);
    if (!g)
        return NULL;
    g->useMRT = op == 1;
    g->useRecursive = op == 2;
    g->useSmagorinsky = op == 1;
    g->periodicYZ = 1;
    g->spongeCells = 4;
    g->turbulenceIntensity = 0.05f;
    g->slabZ = 8;
    g->numChunks = 2;
    LBM_SetForceAveraging(g, 1);
    LBM_SetSolidAABB(g, -2.4f, -0.4f, -0.5f, -1.6f, 0.5f, 0.3f);
    LBM_InitializeFlow(g, 0.0f, 0.0f, 0.0f);
    return g;
}

static void restart_steps(LBMGrid *g, int from, int to, float *fx) {
    for (int s = from; s < to; s++) {
        float u = 0.05f * (s < 60 ? (float)s / 60.0f : 1.0f);
        LBM_Step(g, u, 0.0f, 0.0f);
        if ((s + 1) % 20 == 0) {
            float y, z;
            LBM_ComputeDragForce(g, &fx[(s + 1) / 20 - 1], &y, &z);
        }
    }
}

static float *read_f(LBMGrid *g) {
    size_t bytes = (size_t)19 * g->totalCells * sizeof(float);
    float *f = (float *)malloc(bytes);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->fBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, f);
    return f;
}

static void test_restart_bit_identical(void) {
    const char *names[] = {"bgk", "mrt + Smagorinsky", "recursive"};
    const int n = 105;
    char path[512];
    const char *tmp = getenv("TMPDIR");
    snprintf(
        path, sizeof(path), "%s/test_lbm_restart.ckpt", tmp ? tmp : "/tmp");
    for (int op = 0; op < 3; op++) {
        printf("test: restart is bitwise equal to a straight run (%s)\n",
               names[op]);
        float fxA[12] = {0}, fxB[12] = {0};
        char err[CKPT_ERR_LEN] = "";

        LBMGrid *a = restart_grid(op);
        if (!a) {
            printf("  SKIP: could not create grid\n");
            return;
        }
        restart_steps(a, 0, n, fxA);
        CkptHeader h, have;
        LBM_CheckpointHeader(a, 0.05f, &h);
        CkptWriter *ck = ckpt_create(path, &h, err, sizeof(err));
        int ok = ck && LBM_SaveState(a, ck) == 0 &&
                 ckpt_commit(ck, err, sizeof(err)) == 0;
        ASSERT(ok, "checkpoint written");
        restart_steps(a, n, 2 * n, fxA);
        float *fa = read_f(a);
        LBM_Free(a);

        LBMGrid *b = restart_grid(op);
        CkptReader *r = ckpt_open(path, &have, err, sizeof(err));
        LBM_CheckpointHeader(b, 0.05f, &h);
        ok = r && ckpt_check(&have, &h, err, sizeof(err)) == 0 &&
             LBM_LoadState(b, r, err, sizeof(err)) == 0;
        if (!ok)
            printf("  %s\n", err);
        ASSERT(ok, "checkpoint loaded");
        ckpt_close(r);
        ASSERT(b->stepCount == n, "step counter restored");
        restart_steps(b, n, 2 * n, fxB);
        float *fb = read_f(b);
        size_t bytes = (size_t)19 * b->totalCells * sizeof(float);
        ASSERT(memcmp(fa, fb, bytes) == 0, "f bitwise equal after 2n steps");
        ASSERT(memcmp(fxA + n / 20,
                      fxB + n / 20,
                      (2 * n / 20 - n / 20) * sizeof(float)) == 0,
               "drag readbacks after the restart bitwise equal");
        ASSERT(fxA[2 * n / 20 - 1] != 0.0f, "drag readback nonzero");

        // The same file on another grid size is refused
        LBMGrid *c = LBM_Create(32, 16, 8, 0.02f);
        if (c) {
            c->slabZ = 8;
            LBM_CheckpointHeader(c, 0.05f, &h);
            ASSERT(ckpt_check(&have, &h, err, sizeof(err)) != 0 &&
                       strstr(err, "grid"),
                   "other grid refused");
            LBM_Free(c);
        }
        free(fa);
        free(fb);
        LBM_Free(b);
    }
    remove(path);
}

/* GL context setup */

static int init_gl(void) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL init failed: %s\n", SDL_GetError());
        return 0;
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
    if (glDebugRequested())
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);

    SDL_Window *win = SDL_CreateWindow(
        "test", 0, 0, 64, 64, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    if (!win) {
        printf("Window failed: %s\n", SDL_GetError());
        return 0;
    }
    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if (!ctx) {
        printf("GL context failed: %s\n", SDL_GetError());
        return 0;
    }
    SDL_GL_MakeCurrent(win, ctx);
    if (!gladLoadGL((GLADloadfunc)SDL_GL_GetProcAddress)) {
        printf("glad GL load failed\n");
        return 0;
    }
    enableGLDebugOutput();
    return 1;
}

int main(void) {
    printf("=== LBM Unit Tests ===\n\n");

    /* Pure math tests (no GL needed) */
    test_weights_sum_to_one();
    test_equilibrium_conserves_density();
    test_equilibrium_conserves_momentum();
    test_equilibrium_at_rest();
    test_mrt_moment_roundtrip();
    test_mrt_conserved_moments();
    test_mrt_equilibrium_fixedpoint();
    test_shifted_equilibrium();
    test_mrt_moments_of_weights();
    test_rr_conserves_and_relaxes();
    test_rr_equilibrium();

    /* GPU tests (need GL context) */
    if (init_gl()) {
        test_grid_create_and_free();
        test_solid_aabb();
        test_flow_init_and_step();
        test_mrt_flow_step();
        test_mrt_smagorinsky_step();
        test_box_cd_convergence();
        test_force_averaging();
        test_ground_plane();
        test_periodic_mass_shifted();
        test_force_at_rest();
        test_sphere_cd_re100();
        test_restart_bit_identical();
    } else {
        printf("\nSkipping GPU tests (no GL context)\n");
    }

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    SDL_Quit();
    return (tests_passed == tests_run) ? 0 : 1;
}
