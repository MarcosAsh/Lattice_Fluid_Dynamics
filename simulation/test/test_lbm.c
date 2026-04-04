/*
 * LBM unit tests.
 *
 * Needs an OpenGL context, so run under Xvfb in CI:
 *   xvfb-run ./build/test_lbm
 */

#include "../lib/lbm.h"
#include <glad/gl.h>
#include <SDL2/SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
    ASSERT(fabs(fx) > 1e-10, "drag force nonzero after 10 steps");

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

    /* 128x64x64 gives uniform scaling (16 cells per world unit).
     * Sphere D = 1.0 world -> 16 lattice cells diameter.
     * Blockage ~5%.  Cd reference: 1.09 (Clift, Grace & Weber 1978). */
    float U = 0.05f;
    float diameter = 1.0f; /* world units */
    float Re = 100.0f;

    float scaleX = 128.0f / 8.0f; /* = 16 */
    float charLength = diameter * scaleX;
    float viscosity = (U * charLength) / Re;
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

    LBM_SetSolidSphere(grid, 0.0f, 0.0f, 0.0f, diameter / 2.0f);
    LBM_InitializeFlow(grid, U, 0.0f, 0.0f);

    /* Run 4000 steps: ~2.5 flow-throughs at U=0.05 on 128-cell domain.
     * Cd should be converged by then at Re=100. */
    int nSteps = 4000;
    for (int i = 0; i < nSteps; i++)
        LBM_Step(grid, U, 0.0f, 0.0f);

    /* Use actual projected area instead of pi/4 * D^2 */
    float projArea = LBM_ComputeProjectedArea(grid, 0);
    printf("  projected area = %.1f cells^2\n", projArea);
    float Cd = LBM_ComputeDragCoefficient(grid, U, projArea);

    printf("  Cd = %.3f  (reference: 1.09, tol 30%%)\n", Cd);
    ASSERT(Cd > 0.76f, "sphere Cd > 0.76 (lower bound)");
    ASSERT(Cd < 1.42f, "sphere Cd < 1.42 (upper bound)");

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

    /* GPU tests (need GL context) */
    if (init_gl()) {
        test_grid_create_and_free();
        test_solid_aabb();
        test_flow_init_and_step();
        test_mrt_flow_step();
        test_mrt_smagorinsky_step();
        test_ground_plane();
        test_sphere_cd_re100();
    } else {
        printf("\nSkipping GPU tests (no GL context)\n");
    }

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    SDL_Quit();
    return (tests_passed == tests_run) ? 0 : 1;
}
