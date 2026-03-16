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

#define ASSERT(cond, msg)                                \
    do {                                                 \
        tests_run++;                                     \
        if (!(cond)) {                                   \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        } else {                                         \
            tests_passed++;                              \
        }                                                \
    } while (0)

#define ASSERT_NEAR(a, b, tol, msg)                      \
    do {                                                 \
        tests_run++;                                     \
        if (fabs((double)(a) - (double)(b)) > (tol)) {   \
            printf("  FAIL: %s -- got %.6f, want %.6f"   \
                   " (line %d)\n",                       \
                   msg, (double)(a), (double)(b),        \
                   __LINE__);                            \
        } else {                                         \
            tests_passed++;                              \
        }                                                \
    } while (0)

/* D3Q19 weights (same as in lbm.c) */
static const float w[19] = {
    1.0f / 3.0f,
    1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
    1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};

static const int ex[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1,
    -1, 1, -1, 1, -1, 0, 0, 0, 0};
static const int ey[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, 1, -1,
    -1, 0, 0, 0, 0, 1, -1, 1, -1};
static const int ez[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0,
    0, 1, 1, -1, -1, 1, 1, -1, -1};

static float feq(
    int i, float rho, float ux, float uy, float uz) {
    float eu = ex[i] * ux + ey[i] * uy + ez[i] * uz;
    float u2 = ux * ux + uy * uy + uz * uz;
    float cs2 = 1.0f / 3.0f;
    return w[i] * rho *
           (1.0f + eu / cs2 + (eu * eu) / (2.0f * cs2 * cs2) -
            u2 / (2.0f * cs2));
}

/* ---- Tests ---- */

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
    LBM_SetSolidAABB(grid, -0.5f, -0.25f, -0.25f,
                      0.5f, 0.25f, 0.25f);

    /* Read back solid buffer, check some cells are solid */
    int *solid = (int *)malloc(
        grid->totalCells * sizeof(int));
    glBindBuffer(
        GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       grid->totalCells * sizeof(int),
                       solid);
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
    LBM_SetSolidAABB(grid, -0.3f, -0.15f, -0.15f,
                      0.3f, 0.15f, 0.15f);
    LBM_InitializeFlow(grid, 0.05f, 0.0f, 0.0f);

    /* Run 10 steps */
    for (int i = 0; i < 10; i++)
        LBM_Step(grid, 0.05f, 0.0f, 0.0f);

    /* Compute force, should be nonzero */
    float fx, fy, fz;
    LBM_ComputeDragForce(grid, &fx, &fy, &fz);
    ASSERT(fabs(fx) > 1e-10, "drag force nonzero after 10 steps");

    float Cd = LBM_ComputeDragCoefficient(
        grid, 0.05f, 4.0f);
    ASSERT(Cd > 0, "Cd is positive");
    ASSERT(Cd < 1000, "Cd is not absurd");

    LBM_Free(grid);
}

/* ---- GL context setup ---- */

static int init_gl(void) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL init failed: %s\n", SDL_GetError());
        return 0;
    }
    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_PROFILE_MASK,
        SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);

    SDL_Window *win = SDL_CreateWindow(
        "test", 0, 0, 64, 64,
        SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    if (!win) {
        printf("Window failed: %s\n", SDL_GetError());
        return 0;
    }
    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if (!ctx) {
        printf("GL context failed: %s\n",
               SDL_GetError());
        return 0;
    }
    SDL_GL_MakeCurrent(win, ctx);
    if (!gladLoadGL(
            (GLADloadfunc)SDL_GL_GetProcAddress)) {
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

    /* GPU tests (need GL context) */
    if (init_gl()) {
        test_grid_create_and_free();
        test_solid_aabb();
        test_flow_init_and_step();
    } else {
        printf("\nSkipping GPU tests (no GL context)\n");
    }

    printf("\n=== Results: %d/%d passed ===\n",
           tests_passed, tests_run);
    SDL_Quit();
    return (tests_passed == tests_run) ? 0 : 1;
}
