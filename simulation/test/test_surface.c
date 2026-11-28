/*
 * Surface output tests on a flat wall. No GL needed.
 */

#include "../lib/surface.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_failed = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        tests_run++;                                                           \
        if (!(cond)) {                                                         \
            tests_failed++;                                                    \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__);                   \
        }                                                                      \
    } while (0)

#define N  8
#define U  0.05
#define WZ 3.0 // wall height in cells, fluid above

// Fluid cells at z = 3 touch the wall below, cells at x = 1, z = 6 are
// the reference plane
static void build(Surface *s) {
    int n = N * N + N;
    int *cell = (int *)malloc(n * sizeof(int));
    unsigned char *role = (unsigned char *)malloc(n);
    int *w36 = (int *)calloc((size_t)n * 3, sizeof(int));
    int *links = (int *)calloc(n, sizeof(int));
    int k = 0;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++, k++) {
            cell[k] = x + N * (y + N * 3);
            role[k] = SURF_BOUNDARY;
            // Links with e_z = -1: one face (72 w = 4), four edges (2)
            w36[3 * k + 2] = -12;
            links[k] = 5;
        }
    for (int y = 0; y < N; y++, k++) {
        cell[k] = 1 + N * (y + N * 6);
        role[k] = SURF_REFERENCE;
    }
    surface_init(s, N, N, N, n, cell, role, w36, links);
}

// World coordinates of LBM_SetSolidMesh, lattice x = (w + 4) N / 8
static void wall_mesh(float *t) {
    float x0 = -4.0f, x1 = 4.0f, y0 = -2.0f, y1 = 2.0f;
    float z = (float)(WZ / (N / 4.0) - 2.0);
    float v[2][3][3] = {{{x0, y0, z}, {x1, y0, z}, {x1, y1, z}},
                        {{x0, y0, z}, {x1, y1, z}, {x0, y1, z}}};
    memset(t, 0, 24 * sizeof(float));
    for (int i = 0; i < 2; i++)
        for (int a = 0; a < 3; a++)
            for (int c = 0; c < 3; c++)
                t[12 * i + 4 * a + c] = v[i][a][c];
}

// Two windows of 10 steps, shifted force (0.001, 0, -0.002) per step
static void fill(Surface *s) {
    float *sums = (float *)calloc((size_t)s->n * 4, sizeof(float));
    for (int w = 0; w < 2; w++) {
        for (int k = 0; k < s->n; k++) {
            int b = s->role[k] & SURF_BOUNDARY;
            sums[4 * k + 0] = b ? 0.01f : 0.0f;
            sums[4 * k + 2] = b ? -0.02f : 0.0f;
            sums[4 * k + 3] = b ? 10.1f : 10.0f;
        }
        float drag[3] = {0.064f, 0.0f, (float)(-64.0 / 3.0 - 0.128)};
        surface_add_window(s, sums, 10, drag);
    }
    free(sums);
}

static void test_flat_wall(void) {
    printf("test: flat wall values and force balance\n");
    Surface s;
    build(&s);
    float tris[24];
    wall_mesh(tris);
    CHECK(surface_map_mesh(&s, tris, 2) == 0, "map");
    CHECK(s.ntri > 2 && s.unowned == 0, "triangles split and all in reach");
    fill(&s);

    double area = 0, tri = 0;
    for (int k = 0; k < s.n; k++)
        area += s.area[k];
    for (int t = 0; t < s.ntri; t++) {
        tri += s.triArea[t];
        if (fabs(s.triNormal[3 * t + 2] - 1.0) > 1e-12) {
            CHECK(0, "triangle normal points into the fluid");
            break;
        }
    }
    CHECK(fabs(tri - 64.0) < 1e-4, "mesh area");
    CHECK(fabs(area - tri) < 1e-9, "cell areas add up to the mesh area");
    CHECK(fabs(surface_reference_rho(&s) - 1.0) < 1e-6, "reference density");

    SurfacePoint p;
    surface_point(&s, 3 * N + 4, U, 1.0, &p);
    double q = 0.5 * U * U;
    CHECK(fabs(p.rho - 1.01) < 1e-6, "time mean density");
    CHECK(fabs(p.cp - 0.01 / 3.0 / q) < 1e-3, "cp from density");
    CHECK(fabs(p.force[2] - (-1.0 / 3.0 - 0.002)) < 1e-9, "force includes W");
    CHECK(p.wall && fabs(p.cpWall - 0.002 / q) < 1e-4, "cp_wall");
    CHECK(fabs(p.shear[0] - 0.001) < 1e-9 && fabs(p.shear[2]) < 1e-12,
          "shear is the tangential force over the link area");
    CHECK(fabs(p.cf - 0.001 / q) < 1e-4, "cf");
    CHECK(p.mapped &&
              fabs(p.traction[2] * s.area[3 * N + 4] - p.force[2]) < 1e-12,
          "traction times area is the force");

    double cells[3], mesh[3], drag[3], un[3];
    surface_totals(&s, cells, mesh, drag, un);
    double fz = 64 * (-1.0 / 3.0 - 0.002);
    CHECK(fabs(cells[2] - fz) < 1e-7 && fabs(cells[0] - 0.064) < 1e-7,
          "cell sum");
    CHECK(fabs(mesh[0] - cells[0]) < 1e-12 && fabs(mesh[2] - cells[2]) < 1e-12,
          "mesh integral equals the cell sum");
    CHECK(fabs(drag[0] - 0.064) < 1e-7 && fabs(drag[2] - fz) < 1e-5,
          "drag readback mean");
    CHECK(un[0] == 0 && un[2] == 0, "no unmapped force");

    double cp, cpw, tr[3];
    CHECK(surface_triangle(&s, 0, U, 1.0, &cp, &cpw, tr) &&
              fabs(cpw - 0.002 / q) < 1e-4,
          "triangle cp_wall");

    const char *path = "test_surface.vtk";
    double centre[3] = {4.0, 4.0, 3.5};
    CHECK(surface_write(&s, path, U, 1, centre) == 0, "write");
    FILE *f = fopen("test_surface_plane.csv", "r");
    int rows = -1;
    char line[1024];
    while (f && fgets(line, sizeof(line), f))
        rows++;
    if (f)
        fclose(f);
    CHECK(rows == 2 * N, "plane CSV has the two cell rows next to y = 4");
    f = fopen(path, "r");
    CHECK(f && fgets(line, sizeof(line), f) &&
              strncmp(line, "# vtk DataFile", 14) == 0,
          "mesh file is legacy VTK");
    if (f)
        fclose(f);
    f = fopen("test_surface_cells.vtk", "r");
    CHECK(f != NULL, "cells file written");
    if (f)
        fclose(f);
    remove(path);
    remove("test_surface_cells.vtk");
    remove("test_surface_plane.csv");
    surface_free(&s);
}

static void test_no_mesh(void) {
    printf("test: without a mesh\n");
    Surface s;
    build(&s);
    fill(&s);
    SurfacePoint p;
    surface_point(&s, 0, U, 1.0, &p);
    double q = 0.5 * U * U;
    CHECK(!p.mapped, "no area");
    CHECK(fabs(s.normal[2] - 1.0) < 1e-12, "normal from the links");
    CHECK(p.wall && fabs(p.cpWall - 0.002 / q) < 1e-4, "cp_wall from links");
    surface_free(&s);
}

int main(void) {
    printf("=== Surface Tests ===\n\n");
    test_flat_wall();
    test_no_mesh();
    printf("\n=== Results: %d/%d passed ===\n",
           tests_run - tests_failed,
           tests_run);
    return tests_failed ? 1 : 0;
}
