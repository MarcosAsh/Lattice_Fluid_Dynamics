/*
 * Unit tests for the voxelizer. Pure CPU code -- no GL context needed.
 */

#include "../lib/voxelize.h"
#include "../obj-file-loader/lib/model_loader.h"

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

/*
 * Build a watertight axis-aligned cube of half-extent h centered at the
 * origin, using 8 vertices and 12 triangles.
 */
static Model make_cube(float h) {
    Model m = {0};
    m.vertexCount = 8;
    m.faceCount = 12;
    m.vertices = (Vertex *)calloc(8, sizeof(Vertex));
    m.faces = (Face *)calloc(12, sizeof(Face));

    Vertex verts[8] = {
        {-h, -h, -h}, { h, -h, -h}, { h,  h, -h}, {-h,  h, -h},
        {-h, -h,  h}, { h, -h,  h}, { h,  h,  h}, {-h,  h,  h},
    };
    memcpy(m.vertices, verts, sizeof(verts));

    /* OBJ face indices are 1-based. */
    Face faces[12] = {
        {1, 2, 3}, {1, 3, 4},   /* -Z */
        {5, 7, 6}, {5, 8, 7},   /* +Z */
        {1, 5, 6}, {1, 6, 2},   /* -Y */
        {4, 3, 7}, {4, 7, 8},   /* +Y */
        {1, 4, 8}, {1, 8, 5},   /* -X */
        {2, 6, 7}, {2, 7, 3},   /* +X */
    };
    memcpy(m.faces, faces, sizeof(faces));
    return m;
}

/*
 * Build a thin rectangular slab that is longer in Z than in X,
 * so align_longest_x has to rotate it.
 */
static Model make_slab_z(void) {
    Model m = {0};
    m.vertexCount = 8;
    m.faceCount = 12;
    m.vertices = (Vertex *)calloc(8, sizeof(Vertex));
    m.faces = (Face *)calloc(12, sizeof(Face));

    float hx = 0.1f, hy = 0.2f, hz = 0.8f;
    Vertex verts[8] = {
        {-hx, -hy, -hz}, { hx, -hy, -hz}, { hx,  hy, -hz}, {-hx,  hy, -hz},
        {-hx, -hy,  hz}, { hx, -hy,  hz}, { hx,  hy,  hz}, {-hx,  hy,  hz},
    };
    memcpy(m.vertices, verts, sizeof(verts));

    Face faces[12] = {
        {1, 2, 3}, {1, 3, 4},
        {5, 7, 6}, {5, 8, 7},
        {1, 5, 6}, {1, 6, 2},
        {4, 3, 7}, {4, 7, 8},
        {1, 4, 8}, {1, 8, 5},
        {2, 6, 7}, {2, 7, 3},
    };
    memcpy(m.faces, faces, sizeof(faces));
    return m;
}

static void free_test_model(Model *m) {
    free(m->vertices);
    free(m->faces);
    memset(m, 0, sizeof(*m));
}

/* Test 1: a unit cube in solid mode should produce a (nearly) full grid. */
static void test_cube_solid(void) {
    printf("test_cube_solid\n");
    Model cube = make_cube(1.0f);

    VoxelGrid *g = Voxelize_FromModel(&cube, 16, 0.05f, 1, VOXELIZE_MODE_SOLID);
    CHECK(g != NULL, "voxelize returned non-null");
    if (g) {
        float frac = Voxelize_SolidFraction(g);
        CHECK(frac > 0.80f, "cube should be mostly solid");
        CHECK(frac <= 1.0f, "solid fraction <= 1");
        printf("  cube solid fraction: %.3f\n", frac);
        Voxelize_Free(g);
    }

    free_test_model(&cube);
}

/* Test 2: surface mode on a cube should produce a hollow shell. */
static void test_cube_surface(void) {
    printf("test_cube_surface\n");
    Model cube = make_cube(1.0f);

    VoxelGrid *g =
        Voxelize_FromModel(&cube, 16, 0.05f, 1, VOXELIZE_MODE_SURFACE);
    CHECK(g != NULL, "voxelize returned non-null");
    if (g) {
        float frac = Voxelize_SolidFraction(g);
        /* Shell of a 16^3 grid has maybe ~40-60% cells marked. */
        CHECK(frac > 0.20f && frac < 0.95f, "surface frac in plausible range");
        printf("  cube surface fraction: %.3f\n", frac);

        /* Center cell should be empty. */
        int R = g->resolution;
        int c = R / 2;
        uint8_t center = g->data[c + R * (c + R * c)];
        CHECK(center == 0, "surface mode leaves interior empty");
        Voxelize_Free(g);
    }

    free_test_model(&cube);
}

/* Test 3: align_longest_x on a Z-elongated slab should rotate it into X. */
static void test_align_longest_x(void) {
    printf("test_align_longest_x\n");
    Model slab = make_slab_z();
    int R = 32;

    VoxelGrid *g =
        Voxelize_FromModel(&slab, R, 0.05f, 1, VOXELIZE_MODE_SOLID);
    CHECK(g != NULL, "voxelize returned non-null");
    if (g) {
        /* After alignment, the slab should span more cells in X than in Z. */
        int x_min = R, x_max = -1, z_min = R, z_max = -1;
        for (int z = 0; z < R; z++) {
            for (int y = 0; y < R; y++) {
                for (int x = 0; x < R; x++) {
                    if (g->data[x + R * (y + R * z)]) {
                        if (x < x_min) x_min = x;
                        if (x > x_max) x_max = x;
                        if (z < z_min) z_min = z;
                        if (z > z_max) z_max = z;
                    }
                }
            }
        }
        int x_span = x_max - x_min;
        int z_span = z_max - z_min;
        printf("  x_span=%d z_span=%d\n", x_span, z_span);
        CHECK(x_span > z_span, "longest axis realigned to X");
        Voxelize_Free(g);
    }

    free_test_model(&slab);
}

/* Test 4: voxelization must be deterministic. */
static void test_determinism(void) {
    printf("test_determinism\n");
    Model cube = make_cube(1.0f);

    VoxelGrid *g1 = Voxelize_FromModel(&cube, 16, 0.05f, 1, VOXELIZE_MODE_SOLID);
    VoxelGrid *g2 = Voxelize_FromModel(&cube, 16, 0.05f, 1, VOXELIZE_MODE_SOLID);
    CHECK(g1 && g2, "both voxelizations succeeded");
    if (g1 && g2) {
        size_t n = 16 * 16 * 16;
        CHECK(memcmp(g1->data, g2->data, n) == 0, "two runs produce identical grids");
    }
    Voxelize_Free(g1);
    Voxelize_Free(g2);
    free_test_model(&cube);
}

/* Test 5: binary round-trip through Voxelize_WriteBinary. */
static void test_write_binary(void) {
    printf("test_write_binary\n");
    Model cube = make_cube(1.0f);
    VoxelGrid *g = Voxelize_FromModel(&cube, 8, 0.05f, 1, VOXELIZE_MODE_SOLID);
    CHECK(g != NULL, "voxelize ok");

    const char *path = "voxelize_test_tmp.voxbin";
    int rc = Voxelize_WriteBinary(g, path);
    CHECK(rc == 0, "write binary returns 0");

    FILE *f = fopen(path, "rb");
    CHECK(f != NULL, "file exists");
    if (f) {
        char magic[4];
        uint32_t version = 0, res = 0;
        size_t n_read = fread(magic, 1, 4, f);
        CHECK(n_read == 4 && magic[0] == 'V' && magic[1] == 'O' &&
                  magic[2] == 'X' && magic[3] == 'B',
              "magic header");
        n_read = fread(&version, sizeof(uint32_t), 1, f);
        CHECK(n_read == 1 && version == 1, "version == 1");
        n_read = fread(&res, sizeof(uint32_t), 1, f);
        CHECK(n_read == 1 && res == 8, "resolution == 8");

        size_t expected = 8 * 8 * 8;
        uint8_t *buf = (uint8_t *)malloc(expected);
        size_t got = fread(buf, 1, expected, f);
        CHECK(got == expected, "read full voxel payload");
        CHECK(memcmp(buf, g->data, expected) == 0, "payload round-trips");
        free(buf);
        fclose(f);
        remove(path);
    }

    Voxelize_Free(g);
    free_test_model(&cube);
}

int main(void) {
    printf("voxelize unit tests\n");
    test_cube_solid();
    test_cube_surface();
    test_align_longest_x();
    test_determinism();
    test_write_binary();

    printf("%d/%d checks passed\n", tests_run - tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
