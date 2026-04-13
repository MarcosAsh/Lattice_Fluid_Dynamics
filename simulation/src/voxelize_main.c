/*
 * voxelize_obj: standalone CLI that rasterizes an OBJ mesh into a
 * fixed-resolution binary occupancy grid (.voxbin). Used by the ML
 * dataset pipeline to generate input tensors for the Cd surrogate.
 *
 * Example:
 *   ./build/voxelize_obj assets/3d-files/ahmed_25deg_m.obj \
 *       --resolution 32 --output ahmed25.voxbin
 */

#include "../lib/voxelize.h"
#include "../obj-file-loader/lib/model_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr,
            "usage: %s <input.obj> [--resolution N] [--padding F] "
            "[--mode solid|surface|auto] [--no-align] [--output PATH]\n",
            prog);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    const char *input_path = argv[1];
    int resolution = 32;
    float padding = 0.05f;
    VoxelizeMode mode = VOXELIZE_MODE_AUTO;
    int align_longest_x = 1;
    const char *output_path = NULL;

    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--resolution") == 0 && i + 1 < argc) {
            resolution = atoi(argv[++i]);
        } else if (strcmp(a, "--padding") == 0 && i + 1 < argc) {
            padding = (float)atof(argv[++i]);
        } else if (strcmp(a, "--mode") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            if (strcmp(m, "solid") == 0) mode = VOXELIZE_MODE_SOLID;
            else if (strcmp(m, "surface") == 0) mode = VOXELIZE_MODE_SURFACE;
            else if (strcmp(m, "auto") == 0) mode = VOXELIZE_MODE_AUTO;
            else {
                fprintf(stderr, "unknown mode: %s\n", m);
                return 2;
            }
        } else if (strcmp(a, "--no-align") == 0) {
            align_longest_x = 0;
        } else if (strcmp(a, "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            fprintf(stderr, "unknown argument: %s\n", a);
            usage(argv[0]);
            return 2;
        }
    }

    if (resolution <= 0 || resolution > 512) {
        fprintf(stderr, "resolution must be in [1, 512], got %d\n", resolution);
        return 2;
    }

    Model model = loadOBJ(input_path);
    if (model.vertexCount == 0 || model.faceCount == 0) {
        fprintf(stderr, "failed to load OBJ: %s\n", input_path);
        freeModel(&model);
        return 1;
    }

    VoxelGrid *grid =
        Voxelize_FromModel(&model, resolution, padding, align_longest_x, mode);
    freeModel(&model);

    if (!grid) {
        fprintf(stderr, "voxelization failed for %s\n", input_path);
        return 1;
    }

    float frac = Voxelize_SolidFraction(grid);
    size_t total = (size_t)grid->resolution * grid->resolution * grid->resolution;
    size_t solid = (size_t)(frac * (float)total + 0.5f);
    printf("voxelized %s: %dx%dx%d, solid=%zu/%zu (%.2f%%)\n",
           input_path, grid->resolution, grid->resolution, grid->resolution,
           solid, total, 100.0f * frac);

    int rc = 0;
    if (output_path) {
        if (Voxelize_WriteBinary(grid, output_path) != 0) {
            fprintf(stderr, "failed to write %s\n", output_path);
            rc = 1;
        } else {
            printf("wrote %s\n", output_path);
        }
    }

    Voxelize_Free(grid);
    return rc;
}
