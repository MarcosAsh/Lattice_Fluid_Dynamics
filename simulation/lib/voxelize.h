#ifndef LATTICE_VOXELIZE_H
#define LATTICE_VOXELIZE_H

#include <stdint.h>
#include "../obj-file-loader/lib/model_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VOXELIZE_MODE_SOLID = 0,   /* parity-fill ray casting                  */
    VOXELIZE_MODE_SURFACE = 1, /* mark voxels touched by any triangle AABB */
    VOXELIZE_MODE_AUTO = 2     /* solid, fall back to surface if degenerate */
} VoxelizeMode;

typedef struct {
    int resolution;   /* R (grid is R x R x R)               */
    uint8_t *data;    /* R*R*R bytes, index = x + R*(y + R*z) */
} VoxelGrid;

/*
 * Voxelize a loaded OBJ model into a fixed-resolution occupancy grid.
 *
 * The mesh is centered at the origin, optionally permuted so its longest
 * AABB axis is streamwise (X), and uniformly scaled so the largest
 * dimension spans (2 - 2*padding) world units inside the [-1, 1]^3 cube.
 * Padding gives the CNN a border around the object to see the boundary
 * layer region.
 *
 * Returns NULL on allocation failure or if the model is empty.
 */
VoxelGrid *Voxelize_FromModel(const Model *model,
                              int resolution,
                              float padding,
                              int align_longest_x,
                              VoxelizeMode mode);

/*
 * Write a grid to a simple binary file.
 *
 * Format:
 *   u32 magic    = 'V','O','X','B'
 *   u32 version  = 1
 *   u32 resolution
 *   u8[R*R*R]    occupancy bytes
 *
 * Returns 0 on success, non-zero on I/O failure.
 */
int Voxelize_WriteBinary(const VoxelGrid *grid, const char *path);

/*
 * Fraction of cells marked solid (in [0, 1]).
 */
float Voxelize_SolidFraction(const VoxelGrid *grid);

void Voxelize_Free(VoxelGrid *grid);

#ifdef __cplusplus
}
#endif

#endif /* LATTICE_VOXELIZE_H */
