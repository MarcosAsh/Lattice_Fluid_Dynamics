#ifndef VTI_EXPORT_H
#define VTI_EXPORT_H

#include "lbm.h"

// Running velocity statistics accumulated over many timesteps. Stores the
// per-cell sum and sum-of-squares of each velocity component; the mean,
// RMS fluctuation and turbulent kinetic energy are derived at write time.
typedef struct {
    int total;          // number of cells
    long count;         // samples accumulated so far
    float *sumVel;      // 3*total: running sum of (u, v, w)
    float *sumVelSq;    // 3*total: running sum of (u^2, v^2, w^2)
} VTKStats;

// Allocate an accumulator for a grid of `total` cells, or NULL on failure.
VTKStats *VTKStats_Create(int total);
void VTKStats_Free(VTKStats *stats);

// Read the current velocity field from the GPU and fold it into the
// running statistics. Call once per timestep over the averaging window.
void VTKStats_Accumulate(VTKStats *stats, LBMGrid *grid);

// Write LBM velocity, density (rho) and solid mask as a VTK ImageData
// (.vti) file at path/field_<step>.vti, binary appended format. When
// `stats` is non-NULL and holds at least one sample, also writes the
// time-averaged velocity, per-component RMS fluctuation and turbulent
// kinetic energy as extra point-data arrays. Pass NULL for stats to write
// only the instantaneous fields.
void writeVTI(LBMGrid *grid, VTKStats *stats, const char *path, int step);

#endif // VTI_EXPORT_H
