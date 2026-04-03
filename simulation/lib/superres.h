#ifndef SUPERRES_H
#define SUPERRES_H

#include <glad/gl.h>

typedef struct SuperResUpscaler SuperResUpscaler;

// Create the super-resolution upscaler.
// Loads MLP weights from weightsPath and normalizer from normPath.
// Allocates fine velocity buffer at 2x the coarse grid resolution.
SuperResUpscaler *SR_Create(int coarseX, int coarseY, int coarseZ,
                            const char *weightsPath,
                            const char *normPath);

// Free all resources.
void SR_Free(SuperResUpscaler *sr);

// Run upscaling: reads from coarseVelBuffer, writes to internal fine buffer.
// Call after each LBM step.
void SR_Upscale(SuperResUpscaler *sr, GLuint coarseVelBuffer);

// Get the fine velocity buffer for particle/rendering systems to sample.
GLuint SR_GetFineVelocityBuffer(SuperResUpscaler *sr);

// Get fine grid dimensions.
void SR_GetFineSize(SuperResUpscaler *sr, int *fineX, int *fineY, int *fineZ);

#endif // SUPERRES_H
