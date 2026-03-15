#ifndef LBM_H
#define LBM_H

#include <GL/glew.h>

typedef struct {
    int sizeX, sizeY, sizeZ;
    int totalCells;

    float tau;

    // GPU buffers
    GLuint fBuffer;
    GLuint fNewBuffer;
    GLuint velocityBuffer;
    GLuint solidBuffer;
    GLuint forceBuffer; // For drag calculation

    // Shaders
    GLuint collideShader;
    GLuint streamShader;
    GLuint forceShader;

    // Uniform locations
    GLint collide_gridSizeLoc;
    GLint collide_tauLoc;
    GLint collide_inletVelLoc;
    GLint collide_useRegularizedLoc;
    GLint stream_gridSizeLoc;
    GLint stream_periodicYZLoc;
    GLint force_gridSizeLoc;

    GLint collide_useSmagorinskyLoc;
    GLint collide_smaCsLoc;

    int useRegularized; // 0 = BGK, 1 = regularized
    int useSmagorinsky; // 0 = off, 1 = Smagorinsky SGS
    float smagorinskyCs; // Smagorinsky constant
    int periodicYZ;     // 0 = clamp, 1 = periodic y/z
} LBMGrid;

// Initialize LBM grid
LBMGrid *LBM_Create(int sizeX, int sizeY, int sizeZ, float viscosity);

// Free LBM resources
void LBM_Free(LBMGrid *grid);

// Set solid cells from car model bounds
void LBM_SetSolidAABB(LBMGrid *grid,
                      float minX,
                      float minY,
                      float minZ,
                      float maxX,
                      float maxY,
                      float maxZ);

// Initialize with uniform velocity
void LBM_InitializeFlow(LBMGrid *grid, float ux, float uy, float uz);

// Run one LBM step (collision + streaming)
void LBM_Step(LBMGrid *grid, float inletVelX, float inletVelY, float inletVelZ);

// Get velocity buffer for particle shader to sample
GLuint LBM_GetVelocityBuffer(LBMGrid *grid);

// Compute drag force on solid (momentum exchange)
void LBM_ComputeDragForce(LBMGrid *grid,
                          float *forceX,
                          float *forceY,
                          float *forceZ);

// Compute drag coefficient
float LBM_ComputeDragCoefficient(LBMGrid *grid,
                                 float inletVelocity,
                                 float refArea);

// Compute lift coefficient
float LBM_ComputeLiftCoefficient(LBMGrid *grid,
                                 float inletVelocity,
                                 float refArea);

// Set solid mesh
void LBM_SetSolidMesh(LBMGrid *grid,
                      float *triangles,
                      int numTriangles,
                      float minX,
                      float minY,
                      float minZ,
                      float maxX,
                      float maxY,
                      float maxZ);
#endif // LBM_H