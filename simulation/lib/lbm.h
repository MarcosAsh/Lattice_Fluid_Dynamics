#ifndef LBM_H
#define LBM_H

#include <glad/gl.h>

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
    GLuint qBuffer;     // Bouzidi fractional wall distances

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

    int useRegularized;  // 0 = BGK, 1 = regularized
    int useMRT;          // 0 = off, 1 = MRT collision operator
    int useSmagorinsky;  // 0 = off, 1 = Smagorinsky SGS
    float smagorinskyCs; // Smagorinsky constant
    int periodicYZ;      // 0 = clamp, 1 = periodic y/z

    GLint collide_useMRTLoc;

    // Z-slab chunking for large grids (SSBO > 128 MB)
    int numChunks;       // number of Z-slabs (1 = no splitting)
    int slabZ;           // owned Z-cells per slab
    int haloZ;           // halo cells per side (1 for D3Q19)
    GLint64 maxSSBOSize; // GL_MAX_SHADER_STORAGE_BLOCK_SIZE

    GLint collide_zOffsetLoc;
    GLint stream_zOffsetLoc;
    GLint stream_fNewZOffsetLoc;
    GLint stream_slabZLoc;
    GLint force_zOffsetLoc;
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

// Initialize with uniform velocity. Returns 1 on success, 0 on failure.
int LBM_InitializeFlow(LBMGrid *grid, float ux, float uy, float uz);

// Run one LBM step (collision + streaming)
void LBM_Step(LBMGrid *grid, float inletVelX, float inletVelY, float inletVelZ);

// Get velocity buffer for particle shader to sample
GLuint LBM_GetVelocityBuffer(LBMGrid *grid);

// Compute drag force on solid (momentum exchange)
void LBM_ComputeDragForce(LBMGrid *grid,
                          float *forceX,
                          float *forceY,
                          float *forceZ);

// Compute drag force with pressure/friction decomposition.
// Pressure components may be NULL if not needed.
void LBM_ComputeDragForceDecomposed(LBMGrid *grid,
                                    float *forceX,
                                    float *forceY,
                                    float *forceZ,
                                    float *pressureX,
                                    float *pressureY,
                                    float *pressureZ);

// Compute drag coefficient
float LBM_ComputeDragCoefficient(LBMGrid *grid,
                                 float inletVelocity,
                                 float refArea);

// Compute lift coefficient
float LBM_ComputeLiftCoefficient(LBMGrid *grid,
                                 float inletVelocity,
                                 float refArea);

// Set solid sphere (world coordinates, analytic Bouzidi q)
void LBM_SetSolidSphere(
    LBMGrid *grid, float cx, float cy, float cz, float radius);

// Add a ground plane at worldZ (merges with existing solid buffer).
// Ground cells are marked solid=2 so the force shader can exclude them.
void LBM_AddGroundPlane(LBMGrid *grid, float worldZ);

// Compute the projected frontal area (lattice units^2) of solid==1 cells
// onto the plane perpendicular to the given axis (0=x, 1=y, 2=z).
float LBM_ComputeProjectedArea(LBMGrid *grid, int axis);

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