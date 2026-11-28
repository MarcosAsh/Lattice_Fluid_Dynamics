#ifndef LBM_H
#define LBM_H

#include <glad/gl.h>

// Floats per force workgroup: force, pressure, weight terms (vec4 each)
#define LBM_FORCE_FLOATS 12

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
    int useEntropic;     // 0 = off, 1 = entropic (ELBM) collision
    int useRecursive;    // 1 = recursive regularized collision
    int useSmagorinsky;  // 0 = off, 1 = Smagorinsky SGS
    float smagorinskyCs; // Smagorinsky constant
    int periodicYZ;      // 0 = clamp, 1 = periodic y/z

    float turbulenceIntensity; // inlet TI fraction (0 = uniform inlet)
    int stepCount;             // total LBM steps taken (drives inlet noise)

    // Time-averaged force accumulation: when enabled, LBM_Step
    // dispatches the force shader every step into the (uncleared)
    // force buffer, and LBM_ComputeDragForce* returns the mean over
    // all steps since the last readback instead of an instantaneous
    // snapshot. Kills the sample-to-sample momentum-exchange noise
    // that dominates Cd on coarse grids.
    int forceAveraging;  // 0 = instantaneous (default), 1 = accumulate
    int forceAccumCount; // steps accumulated since last readback

    // Outlet sponge layer (acoustic damping zone)
    int spongeCells;    // width in cells before the outlet (0 = off)
    float spongeTauMax; // relaxation time at the outlet end

    GLint collide_useMRTLoc;
    GLint collide_useEntropicLoc;
    GLint collide_useRecursiveLoc;
    GLint collide_turbIntensityLoc;
    GLint collide_timestepLoc;
    GLint collide_fullGridZLoc;
    GLint collide_spongeCellsLoc;
    GLint collide_spongeTauMaxLoc;

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

    int periodicX;      // 1 = periodic x, disables inlet and outlet
    float bodyForce[3]; // uniform body force density, lattice units
    GLuint diagBuffer;  // stability guard counters

    GLint collide_periodicXLoc;
    GLint collide_bodyForceLoc;
    GLint stream_periodicXLoc;
    GLint stream_fNewZCountLoc;
    GLint stream_wrapPlaneZLoc;

    int forceWorkgroups; // force buffer holds 3 vec4 per workgroup
    GLint force_wgBaseLoc;
    GLsync stepFence; // bounds the depth of the queued step work
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

// Enable/disable per-step force accumulation (time-averaged drag).
void LBM_SetForceAveraging(LBMGrid *grid, int enable);

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

// Exact y-z silhouette area of a world-space mesh, in lattice cells^2
float LBM_MeshFrontalArea(LBMGrid *grid,
                          const float *triangles,
                          int numTriangles);

// Cells reset by the stability guard since the last call, then clears
void LBM_ReadDiagnostics(LBMGrid *grid, int *resetCells, int *nanCells);

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