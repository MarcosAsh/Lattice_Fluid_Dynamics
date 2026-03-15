#include "../lib/lbm.h"
#include "../lib/opengl_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// D3Q19 weights for initialization
static const float w[19] = {1.0f / 3.0f,
                            1.0f / 18.0f,
                            1.0f / 18.0f,
                            1.0f / 18.0f,
                            1.0f / 18.0f,
                            1.0f / 18.0f,
                            1.0f / 18.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f,
                            1.0f / 36.0f};

// D3Q19 velocity directions
static const int ex[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
static const int ey[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1};
static const int ez[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1};

static float feq(int i, float rho, float ux, float uy, float uz) {
    float eu = ex[i] * ux + ey[i] * uy + ez[i] * uz;
    float u2 = ux * ux + uy * uy + uz * uz;
    float cs2 = 1.0f / 3.0f;
    return w[i] * rho *
           (1.0f + eu / cs2 + (eu * eu) / (2.0f * cs2 * cs2) -
            u2 / (2.0f * cs2));
}

LBMGrid *LBM_Create(int sizeX, int sizeY, int sizeZ, float viscosity) {
    LBMGrid *grid = (LBMGrid *)malloc(sizeof(LBMGrid));
    if (!grid)
        return NULL;

    grid->sizeX = sizeX;
    grid->sizeY = sizeY;
    grid->sizeZ = sizeZ;
    grid->totalCells = sizeX * sizeY * sizeZ;

    grid->tau = 0.5f + 3.0f * viscosity;
    if (grid->tau < 0.51f)
        grid->tau = 0.51f;
    if (grid->tau > 2.0f)
        grid->tau = 2.0f;

    printf("LBM Grid: %dx%dx%d (%d cells), tau=%.3f\n",
           sizeX,
           sizeY,
           sizeZ,
           grid->totalCells,
           grid->tau);

    // Allocate GPU buffers
    size_t fSize = 19 * grid->totalCells * sizeof(float);
    size_t velSize = grid->totalCells * 4 * sizeof(float);
    size_t solidSize = grid->totalCells * sizeof(int);
    size_t forceSize =
        4 * sizeof(int); // forceX, forceY, forceZ, count (as ints)

    glGenBuffers(1, &grid->fBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->fBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, fSize, NULL, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &grid->fNewBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->fNewBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, fSize, NULL, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &grid->velocityBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->velocityBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, velSize, NULL, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &grid->solidBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, solidSize, NULL, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &grid->forceBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->forceBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, forceSize, NULL, GL_DYNAMIC_DRAW);

    // Initialize solid mask to all fluid
    int *solidData = (int *)calloc(grid->totalCells, sizeof(int));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, solidSize, solidData);
    free(solidData);

    // Load shaders
    grid->collideShader = createComputeShader("shaders/lbm_collide.comp");
    grid->streamShader = createComputeShader("shaders/lbm_stream.comp");
    grid->forceShader = createComputeShader("shaders/lbm_force.comp");

    if (!grid->collideShader || !grid->streamShader) {
        printf("Failed to create LBM shaders!\n");
        LBM_Free(grid);
        return NULL;
    }

    if (!grid->forceShader) {
        printf("Warning: Force shader not loaded, drag calculation disabled\n");
    }

    grid->useRegularized = 0;
    grid->useSmagorinsky = 0;
    grid->smagorinskyCs = 0.1f;
    grid->periodicYZ = 0;

    // Get uniform locations
    glUseProgram(grid->collideShader);
    grid->collide_gridSizeLoc =
        glGetUniformLocation(grid->collideShader, "gridSize");
    grid->collide_tauLoc = glGetUniformLocation(grid->collideShader, "tau");
    grid->collide_inletVelLoc =
        glGetUniformLocation(grid->collideShader, "inletVelocity");
    grid->collide_useRegularizedLoc =
        glGetUniformLocation(grid->collideShader, "useRegularized");
    grid->collide_useSmagorinskyLoc =
        glGetUniformLocation(grid->collideShader, "useSmagorinsky");
    grid->collide_smaCsLoc =
        glGetUniformLocation(grid->collideShader, "smagorinskyCs");

    glUseProgram(grid->streamShader);
    grid->stream_gridSizeLoc =
        glGetUniformLocation(grid->streamShader, "gridSize");
    grid->stream_periodicYZLoc =
        glGetUniformLocation(grid->streamShader, "periodicYZ");

    if (grid->forceShader) {
        glUseProgram(grid->forceShader);
        grid->force_gridSizeLoc =
            glGetUniformLocation(grid->forceShader, "gridSize");
    }

    printf("LBM initialized successfully\n");
    return grid;
}

void LBM_Free(LBMGrid *grid) {
    if (!grid)
        return;

    if (grid->fBuffer)
        glDeleteBuffers(1, &grid->fBuffer);
    if (grid->fNewBuffer)
        glDeleteBuffers(1, &grid->fNewBuffer);
    if (grid->velocityBuffer)
        glDeleteBuffers(1, &grid->velocityBuffer);
    if (grid->solidBuffer)
        glDeleteBuffers(1, &grid->solidBuffer);
    if (grid->forceBuffer)
        glDeleteBuffers(1, &grid->forceBuffer);
    if (grid->collideShader)
        glDeleteProgram(grid->collideShader);
    if (grid->streamShader)
        glDeleteProgram(grid->streamShader);
    if (grid->forceShader)
        glDeleteProgram(grid->forceShader);

    free(grid);
}

void LBM_SetSolidAABB(LBMGrid *grid,
                      float minX,
                      float minY,
                      float minZ,
                      float maxX,
                      float maxY,
                      float maxZ) {
    float scaleX = grid->sizeX / 8.0f;
    float scaleY = grid->sizeY / 4.0f;
    float scaleZ = grid->sizeZ / 4.0f;

    int gMinX = (int)((minX + 4.0f) * scaleX);
    int gMaxX = (int)((maxX + 4.0f) * scaleX);
    int gMinY = (int)((minY + 2.0f) * scaleY);
    int gMaxY = (int)((maxY + 2.0f) * scaleY);
    int gMinZ = (int)((minZ + 2.0f) * scaleZ);
    int gMaxZ = (int)((maxZ + 2.0f) * scaleZ);

    if (gMinX < 0)
        gMinX = 0;
    if (gMaxX >= grid->sizeX)
        gMaxX = grid->sizeX - 1;
    if (gMinY < 0)
        gMinY = 0;
    if (gMaxY >= grid->sizeY)
        gMaxY = grid->sizeY - 1;
    if (gMinZ < 0)
        gMinZ = 0;
    if (gMaxZ >= grid->sizeZ)
        gMaxZ = grid->sizeZ - 1;

    printf("LBM solid AABB: grid [%d-%d, %d-%d, %d-%d]\n",
           gMinX,
           gMaxX,
           gMinY,
           gMaxY,
           gMinZ,
           gMaxZ);

    int *solidData = (int *)calloc(grid->totalCells, sizeof(int));

    for (int z = gMinZ; z <= gMaxZ; z++) {
        for (int y = gMinY; y <= gMaxY; y++) {
            for (int x = gMinX; x <= gMaxX; x++) {
                int idx = x + y * grid->sizeX + z * grid->sizeX * grid->sizeY;
                solidData[idx] = 1;
            }
        }
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 0, grid->totalCells * sizeof(int), solidData);
    free(solidData);
}

void LBM_InitializeFlow(LBMGrid *grid, float ux, float uy, float uz) {
    float rho = 1.0f;

    float *fData = (float *)malloc(19 * grid->totalCells * sizeof(float));
    float *velData = (float *)malloc(4 * grid->totalCells * sizeof(float));

    for (int idx = 0; idx < grid->totalCells; idx++) {
        for (int i = 0; i < 19; i++) {
            fData[idx * 19 + i] = feq(i, rho, ux, uy, uz);
        }
        velData[idx * 4 + 0] = ux;
        velData[idx * 4 + 1] = uy;
        velData[idx * 4 + 2] = uz;
        velData[idx * 4 + 3] = rho;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->fBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                    0,
                    19 * grid->totalCells * sizeof(float),
                    fData);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->fNewBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                    0,
                    19 * grid->totalCells * sizeof(float),
                    fData);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->velocityBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                    0,
                    4 * grid->totalCells * sizeof(float),
                    velData);

    free(fData);
    free(velData);

    printf("LBM flow initialized: u=(%.3f, %.3f, %.3f)\n", ux, uy, uz);
}

void LBM_Step(LBMGrid *grid,
              float inletVelX,
              float inletVelY,
              float inletVelZ) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, grid->fBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, grid->fNewBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, grid->velocityBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, grid->solidBuffer);

    // Collision step
    glUseProgram(grid->collideShader);
    glUniform3i(
        grid->collide_gridSizeLoc, grid->sizeX, grid->sizeY, grid->sizeZ);
    glUniform1f(grid->collide_tauLoc, grid->tau);
    glUniform3f(grid->collide_inletVelLoc, inletVelX, inletVelY, inletVelZ);
    glUniform1i(grid->collide_useRegularizedLoc, grid->useRegularized);
    glUniform1i(grid->collide_useSmagorinskyLoc, grid->useSmagorinsky);
    glUniform1f(grid->collide_smaCsLoc, grid->smagorinskyCs);

    glDispatchCompute(
        (grid->sizeX + 7) / 8, (grid->sizeY + 7) / 8, (grid->sizeZ + 7) / 8);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Streaming step
    glUseProgram(grid->streamShader);
    glUniform3i(
        grid->stream_gridSizeLoc, grid->sizeX, grid->sizeY, grid->sizeZ);
    glUniform1i(grid->stream_periodicYZLoc, grid->periodicYZ);

    glDispatchCompute(
        (grid->sizeX + 7) / 8, (grid->sizeY + 7) / 8, (grid->sizeZ + 7) / 8);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

GLuint LBM_GetVelocityBuffer(LBMGrid *grid) {
    return grid->velocityBuffer;
}

void LBM_ComputeDragForce(LBMGrid *grid,
                          float *forceX,
                          float *forceY,
                          float *forceZ) {
    *forceX = 0.0f;
    *forceY = 0.0f;
    *forceZ = 0.0f;

    if (!grid->forceShader)
        return;

    // Clear force buffer
    int zeros[4] = {0, 0, 0, 0};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->forceBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(zeros), zeros);

    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, grid->fBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, grid->solidBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, grid->forceBuffer);

    // Run force computation
    glUseProgram(grid->forceShader);
    glUniform3i(grid->force_gridSizeLoc, grid->sizeX, grid->sizeY, grid->sizeZ);

    glDispatchCompute(
        (grid->sizeX + 7) / 8, (grid->sizeY + 7) / 8, (grid->sizeZ + 7) / 8);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Read back results
    int results[4];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->forceBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(results), results);

    // Convert back from int (scaled by 10000)
    *forceX = results[0] / 10000.0f;
    *forceY = results[1] / 10000.0f;
    *forceZ = results[2] / 10000.0f;
}

float LBM_ComputeDragCoefficient(LBMGrid *grid,
                                 float inletVelocity,
                                 float refArea) {
    float fx, fy, fz;
    LBM_ComputeDragForce(grid, &fx, &fy, &fz);

    // Drag force is in x direction (streamwise)
    float dragForce = fabsf(fx);

    // Cd = F / (0.5 * rho * U^2 * A)
    // In lattice units, rho = 1
    float rho = 1.0f;
    float dynamicPressure = 0.5f * rho * inletVelocity * inletVelocity;

    if (dynamicPressure * refArea < 1e-10f)
        return 0.0f;

    float Cd = dragForce / (dynamicPressure * refArea);

    return Cd;
}

float LBM_ComputeLiftCoefficient(LBMGrid *grid,
                                 float inletVelocity,
                                 float refArea) {
    float fx, fy, fz;
    LBM_ComputeDragForce(grid, &fx, &fy, &fz);

    // Lift force is in y direction (vertical)
    float liftForce = fabsf(fy);

    // Cl = F / (0.5 * rho * U^2 * A)
    // In lattice units, rho = 1
    float rho = 1.0f;
    float dynamicPressure = 0.5f * rho * inletVelocity * inletVelocity;

    if (dynamicPressure * refArea < 1e-10f)
        return 0.0f;

    float Cl = liftForce / (dynamicPressure * refArea);

    return Cl;
}

void LBM_SetSolidMesh(LBMGrid *grid,
                      float *triangles,
                      int numTriangles,
                      float minX,
                      float minY,
                      float minZ,
                      float maxX,
                      float maxY,
                      float maxZ) {

    int *solidData = (int *)calloc(grid->totalCells, sizeof(int));

    // Precompute simple bounds per triangle so most tests can be skipped
    // quickly.
    typedef struct {
        float minY, maxY;
        float minZ, maxZ;
        float maxX;
    } TriBounds;

    TriBounds *triBounds =
        (TriBounds *)malloc(numTriangles * sizeof(TriBounds));
    if (!triBounds) {
        printf("Failed to allocate triangle bounds; falling back to simple "
               "mesh test\n");
    } else {
        for (int t = 0; t < numTriangles; t++) {
            float *tri = &triangles[t * 12];
            float v0y = tri[1], v0z = tri[2], v0x = tri[0];
            float v1y = tri[5], v1z = tri[6], v1x = tri[4];
            float v2y = tri[9], v2z = tri[10], v2x = tri[8];

            float minYt = fminf(v0y, fminf(v1y, v2y));
            float maxYt = fmaxf(v0y, fmaxf(v1y, v2y));
            float minZt = fminf(v0z, fminf(v1z, v2z));
            float maxZt = fmaxf(v0z, fmaxf(v1z, v2z));
            float maxXt = fmaxf(v0x, fmaxf(v1x, v2x));

            triBounds[t].minY = minYt;
            triBounds[t].maxY = maxYt;
            triBounds[t].minZ = minZt;
            triBounds[t].maxZ = maxZt;
            triBounds[t].maxX = maxXt;
        }
    }

    // World to grid scaling
    float scaleX = grid->sizeX / 8.0f; // world x: -4 to 4
    float scaleY = grid->sizeY / 4.0f; // world y: -2 to 2
    float scaleZ = grid->sizeZ / 4.0f; // world z: -2 to 2

    int solidCount = 0;

    // For each grid cell, check if it's inside the mesh
    for (int gz = 0; gz < grid->sizeZ; gz++) {
        for (int gy = 0; gy < grid->sizeY; gy++) {
            for (int gx = 0; gx < grid->sizeX; gx++) {
                // Grid cell center in world coords
                float wx = (gx + 0.5f) / scaleX - 4.0f;
                float wy = (gy + 0.5f) / scaleY - 2.0f;
                float wz = (gz + 0.5f) / scaleZ - 2.0f;

                // Quick AABB check first
                if (wx < minX || wx > maxX || wy < minY || wy > maxY ||
                    wz < minZ || wz > maxZ) {
                    continue;
                }

                // Ray casting to check if inside mesh
                // Cast ray in +X direction, count intersections
                int intersections = 0;

                for (int t = 0; t < numTriangles; t++) {
                    if (triBounds) {
                        TriBounds tb = triBounds[t];
                        if (wy < tb.minY || wy > tb.maxY || wz < tb.minZ ||
                            wz > tb.maxZ || wx > tb.maxX) {
                            continue;
                        }
                    }

                    float *tri =
                        &triangles[t * 12]; // 3 verts * 4 floats (with padding)

                    float v0x = tri[0], v0y = tri[1], v0z = tri[2];
                    float v1x = tri[4], v1y = tri[5], v1z = tri[6];
                    float v2x = tri[8], v2y = tri[9], v2z = tri[10];

                    // Ray-triangle intersection (Möller–Trumbore)
                    float e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
                    float e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

                    // Ray direction is (1, 0, 0)
                    float hx = 0, hy = -e2z, hz = e2y;
                    float a = e1x * hx + e1y * hy + e1z * hz;

                    if (a > -0.00001f && a < 0.00001f)
                        continue;

                    float f = 1.0f / a;
                    float sx = wx - v0x, sy = wy - v0y, sz = wz - v0z;
                    float u = f * (sx * hx + sy * hy + sz * hz);

                    if (u < 0.0f || u > 1.0f)
                        continue;

                    float qx = sy * e1z - sz * e1y;
                    float qy = sz * e1x - sx * e1z;
                    float qz = sx * e1y - sy * e1x;

                    float v = f * qx; // ray dir dot q, but ray dir is (1,0,0)

                    if (v < 0.0f || u + v > 1.0f)
                        continue;

                    float dist = f * (e2x * qx + e2y * qy + e2z * qz);

                    if (dist > 0.00001f) {
                        intersections++;
                    }
                }

                // Odd number of intersections = inside
                if (intersections % 2 == 1) {
                    int idx =
                        gx + gy * grid->sizeX + gz * grid->sizeX * grid->sizeY;
                    solidData[idx] = 1;
                    solidCount++;
                }
            }
        }
    }

    printf("LBM mesh solid: %d cells marked as solid\n", solidCount);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 0, grid->totalCells * sizeof(int), solidData);
    free(solidData);
    if (triBounds)
        free(triBounds);
}
