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
    LBMGrid *grid = (LBMGrid *)calloc(1, sizeof(LBMGrid));
    if (!grid)
        return NULL;

    grid->sizeX = sizeX;
    grid->sizeY = sizeY;
    grid->sizeZ = sizeZ;
    grid->totalCells = sizeX * sizeY * sizeZ;

    grid->tau = 0.5f + 3.0f * viscosity;
    if (grid->tau < 0.52f)
        grid->tau = 0.52f;
    if (grid->tau > 2.0f)
        grid->tau = 2.0f;

    printf("LBM Grid: %dx%dx%d (%d cells), tau=%.3f\n",
           sizeX,
           sizeY,
           sizeZ,
           grid->totalCells,
           grid->tau);

    // Query GPU limits for validation
    GLint64 maxSSBOSize = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxSSBOSize);
    GLint maxComputeWG[3] = {0, 0, 0};
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxComputeWG[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxComputeWG[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxComputeWG[2]);

    printf("GPU limits: SSBO=%.0f MB, dispatch=%d x %d x %d\n",
           maxSSBOSize / (1024.0 * 1024.0),
           maxComputeWG[0],
           maxComputeWG[1],
           maxComputeWG[2]);

    // Allocate GPU buffers
    size_t fSize = 19 * grid->totalCells * sizeof(float);
    size_t velSize = grid->totalCells * 4 * sizeof(float);
    size_t solidSize = grid->totalCells * sizeof(int);
    size_t forceSize = 7 * sizeof(int); // total(xyz), count, pressure(xyz)

    size_t totalGPU = 2 * fSize + velSize + solidSize + forceSize;
    printf("GPU memory: f=%.1f MB x2, vel=%.1f MB, solid=%.1f MB, "
           "total=%.1f MB\n",
           fSize / (1024.0 * 1024.0),
           velSize / (1024.0 * 1024.0),
           solidSize / (1024.0 * 1024.0),
           totalGPU / (1024.0 * 1024.0));

    // Compute Z-slab chunking for SSBO binding limit
    grid->maxSSBOSize = maxSSBOSize;
    grid->haloZ = 1; // D3Q19 streaming reads at most 1 cell away
    size_t fSliceBytes = (size_t)19 * sizeX * sizeY * sizeof(float);
    if (maxSSBOSize > 0 && fSliceBytes > 0) {
        int maxZPerChunk = (int)(maxSSBOSize / fSliceBytes);
        if (maxZPerChunk >= sizeZ) {
            grid->numChunks = 1;
            grid->slabZ = sizeZ;
        } else {
            int effectiveZ = maxZPerChunk - 2 * grid->haloZ;
            if (effectiveZ < 8)
                effectiveZ = 8;
            grid->numChunks = (sizeZ + effectiveZ - 1) / effectiveZ;
            grid->slabZ = effectiveZ;
            printf("SSBO slab split: %d chunks of %d Z-cells "
                   "(limit %.0f MB, slice %.2f MB)\n",
                   grid->numChunks,
                   grid->slabZ,
                   maxSSBOSize / (1024.0 * 1024.0),
                   fSliceBytes / (1024.0 * 1024.0));
        }
    } else {
        grid->numChunks = 1;
        grid->slabZ = sizeZ;
    }

    // Validate dispatch size
    int dispX = (sizeX + 7) / 8;
    int dispY = (sizeY + 7) / 8;
    int dispZ = (sizeZ + 7) / 8;
    if (dispX > maxComputeWG[0] || dispY > maxComputeWG[1] ||
        dispZ > maxComputeWG[2]) {
        printf("ERROR: dispatch %dx%dx%d exceeds GPU limit %dx%dx%d\n",
               dispX,
               dispY,
               dispZ,
               maxComputeWG[0],
               maxComputeWG[1],
               maxComputeWG[2]);
        free(grid);
        return NULL;
    }

    // Drain any prior GL errors
    while (glGetError() != GL_NO_ERROR) {
    }

    // Allocate each SSBO with zero-initialized CPU data passed directly
    // to glBufferData. Passing data (not NULL) forces the NVIDIA driver
    // to fully map every page in the GPU's virtual address space during
    // the DMA transfer. NULL allocations use lazy mapping which causes
    // MMU faults (FAULT_PDE) on T4 GPUs with large buffers.

    void *fZeros = calloc(1, fSize);
    void *velZeros = calloc(1, velSize);
    void *solidZeros = calloc(1, solidSize);
    if (!fZeros || !velZeros || !solidZeros) {
        printf("ERROR: CPU alloc failed for GPU init buffers (%.1f MB)\n",
               (2.0 * fSize + velSize + solidSize) / (1024.0 * 1024.0));
        free(fZeros);
        free(velZeros);
        free(solidZeros);
        free(grid);
        return NULL;
    }

    // Use glBufferStorage (immutable store) for the large f buffers.
    // Unlike glBufferData, immutable storage cannot be orphaned or
    // reallocated, giving the driver stronger page table guarantees.
    glGenBuffers(1, &grid->fBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->fBuffer);
    glBufferStorage(
        GL_SHADER_STORAGE_BUFFER, fSize, fZeros, GL_DYNAMIC_STORAGE_BIT);
    if (glGetError() != GL_NO_ERROR) {
        printf("ERROR: GPU alloc failed for fBuffer (%.1f MB)\n",
               fSize / (1024.0 * 1024.0));
        free(fZeros);
        free(velZeros);
        free(solidZeros);
        LBM_Free(grid);
        return NULL;
    }

    glGenBuffers(1, &grid->fNewBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->fNewBuffer);
    glBufferStorage(
        GL_SHADER_STORAGE_BUFFER, fSize, fZeros, GL_DYNAMIC_STORAGE_BIT);
    if (glGetError() != GL_NO_ERROR) {
        printf("ERROR: GPU alloc failed for fNewBuffer (%.1f MB)\n",
               fSize / (1024.0 * 1024.0));
        free(fZeros);
        free(velZeros);
        free(solidZeros);
        LBM_Free(grid);
        return NULL;
    }
    free(fZeros);

    glGenBuffers(1, &grid->velocityBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->velocityBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, velSize, velZeros, GL_DYNAMIC_COPY);
    if (glGetError() != GL_NO_ERROR) {
        printf("ERROR: GPU alloc failed for velocityBuffer (%.1f MB)\n",
               velSize / (1024.0 * 1024.0));
        free(velZeros);
        free(solidZeros);
        LBM_Free(grid);
        return NULL;
    }
    free(velZeros);

    glGenBuffers(1, &grid->solidBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER, solidSize, solidZeros, GL_STATIC_DRAW);
    if (glGetError() != GL_NO_ERROR) {
        printf("ERROR: GPU alloc failed for solidBuffer (%.1f MB)\n",
               solidSize / (1024.0 * 1024.0));
        free(solidZeros);
        LBM_Free(grid);
        return NULL;
    }
    free(solidZeros);

    int forceZeros[7] = {0, 0, 0, 0, 0, 0, 0};
    glGenBuffers(1, &grid->forceBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->forceBuffer);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER, forceSize, forceZeros, GL_DYNAMIC_COPY);
    if (glGetError() != GL_NO_ERROR) {
        printf("ERROR: GPU alloc failed for forceBuffer\n");
        LBM_Free(grid);
        return NULL;
    }

    // Bouzidi q-value buffer: 19 floats per cell, initialized to -1
    // (sentinel = no boundary link). Filled by LBM_SetSolidMesh.
    size_t qSize = (size_t)19 * grid->totalCells * sizeof(float);
    float *qInit = (float *)malloc(qSize);
    if (qInit) {
        for (size_t i = 0; i < (size_t)19 * grid->totalCells; i++)
            qInit[i] = -1.0f;
    }
    glGenBuffers(1, &grid->qBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->qBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, qSize, qInit, GL_STATIC_DRAW);
    free(qInit);
    if (glGetError() != GL_NO_ERROR) {
        printf("ERROR: GPU alloc failed for qBuffer (%.1f MB)\n",
               qSize / (1024.0 * 1024.0));
        LBM_Free(grid);
        return NULL;
    }

    // Force GPU to finish all pending DMA transfers before any compute
    // dispatch. Without this, NVIDIA T4 can evict freshly uploaded pages.
    glFinish();
    printf("GPU buffers allocated and committed\n");

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
    grid->useMRT = 0;
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
    grid->collide_useMRTLoc =
        glGetUniformLocation(grid->collideShader, "useMRT");
    grid->collide_zOffsetLoc =
        glGetUniformLocation(grid->collideShader, "zOffset");

    glUseProgram(grid->streamShader);
    grid->stream_gridSizeLoc =
        glGetUniformLocation(grid->streamShader, "gridSize");
    grid->stream_periodicYZLoc =
        glGetUniformLocation(grid->streamShader, "periodicYZ");
    grid->stream_zOffsetLoc =
        glGetUniformLocation(grid->streamShader, "zOffset");
    grid->stream_fNewZOffsetLoc =
        glGetUniformLocation(grid->streamShader, "fNewZOffset");
    grid->stream_slabZLoc =
        glGetUniformLocation(grid->streamShader, "slabZ");

    if (grid->forceShader) {
        glUseProgram(grid->forceShader);
        grid->force_gridSizeLoc =
            glGetUniformLocation(grid->forceShader, "gridSize");
        grid->force_zOffsetLoc =
            glGetUniformLocation(grid->forceShader, "zOffset");
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
    if (grid->qBuffer)
        glDeleteBuffers(1, &grid->qBuffer);
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

    // Set Bouzidi q = 0.5 (midpoint) for all boundary links.
    // This is equivalent to standard bounce-back but lets the force
    // shader find the links via q_val >= 0.
    size_t qCount = (size_t)19 * grid->totalCells;
    float *qData = (float *)malloc(qCount * sizeof(float));
    for (size_t qi = 0; qi < qCount; qi++)
        qData[qi] = -1.0f;

    int qLinks = 0;
    for (int gz = 0; gz < grid->sizeZ; gz++) {
        for (int gy = 0; gy < grid->sizeY; gy++) {
            for (int gx = 0; gx < grid->sizeX; gx++) {
                int ci = gx + gy * grid->sizeX + gz * grid->sizeX * grid->sizeY;
                if (solidData[ci] == 1)
                    continue;
                for (int i = 1; i < 19; i++) {
                    int nx = gx + ex[i], ny = gy + ey[i], nz = gz + ez[i];
                    if (nx < 0 || nx >= grid->sizeX || ny < 0 ||
                        ny >= grid->sizeY || nz < 0 || nz >= grid->sizeZ)
                        continue;
                    int ni =
                        nx + ny * grid->sizeX + nz * grid->sizeX * grid->sizeY;
                    if (solidData[ni] == 1) {
                        qData[ci * 19 + i] = 0.5f;
                        qLinks++;
                    }
                }
            }
        }
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->qBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, qCount * sizeof(float), qData);
    free(qData);
    free(solidData);
}

void LBM_SetSolidSphere(
    LBMGrid *grid, float cx, float cy, float cz, float radius) {
    float scaleX = grid->sizeX / 8.0f;
    float scaleY = grid->sizeY / 4.0f;
    float scaleZ = grid->sizeZ / 4.0f;

    int *solidData = (int *)calloc(grid->totalCells, sizeof(int));
    int solidCount = 0;

    for (int gz = 0; gz < grid->sizeZ; gz++) {
        for (int gy = 0; gy < grid->sizeY; gy++) {
            for (int gx = 0; gx < grid->sizeX; gx++) {
                float wx = (gx + 0.5f) / scaleX - 4.0f;
                float wy = (gy + 0.5f) / scaleY - 2.0f;
                float wz = (gz + 0.5f) / scaleZ - 2.0f;
                float dx = wx - cx, dy = wy - cy, dz = wz - cz;
                if (dx * dx + dy * dy + dz * dz <= radius * radius) {
                    int idx =
                        gx + gy * grid->sizeX + gz * grid->sizeX * grid->sizeY;
                    solidData[idx] = 1;
                    solidCount++;
                }
            }
        }
    }

    printf("LBM solid sphere: %d cells (R=%.2f at (%.2f,%.2f,%.2f))\n",
           solidCount,
           radius,
           cx,
           cy,
           cz);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 0, grid->totalCells * sizeof(int), solidData);

    // Bouzidi q via analytic ray-sphere intersection
    size_t qCount = (size_t)19 * grid->totalCells;
    float *qData = (float *)malloc(qCount * sizeof(float));
    for (size_t qi = 0; qi < qCount; qi++)
        qData[qi] = -1.0f;

    int qLinks = 0;
    for (int gz = 0; gz < grid->sizeZ; gz++) {
        for (int gy = 0; gy < grid->sizeY; gy++) {
            for (int gx = 0; gx < grid->sizeX; gx++) {
                int ci = gx + gy * grid->sizeX + gz * grid->sizeX * grid->sizeY;
                if (solidData[ci] == 1)
                    continue;

                float wx = (gx + 0.5f) / scaleX - 4.0f;
                float wy = (gy + 0.5f) / scaleY - 2.0f;
                float wz = (gz + 0.5f) / scaleZ - 2.0f;

                for (int i = 1; i < 19; i++) {
                    int nx = gx + ex[i], ny = gy + ey[i], nz = gz + ez[i];
                    if (nx < 0 || nx >= grid->sizeX || ny < 0 ||
                        ny >= grid->sizeY || nz < 0 || nz >= grid->sizeZ)
                        continue;
                    int ni =
                        nx + ny * grid->sizeX + nz * grid->sizeX * grid->sizeY;
                    if (solidData[ni] != 1)
                        continue;

                    // Ray-sphere intersection for exact q
                    float dx = (float)ex[i] / scaleX;
                    float dy = (float)ey[i] / scaleY;
                    float dz = (float)ez[i] / scaleZ;
                    float ox = wx - cx, oy = wy - cy, oz = wz - cz;
                    float a = dx * dx + dy * dy + dz * dz;
                    float b = 2.0f * (ox * dx + oy * dy + oz * dz);
                    float c = ox * ox + oy * oy + oz * oz - radius * radius;
                    float disc = b * b - 4.0f * a * c;

                    float q = 0.5f; // fallback
                    if (disc >= 0.0f) {
                        float t = (-b - sqrtf(disc)) / (2.0f * a);
                        if (t > 0.0f && t <= 1.0f)
                            q = t;
                    }
                    if (q < 0.01f)
                        q = 0.01f;
                    if (q > 0.99f)
                        q = 0.99f;

                    qData[ci * 19 + i] = q;
                    qLinks++;
                }
            }
        }
    }

    printf("Sphere Bouzidi: %d boundary links (analytic q)\n", qLinks);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->qBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, qCount * sizeof(float), qData);
    free(qData);
    free(solidData);
}

void LBM_AddGroundPlane(LBMGrid *grid, float worldZ) {
    float scaleZ = grid->sizeZ / 4.0f;
    int gzGround = (int)((worldZ + 2.0f) * scaleZ);
    if (gzGround < 0)
        gzGround = 0;
    if (gzGround >= grid->sizeZ)
        gzGround = grid->sizeZ - 1;

    // Read back existing buffers
    int *solidData = (int *)malloc(grid->totalCells * sizeof(int));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 0, grid->totalCells * sizeof(int), solidData);

    size_t qCount = (size_t)19 * grid->totalCells;
    float *qData = (float *)malloc(qCount * sizeof(float));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->qBuffer);
    glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 0, qCount * sizeof(float), qData);

    // Mark ground cells as solid=2 (preserve existing body=1)
    int groundCells = 0;
    for (int gz = 0; gz <= gzGround; gz++) {
        for (int gy = 0; gy < grid->sizeY; gy++) {
            for (int gx = 0; gx < grid->sizeX; gx++) {
                int idx =
                    gx + gy * grid->sizeX + gz * grid->sizeX * grid->sizeY;
                if (solidData[idx] == 0) {
                    solidData[idx] = 2;
                    groundCells++;
                }
            }
        }
    }

    // Set Bouzidi q for fluid cells just above ground
    int gzAbove = gzGround + 1;
    int groundLinks = 0;
    if (gzAbove < grid->sizeZ) {
        for (int gy = 0; gy < grid->sizeY; gy++) {
            for (int gx = 0; gx < grid->sizeX; gx++) {
                int ci =
                    gx + gy * grid->sizeX + gzAbove * grid->sizeX * grid->sizeY;
                if (solidData[ci] != 0)
                    continue;
                // Set q for all downward directions (ez[i] == -1)
                for (int i = 1; i < 19; i++) {
                    if (ez[i] != -1)
                        continue;
                    int nz = gzAbove + ez[i]; // = gzGround
                    int ny = gy + ey[i], nx = gx + ex[i];
                    if (nx < 0 || nx >= grid->sizeX || ny < 0 ||
                        ny >= grid->sizeY || nz < 0)
                        continue;
                    int ni =
                        nx + ny * grid->sizeX + nz * grid->sizeX * grid->sizeY;
                    if (solidData[ni] != 0 && qData[ci * 19 + i] < 0.0f) {
                        qData[ci * 19 + i] = 0.5f;
                        groundLinks++;
                    }
                }
            }
        }
    }

    printf("Ground plane: z_lattice=%d, %d cells, %d boundary links\n",
           gzGround,
           groundCells,
           groundLinks);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 0, grid->totalCells * sizeof(int), solidData);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->qBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, qCount * sizeof(float), qData);
    free(solidData);
    free(qData);
}

float LBM_ComputeProjectedArea(LBMGrid *grid, int axis) {
    int *solidData = (int *)malloc(grid->totalCells * sizeof(int));
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 0, grid->totalCells * sizeof(int), solidData);

    int count = 0;
    if (axis == 0) {
        // Project onto Y-Z plane: count columns with any solid==1
        for (int gz = 0; gz < grid->sizeZ; gz++) {
            for (int gy = 0; gy < grid->sizeY; gy++) {
                int found = 0;
                for (int gx = 0; gx < grid->sizeX && !found; gx++) {
                    int idx =
                        gx + gy * grid->sizeX + gz * grid->sizeX * grid->sizeY;
                    if (solidData[idx] == 1)
                        found = 1;
                }
                count += found;
            }
        }
    } else if (axis == 1) {
        for (int gz = 0; gz < grid->sizeZ; gz++) {
            for (int gx = 0; gx < grid->sizeX; gx++) {
                int found = 0;
                for (int gy = 0; gy < grid->sizeY && !found; gy++) {
                    int idx =
                        gx + gy * grid->sizeX + gz * grid->sizeX * grid->sizeY;
                    if (solidData[idx] == 1)
                        found = 1;
                }
                count += found;
            }
        }
    } else {
        for (int gy = 0; gy < grid->sizeY; gy++) {
            for (int gx = 0; gx < grid->sizeX; gx++) {
                int found = 0;
                for (int gz = 0; gz < grid->sizeZ && !found; gz++) {
                    int idx =
                        gx + gy * grid->sizeX + gz * grid->sizeX * grid->sizeY;
                    if (solidData[idx] == 1)
                        found = 1;
                }
                count += found;
            }
        }
    }

    free(solidData);
    return (float)count;
}

int LBM_InitializeFlow(LBMGrid *grid, float ux, float uy, float uz) {
    float rho = 1.0f;

    float *fData = (float *)malloc(19 * grid->totalCells * sizeof(float));
    float *velData = (float *)malloc(4 * grid->totalCells * sizeof(float));

    if (!fData || !velData) {
        fprintf(stderr,
                "ERROR: Failed to allocate CPU buffers for LBM init "
                "(need %.1f MB)\n",
                (19 * grid->totalCells * sizeof(float) +
                 4 * grid->totalCells * sizeof(float)) /
                    (1024.0 * 1024.0));
        free(fData);
        free(velData);
        return 0;
    }

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
    return 1;
}

void LBM_Step(LBMGrid *grid,
              float inletVelX,
              float inletVelY,
              float inletVelZ) {
    // Bind unsplit buffers once
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, grid->velocityBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, grid->solidBuffer);

    size_t fSliceBytes =
        (size_t)19 * grid->sizeX * grid->sizeY * sizeof(float);
    int dispX = (grid->sizeX + 7) / 8;
    int dispY = (grid->sizeY + 7) / 8;

    // Collision: per-slab dispatch
    glUseProgram(grid->collideShader);
    glUniform1f(grid->collide_tauLoc, grid->tau);
    glUniform3f(grid->collide_inletVelLoc, inletVelX, inletVelY, inletVelZ);
    glUniform1i(grid->collide_useRegularizedLoc, grid->useRegularized);
    glUniform1i(grid->collide_useSmagorinskyLoc, grid->useSmagorinsky);
    glUniform1f(grid->collide_smaCsLoc, grid->smagorinskyCs);
    glUniform1i(grid->collide_useMRTLoc, grid->useMRT);

    for (int c = 0; c < grid->numChunks; c++) {
        int zStart = c * grid->slabZ;
        int zEnd = zStart + grid->slabZ;
        if (zEnd > grid->sizeZ)
            zEnd = grid->sizeZ;
        int slabCells = zEnd - zStart;

        if (grid->numChunks == 1) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, grid->fBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, grid->fNewBuffer);
        } else {
            GLintptr off = (GLintptr)(zStart * fSliceBytes);
            GLsizeiptr sz = (GLsizeiptr)(slabCells * fSliceBytes);
            glBindBufferRange(
                GL_SHADER_STORAGE_BUFFER, 2, grid->fBuffer, off, sz);
            glBindBufferRange(
                GL_SHADER_STORAGE_BUFFER, 3, grid->fNewBuffer, off, sz);
        }

        glUniform3i(
            grid->collide_gridSizeLoc, grid->sizeX, grid->sizeY, slabCells);
        glUniform1i(grid->collide_zOffsetLoc, zStart);

        glDispatchCompute(dispX, dispY, (slabCells + 7) / 8);
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Streaming: per-slab dispatch with halo
    glUseProgram(grid->streamShader);
    glUniform3i(
        grid->stream_gridSizeLoc, grid->sizeX, grid->sizeY, grid->sizeZ);
    glUniform1i(grid->stream_periodicYZLoc, grid->periodicYZ);

    for (int c = 0; c < grid->numChunks; c++) {
        int zStart = c * grid->slabZ;
        int zEnd = zStart + grid->slabZ;
        if (zEnd > grid->sizeZ)
            zEnd = grid->sizeZ;
        int slabCells = zEnd - zStart;

        int fNewZOff = 0;
        if (grid->numChunks == 1) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, grid->fBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, grid->fNewBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, grid->qBuffer);
        } else {
            // f[] and q[]: slab-only range (write target)
            GLintptr wOff = (GLintptr)(zStart * fSliceBytes);
            GLsizeiptr wSz = (GLsizeiptr)(slabCells * fSliceBytes);
            glBindBufferRange(
                GL_SHADER_STORAGE_BUFFER, 2, grid->fBuffer, wOff, wSz);
            glBindBufferRange(
                GL_SHADER_STORAGE_BUFFER, 7, grid->qBuffer, wOff, wSz);

            // fNew[]: slab + halo range (read source)
            int rStart = zStart - grid->haloZ;
            if (rStart < 0)
                rStart = 0;
            int rEnd = zEnd + grid->haloZ;
            if (rEnd > grid->sizeZ)
                rEnd = grid->sizeZ;
            GLintptr rOff = (GLintptr)(rStart * fSliceBytes);
            GLsizeiptr rSz = (GLsizeiptr)((rEnd - rStart) * fSliceBytes);
            glBindBufferRange(
                GL_SHADER_STORAGE_BUFFER, 3, grid->fNewBuffer, rOff, rSz);

            fNewZOff = rStart;
        }

        glUniform1i(grid->stream_zOffsetLoc, zStart);
        glUniform1i(grid->stream_fNewZOffsetLoc, fNewZOff);
        glUniform1i(grid->stream_slabZLoc, slabCells);

        glDispatchCompute(dispX, dispY, (slabCells + 7) / 8);
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

GLuint LBM_GetVelocityBuffer(LBMGrid *grid) {
    return grid->velocityBuffer;
}

void LBM_ComputeDragForce(LBMGrid *grid,
                          float *forceX,
                          float *forceY,
                          float *forceZ) {
    LBM_ComputeDragForceDecomposed(
        grid, forceX, forceY, forceZ, NULL, NULL, NULL);
}

void LBM_ComputeDragForceDecomposed(LBMGrid *grid,
                                    float *forceX,
                                    float *forceY,
                                    float *forceZ,
                                    float *pressureX,
                                    float *pressureY,
                                    float *pressureZ) {
    *forceX = 0.0f;
    *forceY = 0.0f;
    *forceZ = 0.0f;
    if (pressureX)
        *pressureX = 0.0f;
    if (pressureY)
        *pressureY = 0.0f;
    if (pressureZ)
        *pressureZ = 0.0f;

    if (!grid->forceShader)
        return;

    // Clear force buffer (7 ints: total xyz, count, pressure xyz)
    int zeros[7] = {0, 0, 0, 0, 0, 0, 0};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->forceBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(zeros), zeros);

    // Bind unsplit buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, grid->velocityBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, grid->solidBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, grid->forceBuffer);

    glUseProgram(grid->forceShader);

    size_t fSliceBytes =
        (size_t)19 * grid->sizeX * grid->sizeY * sizeof(float);
    int dispX = (grid->sizeX + 7) / 8;
    int dispY = (grid->sizeY + 7) / 8;

    for (int c = 0; c < grid->numChunks; c++) {
        int zStart = c * grid->slabZ;
        int zEnd = zStart + grid->slabZ;
        if (zEnd > grid->sizeZ)
            zEnd = grid->sizeZ;
        int slabCells = zEnd - zStart;

        if (grid->numChunks == 1) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, grid->fBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, grid->fNewBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, grid->qBuffer);
        } else {
            GLintptr off = (GLintptr)(zStart * fSliceBytes);
            GLsizeiptr sz = (GLsizeiptr)(slabCells * fSliceBytes);
            glBindBufferRange(
                GL_SHADER_STORAGE_BUFFER, 2, grid->fBuffer, off, sz);
            glBindBufferRange(
                GL_SHADER_STORAGE_BUFFER, 3, grid->fNewBuffer, off, sz);
            glBindBufferRange(
                GL_SHADER_STORAGE_BUFFER, 7, grid->qBuffer, off, sz);
        }

        glUniform3i(
            grid->force_gridSizeLoc, grid->sizeX, grid->sizeY, slabCells);
        glUniform1i(grid->force_zOffsetLoc, zStart);

        glDispatchCompute(dispX, dispY, (slabCells + 7) / 8);
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                    GL_BUFFER_UPDATE_BARRIER_BIT);

    // Read back results
    int results[7];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->forceBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(results), results);

    // Convert back from int (scaled by 10000)
    *forceX = results[0] / 10000.0f;
    *forceY = results[1] / 10000.0f;
    *forceZ = results[2] / 10000.0f;
    if (pressureX)
        *pressureX = results[4] / 10000.0f;
    if (pressureY)
        *pressureY = results[5] / 10000.0f;
    if (pressureZ)
        *pressureZ = results[6] / 10000.0f;
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

    // Compute Bouzidi q values: for each fluid cell with a solid
    // neighbor, ray-cast along the lattice link to find the fractional
    // distance to the actual mesh surface.
    size_t qCount = (size_t)19 * grid->totalCells;
    float *qData = (float *)malloc(qCount * sizeof(float));
    for (size_t qi = 0; qi < qCount; qi++)
        qData[qi] = -1.0f;

    int qLinks = 0;

    for (int gz = 0; gz < grid->sizeZ; gz++) {
        for (int gy = 0; gy < grid->sizeY; gy++) {
            for (int gx = 0; gx < grid->sizeX; gx++) {
                int cellIdx =
                    gx + gy * grid->sizeX + gz * grid->sizeX * grid->sizeY;
                if (solidData[cellIdx] == 1)
                    continue; // only fluid cells

                float wx = (gx + 0.5f) / scaleX - 4.0f;
                float wy = (gy + 0.5f) / scaleY - 2.0f;
                float wz = (gz + 0.5f) / scaleZ - 2.0f;

                for (int i = 1; i < 19; i++) {
                    int nx = gx + ex[i];
                    int ny = gy + ey[i];
                    int nz = gz + ez[i];

                    // Skip if neighbor is in bounds and fluid
                    if (nx >= 0 && nx < grid->sizeX && ny >= 0 &&
                        ny < grid->sizeY && nz >= 0 && nz < grid->sizeZ) {
                        int ni = nx + ny * grid->sizeX +
                                 nz * grid->sizeX * grid->sizeY;
                        if (solidData[ni] == 0)
                            continue;
                    } else {
                        continue; // boundary, not solid
                    }

                    // Ray from fluid cell toward solid neighbor
                    float dx = (float)ex[i] / scaleX;
                    float dy = (float)ey[i] / scaleY;
                    float dz = (float)ez[i] / scaleZ;

                    // Find nearest triangle intersection (Moller-Trumbore)
                    float bestT = 2.0f; // > 1 means no hit

                    for (int t = 0; t < numTriangles; t++) {
                        if (triBounds) {
                            TriBounds tb = triBounds[t];
                            // Quick reject: skip triangles far from ray
                            float rxMin = fminf(wx, wx + dx);
                            float rxMax = fmaxf(wx, wx + dx);
                            float ryMin = fminf(wy, wy + dy);
                            float ryMax = fmaxf(wy, wy + dy);
                            float rzMin = fminf(wz, wz + dz);
                            float rzMax = fmaxf(wz, wz + dz);
                            if (rxMax < tb.minY || ryMax < tb.maxY) {
                            } // wrong fields
                            // Use simple distance check instead
                        }

                        float *tri = &triangles[t * 12];
                        float v0x = tri[0], v0y = tri[1], v0z = tri[2];
                        float v1x = tri[4], v1y = tri[5], v1z = tri[6];
                        float v2x = tri[8], v2y = tri[9], v2z = tri[10];

                        float e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
                        float e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

                        float hx = dy * e2z - dz * e2y;
                        float hy = dz * e2x - dx * e2z;
                        float hz = dx * e2y - dy * e2x;
                        float a = e1x * hx + e1y * hy + e1z * hz;

                        if (a > -1e-6f && a < 1e-6f)
                            continue;

                        float f = 1.0f / a;
                        float sx = wx - v0x, sy = wy - v0y, sz = wz - v0z;
                        float u = f * (sx * hx + sy * hy + sz * hz);
                        if (u < 0.0f || u > 1.0f)
                            continue;

                        float qx = sy * e1z - sz * e1y;
                        float qy = sz * e1x - sx * e1z;
                        float qz = sx * e1y - sy * e1x;
                        float v = f * (dx * qx + dy * qy + dz * qz);
                        if (v < 0.0f || u + v > 1.0f)
                            continue;

                        float tHit = f * (e2x * qx + e2y * qy + e2z * qz);
                        if (tHit > 0.0f && tHit < bestT)
                            bestT = tHit;
                    }

                    // q is the fractional distance along the lattice link
                    float q;
                    if (bestT <= 1.0f)
                        q = bestT;
                    else
                        q = 0.5f; // fallback: midpoint

                    // Clamp to avoid degeneracy
                    if (q < 0.01f)
                        q = 0.01f;
                    if (q > 0.99f)
                        q = 0.99f;

                    int qIdx = cellIdx * 19 + i;
                    qData[qIdx] = q;
                    qLinks++;
                }
            }
        }
    }

    printf("Bouzidi: %d boundary links computed\n", qLinks);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->qBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, qCount * sizeof(float), qData);
    free(qData);

    free(solidData);
    if (triBounds)
        free(triBounds);
}
