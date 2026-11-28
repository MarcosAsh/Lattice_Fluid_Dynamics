#include "../lib/lbm.h"
#include "../lib/opengl_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int ex[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
static const int ey[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1};
static const int ez[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1};
// 72 w_i, as in lbm_force.comp
static const int c72[19] = {
    24, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

int LBM_SurfaceCells(LBMGrid *grid, int refPlaneX, Surface *out) {
    int nx = grid->sizeX, ny = grid->sizeY, nz = grid->sizeZ;
    size_t planeCells = (size_t)nx * ny;
    int *solid = (int *)malloc(planeCells * sizeof(int));
    float *q = (float *)malloc(planeCells * 19 * sizeof(float));
    int cap = 1024, n = 0;
    int *cell = (int *)malloc(cap * sizeof(int));
    unsigned char *role = (unsigned char *)malloc(cap);
    int *w36 = (int *)malloc((size_t)cap * 3 * sizeof(int));
    int *links = (int *)malloc(cap * sizeof(int));
    int rc = -1;
    if (!solid || !q || !cell || !role || !w36 || !links)
        goto done;

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    // One z plane at a time keeps the readback small
    for (int z = 0; z < nz; z++) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                           (GLintptr)(z * planeCells * sizeof(int)),
                           planeCells * sizeof(int),
                           solid);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->qBuffer);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                           (GLintptr)(z * planeCells * 19 * sizeof(float)),
                           planeCells * 19 * sizeof(float),
                           q);
        for (size_t c = 0; c < planeCells; c++) {
            // Same cells as the force shader sums
            if (solid[c] != 0)
                continue;
            int w[3] = {0, 0, 0}, nl = 0;
            for (int i = 1; i < 19; i++)
                if (q[c * 19 + i] >= 0.0f) {
                    w[0] += c72[i] * ex[i];
                    w[1] += c72[i] * ey[i];
                    w[2] += c72[i] * ez[i];
                    nl++;
                }
            unsigned char r = 0;
            if (nl > 0)
                r |= SURF_BOUNDARY;
            if ((int)(c % nx) == refPlaneX)
                r |= SURF_REFERENCE;
            if (!r)
                continue;
            if (n == cap) {
                cap *= 2;
                int *a = (int *)realloc(cell, cap * sizeof(int));
                if (a)
                    cell = a;
                unsigned char *b = (unsigned char *)realloc(role, cap);
                if (b)
                    role = b;
                int *d = (int *)realloc(w36, (size_t)cap * 3 * sizeof(int));
                if (d)
                    w36 = d;
                int *e = (int *)realloc(links, cap * sizeof(int));
                if (e)
                    links = e;
                if (!a || !b || !d || !e)
                    goto done;
            }
            cell[n] = (int)(z * planeCells + c);
            role[n] = r;
            memcpy(&w36[3 * n], w, sizeof(w));
            links[n] = nl;
            n++;
        }
    }
    rc = surface_init(out, nx, ny, nz, n, cell, role, w36, links);
    cell = NULL;
    role = NULL;
    w36 = NULL;
    links = NULL;
done:
    free(solid);
    free(q);
    free(cell);
    free(role);
    free(w36);
    free(links);
    return rc;
}

int LBM_SurfaceStart(LBMGrid *grid, const Surface *s) {
    if (!grid->surfaceShader) {
        grid->surfaceShader = createComputeShader("shaders/lbm_surface.comp");
        if (!grid->surfaceShader)
            return -1;
        grid->surface_listStartLoc =
            glGetUniformLocation(grid->surfaceShader, "listStart");
        grid->surface_listCountLoc =
            glGetUniformLocation(grid->surfaceShader, "listCount");
        grid->surface_sliceStartLoc =
            glGetUniformLocation(grid->surfaceShader, "sliceStart");
    }
    free(grid->surfaceCells);
    grid->surfaceCells =
        (int *)malloc((s->n > 0 ? (size_t)s->n : 1) * sizeof(int));
    float *zeros =
        (float *)calloc((s->n > 0 ? (size_t)s->n : 1) * 4, sizeof(float));
    if (!grid->surfaceCells || !zeros) {
        free(zeros);
        return -1;
    }
    memcpy(grid->surfaceCells, s->cell, (size_t)s->n * sizeof(int));
    grid->surfaceCount = s->n;
    if (!grid->surfaceListBuffer)
        glGenBuffers(1, &grid->surfaceListBuffer);
    if (!grid->surfaceSumBuffer)
        glGenBuffers(1, &grid->surfaceSumBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->surfaceListBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (s->n > 0 ? (size_t)s->n : 1) * sizeof(int),
                 grid->surfaceCells,
                 GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->surfaceSumBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (s->n > 0 ? (size_t)s->n : 1) * 4 * sizeof(float),
                 zeros,
                 GL_DYNAMIC_COPY);
    free(zeros);
    grid->surfaceSteps = 0;
    grid->surfaceAccumulate = 1;
    return 0;
}

// First list entry at or after global cell g
static int lowerBound(const int *a, int n, long g) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (a[mid] < g)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

void LBM_SurfaceStep(LBMGrid *grid) {
    if (!grid->surfaceAccumulate || grid->surfaceCount == 0)
        return;
    size_t plane = (size_t)grid->sizeX * grid->sizeY;
    size_t fSliceBytes = 19 * plane * sizeof(float);
    glUseProgram(grid->surfaceShader);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, grid->velocityBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, grid->surfaceListBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, grid->surfaceSumBuffer);
    for (int c = 0; c < grid->numChunks; c++) {
        int zStart = c * grid->slabZ;
        int zEnd = zStart + grid->slabZ;
        if (zEnd > grid->sizeZ)
            zEnd = grid->sizeZ;
        int k0 = lowerBound(
            grid->surfaceCells, grid->surfaceCount, (long)zStart * plane);
        int k1 = lowerBound(
            grid->surfaceCells, grid->surfaceCount, (long)zEnd * plane);
        if (k1 == k0)
            continue;
        if (grid->numChunks == 1) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, grid->fBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, grid->fNewBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, grid->qBuffer);
        } else {
            GLintptr off = (GLintptr)(zStart * fSliceBytes);
            GLsizeiptr sz = (GLsizeiptr)((zEnd - zStart) * fSliceBytes);
            glBindBufferRange(
                GL_SHADER_STORAGE_BUFFER, 2, grid->fBuffer, off, sz);
            glBindBufferRange(
                GL_SHADER_STORAGE_BUFFER, 3, grid->fNewBuffer, off, sz);
            glBindBufferRange(
                GL_SHADER_STORAGE_BUFFER, 7, grid->qBuffer, off, sz);
        }
        glUniform1i(grid->surface_listStartLoc, k0);
        glUniform1i(grid->surface_listCountLoc, k1 - k0);
        glUniform1i(grid->surface_sliceStartLoc, (int)(zStart * plane));
        glDispatchCompute((k1 - k0 + 63) / 64, 1, 1);
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    grid->surfaceSteps++;
}

int LBM_SurfaceReadWindow(LBMGrid *grid, float *sums) {
    size_t bytes = (size_t)grid->surfaceCount * 4 * sizeof(float);
    int steps = grid->surfaceSteps;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                    GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->surfaceSumBuffer);
    if (sums && bytes)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, sums);
    float *zeros = (float *)calloc(bytes ? bytes : 1, 1);
    if (zeros && bytes)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, zeros);
    free(zeros);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    grid->surfaceSteps = 0;
    return steps;
}

void LBM_SurfaceFree(LBMGrid *grid) {
    if (grid->surfaceShader)
        glDeleteProgram(grid->surfaceShader);
    if (grid->surfaceListBuffer)
        glDeleteBuffers(1, &grid->surfaceListBuffer);
    if (grid->surfaceSumBuffer)
        glDeleteBuffers(1, &grid->surfaceSumBuffer);
    free(grid->surfaceCells);
    grid->surfaceShader = 0;
    grid->surfaceListBuffer = grid->surfaceSumBuffer = 0;
    grid->surfaceCells = NULL;
    grid->surfaceCount = 0;
    grid->surfaceAccumulate = 0;
}
