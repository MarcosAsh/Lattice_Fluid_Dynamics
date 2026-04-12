#define _USE_MATH_DEFINES
#include "../lib/model_bounds.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CollisionGrid buildCollisionGrid(GPUTriangle *tris,
                                 int numTris,
                                 float bminX,
                                 float bminY,
                                 float bminZ,
                                 float bmaxX,
                                 float bmaxY,
                                 float bmaxZ) {
    CollisionGrid g = {0};
    int R = COLL_GRID_RES;
    g.totalCells = R * R * R;

    // Expand AABB slightly so particles near the surface still map
    // into a valid cell.
    float pad = 0.05f;
    g.minX = bminX - pad;
    g.minY = bminY - pad;
    g.minZ = bminZ - pad;
    float gMaxX = bmaxX + pad;
    float gMaxY = bmaxY + pad;
    float gMaxZ = bmaxZ + pad;
    g.cellSizeX = (gMaxX - g.minX) / R;
    g.cellSizeY = (gMaxY - g.minY) / R;
    g.cellSizeZ = (gMaxZ - g.minZ) / R;

    // First pass: count how many cells each triangle overlaps
    g.cellCount = (int *)calloc(g.totalCells, sizeof(int));
    int *tmpCount = (int *)calloc(g.totalCells, sizeof(int));
    if (!g.cellCount || !tmpCount) {
        free(g.cellCount);
        free(tmpCount);
        memset(&g, 0, sizeof(g));
        return g;
    }

    for (int t = 0; t < numTris; t++) {
        float txMin = fminf(tris[t].v0x, fminf(tris[t].v1x, tris[t].v2x));
        float txMax = fmaxf(tris[t].v0x, fmaxf(tris[t].v1x, tris[t].v2x));
        float tyMin = fminf(tris[t].v0y, fminf(tris[t].v1y, tris[t].v2y));
        float tyMax = fmaxf(tris[t].v0y, fmaxf(tris[t].v1y, tris[t].v2y));
        float tzMin = fminf(tris[t].v0z, fminf(tris[t].v1z, tris[t].v2z));
        float tzMax = fmaxf(tris[t].v0z, fmaxf(tris[t].v1z, tris[t].v2z));

        int x0 = (int)floorf((txMin - g.minX) / g.cellSizeX);
        int x1 = (int)floorf((txMax - g.minX) / g.cellSizeX);
        int y0 = (int)floorf((tyMin - g.minY) / g.cellSizeY);
        int y1 = (int)floorf((tyMax - g.minY) / g.cellSizeY);
        int z0 = (int)floorf((tzMin - g.minZ) / g.cellSizeZ);
        int z1 = (int)floorf((tzMax - g.minZ) / g.cellSizeZ);

        if (x0 < 0)
            x0 = 0;
        if (x1 >= R)
            x1 = R - 1;
        if (y0 < 0)
            y0 = 0;
        if (y1 >= R)
            y1 = R - 1;
        if (z0 < 0)
            z0 = 0;
        if (z1 >= R)
            z1 = R - 1;

        for (int cz = z0; cz <= z1; cz++)
            for (int cy = y0; cy <= y1; cy++)
                for (int cx = x0; cx <= x1; cx++) {
                    int idx = cx + cy * R + cz * R * R;
                    g.cellCount[idx]++;
                    g.totalIndices++;
                }
    }

    // Prefix sum to get cellStart
    g.cellStart = (int *)malloc(g.totalCells * sizeof(int));
    if (!g.cellStart) {
        free(g.cellCount);
        free(tmpCount);
        memset(&g, 0, sizeof(g));
        return g;
    }
    g.cellStart[0] = 0;
    for (int i = 1; i < g.totalCells; i++)
        g.cellStart[i] = g.cellStart[i - 1] + g.cellCount[i - 1];

    // Second pass: fill triIndices
    g.triIndices = (int *)malloc(g.totalIndices * sizeof(int));
    if (!g.triIndices) {
        free(g.cellStart);
        free(g.cellCount);
        free(tmpCount);
        memset(&g, 0, sizeof(g));
        return g;
    }

    for (int t = 0; t < numTris; t++) {
        float txMin = fminf(tris[t].v0x, fminf(tris[t].v1x, tris[t].v2x));
        float txMax = fmaxf(tris[t].v0x, fmaxf(tris[t].v1x, tris[t].v2x));
        float tyMin = fminf(tris[t].v0y, fminf(tris[t].v1y, tris[t].v2y));
        float tyMax = fmaxf(tris[t].v0y, fmaxf(tris[t].v1y, tris[t].v2y));
        float tzMin = fminf(tris[t].v0z, fminf(tris[t].v1z, tris[t].v2z));
        float tzMax = fmaxf(tris[t].v0z, fmaxf(tris[t].v1z, tris[t].v2z));

        int x0 = (int)floorf((txMin - g.minX) / g.cellSizeX);
        int x1 = (int)floorf((txMax - g.minX) / g.cellSizeX);
        int y0 = (int)floorf((tyMin - g.minY) / g.cellSizeY);
        int y1 = (int)floorf((tyMax - g.minY) / g.cellSizeY);
        int z0 = (int)floorf((tzMin - g.minZ) / g.cellSizeZ);
        int z1 = (int)floorf((tzMax - g.minZ) / g.cellSizeZ);

        if (x0 < 0)
            x0 = 0;
        if (x1 >= R)
            x1 = R - 1;
        if (y0 < 0)
            y0 = 0;
        if (y1 >= R)
            y1 = R - 1;
        if (z0 < 0)
            z0 = 0;
        if (z1 >= R)
            z1 = R - 1;

        for (int cz = z0; cz <= z1; cz++)
            for (int cy = y0; cy <= y1; cy++)
                for (int cx = x0; cx <= x1; cx++) {
                    int idx = cx + cy * R + cz * R * R;
                    int pos = g.cellStart[idx] + tmpCount[idx];
                    g.triIndices[pos] = t;
                    tmpCount[idx]++;
                }
    }

    free(tmpCount);
    return g;
}

CarBounds computeModelBounds(Model *model,
                             float scale,
                             float offsetX,
                             float offsetY,
                             float offsetZ,
                             float rotationY) {
    CarBounds bounds;

    if (model == NULL || model->vertexCount == 0) {
        printf("Warning: No model vertices, using default bounds\n");
        bounds.minX = bounds.minY = bounds.minZ = -0.5f;
        bounds.maxX = bounds.maxY = bounds.maxZ = 0.5f;
        bounds.centerX = bounds.centerY = bounds.centerZ = 0.0f;
        return bounds;
    }

    float radY = rotationY * M_PI / 180.0f;
    float cosY = cosf(radY);
    float sinY = sinf(radY);

    bounds.minX = bounds.minY = bounds.minZ = FLT_MAX;
    bounds.maxX = bounds.maxY = bounds.maxZ = -FLT_MAX;

    for (int i = 0; i < model->vertexCount; i++) {
        float x = model->vertices[i].x * scale + offsetX;
        float y = model->vertices[i].y * scale + offsetY;
        float z = model->vertices[i].z * scale + offsetZ;

        float rotatedX = x * cosY - z * sinY;
        float rotatedZ = x * sinY + z * cosY;

        if (rotatedX < bounds.minX)
            bounds.minX = rotatedX;
        if (y < bounds.minY)
            bounds.minY = y;
        if (rotatedZ < bounds.minZ)
            bounds.minZ = rotatedZ;
        if (rotatedX > bounds.maxX)
            bounds.maxX = rotatedX;
        if (y > bounds.maxY)
            bounds.maxY = y;
        if (rotatedZ > bounds.maxZ)
            bounds.maxZ = rotatedZ;
    }

    bounds.centerX = (bounds.minX + bounds.maxX) * 0.5f;
    bounds.centerY = (bounds.minY + bounds.maxY) * 0.5f;
    bounds.centerZ = (bounds.minZ + bounds.maxZ) * 0.5f;

    printf("Model bounds (rotated %.1f deg): min(%.2f, %.2f, %.2f) max(%.2f, "
           "%.2f, %.2f)\n",
           rotationY,
           bounds.minX,
           bounds.minY,
           bounds.minZ,
           bounds.maxX,
           bounds.maxY,
           bounds.maxZ);
    printf("Model center: (%.2f, %.2f, %.2f)\n",
           bounds.centerX,
           bounds.centerY,
           bounds.centerZ);

    return bounds;
}

GPUTriangle *createTriangleBuffer(Model *model,
                                  float scale,
                                  float offsetX,
                                  float offsetY,
                                  float offsetZ,
                                  float rotationY,
                                  int *outCount) {
    *outCount = 0;

    if (!model || model->faceCount == 0 || model->vertexCount == 0) {
        printf("No model data for triangle buffer\n");
        return NULL;
    }

    printf("Creating triangle buffer: %d faces, %d vertices\n",
           model->faceCount,
           model->vertexCount);

    GPUTriangle *tris =
        (GPUTriangle *)malloc(model->faceCount * sizeof(GPUTriangle));
    if (!tris) {
        printf("Failed to allocate triangle buffer!\n");
        return NULL;
    }

    float radY = rotationY * M_PI / 180.0f;
    float cosY = cosf(radY);
    float sinY = sinf(radY);

    int validCount = 0;
    for (int i = 0; i < model->faceCount; i++) {
        int idx0 = model->faces[i].v1 - 1;
        int idx1 = model->faces[i].v2 - 1;
        int idx2 = model->faces[i].v3 - 1;

        if (idx0 < 0 || idx0 >= model->vertexCount || idx1 < 0 ||
            idx1 >= model->vertexCount || idx2 < 0 ||
            idx2 >= model->vertexCount) {
            continue;
        }

        Vertex v0 = model->vertices[idx0];
        Vertex v1 = model->vertices[idx1];
        Vertex v2 = model->vertices[idx2];

        float x0 = v0.x * scale + offsetX;
        float y0 = v0.y * scale + offsetY;
        float z0 = v0.z * scale + offsetZ;
        tris[validCount].v0x = x0 * cosY - z0 * sinY;
        tris[validCount].v0y = y0;
        tris[validCount].v0z = x0 * sinY + z0 * cosY;
        tris[validCount].pad0 = 0;

        float x1 = v1.x * scale + offsetX;
        float y1 = v1.y * scale + offsetY;
        float z1 = v1.z * scale + offsetZ;
        tris[validCount].v1x = x1 * cosY - z1 * sinY;
        tris[validCount].v1y = y1;
        tris[validCount].v1z = x1 * sinY + z1 * cosY;
        tris[validCount].pad1 = 0;

        float x2 = v2.x * scale + offsetX;
        float y2 = v2.y * scale + offsetY;
        float z2 = v2.z * scale + offsetZ;
        tris[validCount].v2x = x2 * cosY - z2 * sinY;
        tris[validCount].v2y = y2;
        tris[validCount].v2z = x2 * sinY + z2 * cosY;
        tris[validCount].pad2 = 0;

        validCount++;
    }

    *outCount = validCount;
    printf("Created %d valid triangles\n", validCount);
    return tris;
}
