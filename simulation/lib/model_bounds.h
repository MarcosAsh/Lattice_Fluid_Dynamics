#ifndef MODEL_BOUNDS_H
#define MODEL_BOUNDS_H

#include "../obj-file-loader/lib/model_loader.h"

#define COLL_GRID_RES 8

typedef struct {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    float centerX, centerY, centerZ;
} CarBounds;

typedef struct {
    float v0x, v0y, v0z, pad0;
    float v1x, v1y, v1z, pad1;
    float v2x, v2y, v2z, pad2;
} GPUTriangle;

// Uniform grid for spatial acceleration of per-triangle collision.
// Triangles are binned into grid cells so particles only test nearby ones.
typedef struct {
    int *cellStart;  // offset into triIndices per cell
    int *cellCount;  // triangle count per cell
    int *triIndices; // packed triangle indices
    int totalIndices;
    int totalCells;
    float minX, minY, minZ;
    float cellSizeX, cellSizeY, cellSizeZ;
} CollisionGrid;

CarBounds computeModelBounds(Model *model,
                             float scale,
                             float offsetX,
                             float offsetY,
                             float offsetZ,
                             float rotationY);

GPUTriangle *createTriangleBuffer(Model *model,
                                  float scale,
                                  float offsetX,
                                  float offsetY,
                                  float offsetZ,
                                  float rotationY,
                                  int *outCount);

CollisionGrid buildCollisionGrid(GPUTriangle *tris,
                                 int numTris,
                                 float bminX,
                                 float bminY,
                                 float bminZ,
                                 float bmaxX,
                                 float bmaxY,
                                 float bmaxZ);

#endif // MODEL_BOUNDS_H
