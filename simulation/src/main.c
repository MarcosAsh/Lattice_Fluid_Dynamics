#define _USE_MATH_DEFINES
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <glad/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <getopt.h>
#include <string.h>
#include <stdint.h>

#include "../lib/gl_context.h"
#include "../lib/lbm.h"
#include "../lib/fluid_cube.h"
#include "../lib/particle_system.h"
#include "../obj-file-loader/lib/model_loader.h"
#include "../lib/render_model.h"
#include "../lib/opengl_utils.h"
#include "../lib/config.h"
#include "../lib/ml_predict.h"

#define GPU_PARTICLES MAX_PARTICLES

// Global model transform (definitions)
float g_modelScale = 1.0f;
float g_offsetX = 0.0f;
float g_offsetY = -0.1f;
float g_offsetZ = -0.9f;
float g_carRotationY = 360.0f;

// Slider variables
int sliderX = 100;
int sliderY = 50;
int sliderWidth = 200;
int sliderHeight = 20;
int handleWidth = 10;
int handleX = 100;
int isDragging = 0;
float windSpeed = 1.0f;

// Camera rotation variables
float cameraAngleY = 0.0f;
float cameraAngleX = 0.3f;
float cameraDistance = 6.0f;
float cameraTargetX = -1.5f; // between car center and wake
float cameraTargetY = 0.0f;
float cameraTargetZ = 0.0f;

// Mouse control variables
int mouseDown = 0;
int middleMouseDown = 0;
int lastMouseX = 0;
int lastMouseY = 0;

// Visualization mode
int visualizationMode = 1;
float maxSpeed = 0.5f;

// LBM settings
int useLBM = 1;
LBMGrid *lbmGrid = NULL;
int lbmSubsteps = 10;

// Pause / single-step
int paused = 0;
int stepOnce = 0;

const char *vizModeNames[] = {"Depth",
                              "Velocity Magnitude",
                              "Velocity Direction",
                              "Particle Lifetime",
                              "Turbulence",
                              "Flow Progress",
                              "Vorticity",
                              "Pathlines",
                              "Pressure",
                              "Streamlines"};
const int numVizModes = 10;

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
#define COLL_GRID_RES 8

typedef struct {
    int *cellStart;  // offset into triIndices per cell
    int *cellCount;  // triangle count per cell
    int *triIndices; // packed triangle indices
    int totalIndices;
    int totalCells;
    float minX, minY, minZ;
    float cellSizeX, cellSizeY, cellSizeZ;
} CollisionGrid;

static CollisionGrid buildCollisionGrid(GPUTriangle *tris,
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
    g.cellStart[0] = 0;
    for (int i = 1; i < g.totalCells; i++)
        g.cellStart[i] = g.cellStart[i - 1] + g.cellCount[i - 1];

    // Second pass: fill triIndices
    g.triIndices = (int *)malloc(g.totalIndices * sizeof(int));

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

void checkGLError(const char *label) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        printf("OpenGL error at %s: 0x%x\n", label, err);
    }
}

void calculateViewMatrix(float *view,
                         float angleY,
                         float angleX,
                         float distance,
                         float targetX,
                         float targetY,
                         float targetZ) {
    float eyeX = targetX + distance * sinf(angleY) * cosf(angleX);
    float eyeY = targetY + distance * sinf(angleX);
    float eyeZ = targetZ + distance * cosf(angleY) * cosf(angleX);

    float forward[3] = {targetX - eyeX, targetY - eyeY, targetZ - eyeZ};
    float forwardLength =
        sqrtf(forward[0] * forward[0] + forward[1] * forward[1] +
              forward[2] * forward[2]);
    if (forwardLength < 1e-8f)
        forwardLength = 1e-8f;
    forward[0] /= forwardLength;
    forward[1] /= forwardLength;
    forward[2] /= forwardLength;

    float up[3] = {0.0f, 1.0f, 0.0f};

    float side[3] = {forward[1] * up[2] - forward[2] * up[1],
                     forward[2] * up[0] - forward[0] * up[2],
                     forward[0] * up[1] - forward[1] * up[0]};
    float sideLength =
        sqrtf(side[0] * side[0] + side[1] * side[1] + side[2] * side[2]);
    if (sideLength < 1e-8f)
        sideLength = 1e-8f;
    side[0] /= sideLength;
    side[1] /= sideLength;
    side[2] /= sideLength;

    up[0] = side[1] * forward[2] - side[2] * forward[1];
    up[1] = side[2] * forward[0] - side[0] * forward[2];
    up[2] = side[0] * forward[1] - side[1] * forward[0];

    view[0] = side[0];
    view[1] = up[0];
    view[2] = -forward[0];
    view[3] = 0.0f;

    view[4] = side[1];
    view[5] = up[1];
    view[6] = -forward[1];
    view[7] = 0.0f;

    view[8] = side[2];
    view[9] = up[2];
    view[10] = -forward[2];
    view[11] = 0.0f;

    view[12] = -(side[0] * eyeX + side[1] * eyeY + side[2] * eyeZ);
    view[13] = -(up[0] * eyeX + up[1] * eyeY + up[2] * eyeZ);
    view[14] = forward[0] * eyeX + forward[1] * eyeY + forward[2] * eyeZ;
    view[15] = 1.0f;
}

void saveFrameToPPM(const char *filename, int width, int height) {
    unsigned char *pixels = (unsigned char *)malloc(width * height * 3);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Could not open %s for writing\n", filename);
        free(pixels);
        return;
    }

    fprintf(f, "P6\n%d %d\n255\n", width, height);

    for (int y = height - 1; y >= 0; y--) {
        fwrite(pixels + y * width * 3, 1, width * 3, f);
    }

    fclose(f);
    free(pixels);
}

// Write LBM velocity field as VTK ImageData (.vti) for ParaView.
// Binary appended format for compact output.
static void writeVTI(LBMGrid *grid, const char *path, int step) {
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/field_%06d.vti", path, step);

    int nx = grid->sizeX;
    int ny = grid->sizeY;
    int nz = grid->sizeZ;
    int total = nx * ny * nz;

    // Read velocity (vec4) and solid (int) from GPU
    size_t velBytes = (size_t)total * 4 * sizeof(float);
    size_t solidBytes = (size_t)total * sizeof(int);
    float *vel = (float *)malloc(velBytes);
    int *solid = (int *)malloc(solidBytes);
    if (!vel || !solid) {
        free(vel);
        free(solid);
        return;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->velocityBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, velBytes, vel);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, solidBytes, solid);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    FILE *f = fopen(filename, "wb");
    if (!f) {
        free(vel);
        free(solid);
        return;
    }

    // VTI header
    fprintf(f,
            "<?xml version=\"1.0\"?>\n"
            "<VTKFile type=\"ImageData\" version=\"1.0\""
            " byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
            "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\""
            " Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
            "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n"
            "      <PointData Vectors=\"velocity\" Scalars=\"solid\">\n"
            "        <DataArray type=\"Float32\" Name=\"velocity\""
            " NumberOfComponents=\"3\" format=\"appended\""
            " offset=\"0\"/>\n"
            "        <DataArray type=\"Int32\" Name=\"solid\""
            " format=\"appended\" offset=\"%lu\"/>\n"
            "      </PointData>\n"
            "    </Piece>\n"
            "  </ImageData>\n"
            "  <AppendedData encoding=\"raw\">\n_",
            nx, ny, nz, nx, ny, nz,
            (unsigned long)(sizeof(uint64_t) +
                            (size_t)total * 3 * sizeof(float)));

    // Velocity array (strip w component from vec4)
    uint64_t velDataSize = (uint64_t)total * 3 * sizeof(float);
    fwrite(&velDataSize, sizeof(uint64_t), 1, f);
    for (int i = 0; i < total; i++) {
        fwrite(&vel[i * 4], sizeof(float), 3, f);
    }

    // Solid array
    uint64_t solidDataSize = (uint64_t)total * sizeof(int);
    fwrite(&solidDataSize, sizeof(uint64_t), 1, f);
    fwrite(solid, sizeof(int), total, f);

    fprintf(f, "\n  </AppendedData>\n</VTKFile>\n");
    fclose(f);
    free(vel);
    free(solid);
    printf("VTK: wrote %s\n", filename);
}

int main(int argc, char *argv[]) {
    printf("Starting 3D Fluid Simulation...\n");
    srand(time(NULL));

    // Parse command line arguments
    float windSpeed = 1.0f;
    int visualizationMode = 1;
    int collisionMode = 2;
    int renderDuration = 0;
    char outputPath[256] = "";
    char modelPath[512] = "assets/3d-files/car-model.obj";
    int slantAngle = 0;
    float reynoldsNumber = 0.0f;
    int scaleFromCLI = 0;
    int gridX = 128, gridY = 64, gridZ = 64;
    float smagorinskyCs = 0.1f;
    int useMRT = 0;
    char vtkOutputPath[256] = "";
    int vtkInterval = 100; // frames between VTK dumps

    static struct option long_options[] = {
        {"wind", required_argument, 0, 'w'},
        {"viz", required_argument, 0, 'v'},
        {"collision", required_argument, 0, 'c'},
        {"duration", required_argument, 0, 'd'},
        {"output", required_argument, 0, 'o'},
        {"model", required_argument, 0, 'm'},
        {"angle", required_argument, 0, 'a'},
        {"scale", required_argument, 0, 's'},
        {"reynolds", required_argument, 0, 'r'},
        {"grid", required_argument, 0, 'g'},
        {"smagorinsky", required_argument, 0, 'S'},
        {"mrt", no_argument, 0, 'M'},
        {"vtk-output", required_argument, 0, 'V'},
        {"vtk-interval", required_argument, 0, 'I'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}

    };

    int opt;
    while ((opt = getopt_long(
                argc, argv, "w:v:c:d:o:m:a:r:s:g:S:Mh", long_options, NULL)) !=
           -1) {
        switch (opt) {
        case 'w':
            windSpeed = atof(optarg);
            if (windSpeed < 0)
                windSpeed = 0;
            if (windSpeed > 5)
                windSpeed = 5;
            break;
        case 'v':
            visualizationMode = atoi(optarg);
            if (visualizationMode < 0)
                visualizationMode = 0;
            if (visualizationMode > 6)
                visualizationMode = 6;
            break;
        case 'c':
            collisionMode = atoi(optarg);
            if (collisionMode < 0)
                collisionMode = 0;
            if (collisionMode > 3)
                collisionMode = 3;
            break;
        case 'd':
            renderDuration = atoi(optarg);
            if (renderDuration < 0)
                renderDuration = 0;
            break;
        case 'o':
            strncpy(outputPath, optarg, sizeof(outputPath) - 1);
            outputPath[sizeof(outputPath) - 1] = '\0';
            break;
        case 'm':
            strncpy(modelPath, optarg, sizeof(modelPath) - 1);
            modelPath[sizeof(modelPath) - 1] = '\0';
            break;
        case 'a':
            slantAngle = atoi(optarg);
            if (slantAngle == 25) {
                strncpy(modelPath,
                        "assets/3d-files/ahmed_25deg_m.obj",
                        sizeof(modelPath) - 1);
            } else if (slantAngle == 35) {
                strncpy(modelPath,
                        "assets/3d-files/ahmed_35deg_m.obj",
                        sizeof(modelPath) - 1);
            }
            break;
        case 's':
            g_modelScale = atof(optarg);
            scaleFromCLI = 1;
            break;
        case 'r':
            reynoldsNumber = atof(optarg);
            if (reynoldsNumber < 0)
                reynoldsNumber = 0;
            break;
        case 'S':
            smagorinskyCs = atof(optarg);
            if (smagorinskyCs < 0.0f)
                smagorinskyCs = 0.0f;
            if (smagorinskyCs > 0.5f)
                smagorinskyCs = 0.5f;
            break;
        case 'g':
            if (sscanf(optarg, "%dx%dx%d", &gridX, &gridY, &gridZ) != 3) {
                /* Try single number: NxN/2xN/2 */
                int n = atoi(optarg);
                if (n > 0) {
                    gridX = n;
                    gridY = n / 2;
                    gridZ = n / 2;
                }
            }
            break;
        case 'M':
            useMRT = 1;
            break;
        case 'V':
            strncpy(vtkOutputPath, optarg, sizeof(vtkOutputPath) - 1);
            vtkOutputPath[sizeof(vtkOutputPath) - 1] = '\0';
            break;
        case 'I':
            vtkInterval = atoi(optarg);
            if (vtkInterval < 1)
                vtkInterval = 1;
            break;
        case 'h':
        default:
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -w, --wind=SPEED      Wind speed 0-5 (default: 1.0)\n");
            printf("  -v, --viz=MODE        Visualization mode 0-6 (default: "
                   "1)\n");
            printf("  -c, --collision=MODE  Collision 0=off, 1=AABB, 2=mesh, "
                   "3=voxel "
                   "(default: 1)\n");
            printf("  -d, --duration=SECS   Render duration (0=interactive, "
                   "default: 0)\n");
            printf("  -o, --output=PATH     Output directory for frames\n");
            printf("  -m, --model=PATH      Path to OBJ model file\n");
            printf(
                "  -a, --angle=DEGREES   Ahmed body slant angle (25 or 35)\n");
            printf(
                "  -s, --scale=SCALE     Model scale factor (default: 0.05)\n");
            printf("  -r, --reynolds=RE     Target Reynolds number "
                   "(0=derive from wind speed)\n");
            printf("  -g, --grid=XxYxZ      Grid size (default: 128x64x64)\n");
            printf("  -S, --smagorinsky=CS  Smagorinsky constant 0-0.5 "
                   "(default: 0.1)\n");
            printf("  -M, --mrt             Enable MRT collision operator\n");
            printf("  --vtk-output=PATH     Directory for VTK field dumps\n");
            printf("  --vtk-interval=N      Frames between VTK dumps "
                   "(default: 100)\n");
            printf("  -h, --help            Show this help\n");
            return 0;
        }
    }

    // Print config
    printf("Configuration:\n");
    printf("  Model: %s\n", modelPath);
    if (slantAngle > 0)
        printf("  Slant Angle: %d°\n", slantAngle);
    printf("  Wind Speed: %.1f m/s\n", windSpeed);
    printf("  Visualization: %d\n", visualizationMode);
    printf("  Collision: %d\n", collisionMode);
    if (reynoldsNumber > 0)
        printf("  Reynolds: %.0f\n", reynoldsNumber);

    if (collisionMode == 3 && !useLBM) {
        printf("  Voxel collision (mode 3) requires LBM; falling back to AABB "
               "(mode 1)\n");
        collisionMode = 1;
    }

    if (renderDuration > 0) {
        printf("  Render Mode: %d seconds to %s\n", renderDuration, outputPath);
    } else {
        printf("  Mode: Interactive\n");
    }

    // Create GL context: EGL for headless, SDL for interactive
    GLContext *glCtx;
    if (renderDuration > 0) {
        glCtx = GLContext_CreateHeadless(WIDTH, HEIGHT);
    } else {
        glCtx = GLContext_CreateInteractive(WIDTH, HEIGHT);
    }
    if (!glCtx) {
        printf("Failed to create GL context\n");
        return 1;
    }

    printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
    printf("GLSL Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    printf("Renderer: %s\n", glGetString(GL_RENDERER));

    SDL_GL_SetSwapInterval(1);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    checkGLError("After initial GL setup");

    float aspect = (float)WIDTH / HEIGHT;
    float fov = 45.0f;
    float near = 0.1f;
    float far = 100.0f;
    float top = tanf(fov * 0.5f * M_PI / 180.0f) * near;
    float bottom = -top;
    float right = top * aspect;
    float left = -right;

    float projection[16] = {(2.0f * near) / (right - left),
                            0.0f,
                            0.0f,
                            0.0f,
                            0.0f,
                            (2.0f * near) / (top - bottom),
                            0.0f,
                            0.0f,
                            (right + left) / (right - left),
                            (top + bottom) / (top - bottom),
                            -(far + near) / (far - near),
                            -1.0f,
                            0.0f,
                            0.0f,
                            -(2.0f * far * near) / (far - near),
                            0.0f};

    float view[16];
    calculateViewMatrix(view,
                        cameraAngleY,
                        cameraAngleX,
                        cameraDistance,
                        cameraTargetX,
                        cameraTargetY,
                        cameraTargetZ);

    printf("Loading shaders...\n");
    GLuint particleShaderProgram =
        createShaderProgram("shaders/particle.vert", "shaders/particle.frag");
    if (particleShaderProgram == 0) {
        printf("Failed to create particle shader program\n");
        GLContext_Destroy(glCtx);
        return 1;
    }

    GLuint computeShaderProgram = createComputeShader("shaders/particle.comp");
    if (computeShaderProgram == 0) {
        printf("Failed to create compute shader program\n");
        glDeleteProgram(particleShaderProgram);
        GLContext_Destroy(glCtx);
        return 1;
    }

    checkGLError("After shader creation");

    glUseProgram(particleShaderProgram);
    GLuint projectionLoc =
        glGetUniformLocation(particleShaderProgram, "projection");
    GLuint viewLoc = glGetUniformLocation(particleShaderProgram, "view");
    GLuint vizModeLoc =
        glGetUniformLocation(particleShaderProgram, "visualizationMode");
    GLuint maxSpeedLoc =
        glGetUniformLocation(particleShaderProgram, "maxSpeed");

    printf("Particle shader uniform locations: projection=%d, view=%d, "
           "vizMode=%d, maxSpeed=%d\n",
           projectionLoc,
           viewLoc,
           vizModeLoc,
           maxSpeedLoc);

    if (projectionLoc != -1) {
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, projection);
    }

    checkGLError("After setting uniforms");

    printf("Loading 3D model: %s\n", modelPath);
    Model carModel = loadOBJ(modelPath);

    if (carModel.vertexCount == 0) {
        printf("Trying alternative path...\n");
        carModel = loadOBJ("/home/marcos_ashton/3dFluidDynamicsInC/assets/"
                           "3d-files/car-model.obj");
    }
    if (carModel.vertexCount == 0) {
        printf("Trying another path...\n");
        carModel = loadOBJ("../assets/3d-files/car-model.obj");
    }

    printf("Model loaded: %d vertices, %d faces\n",
           carModel.vertexCount,
           carModel.faceCount);

    // Compute model center and auto-center it
    if (carModel.vertexCount > 0) {
        float minX = FLT_MAX, minY = FLT_MAX, minZ = FLT_MAX;
        float maxX = -FLT_MAX, maxY = -FLT_MAX, maxZ = -FLT_MAX;

        for (int i = 0; i < carModel.vertexCount; i++) {
            if (carModel.vertices[i].x < minX)
                minX = carModel.vertices[i].x;
            if (carModel.vertices[i].y < minY)
                minY = carModel.vertices[i].y;
            if (carModel.vertices[i].z < minZ)
                minZ = carModel.vertices[i].z;
            if (carModel.vertices[i].x > maxX)
                maxX = carModel.vertices[i].x;
            if (carModel.vertices[i].y > maxY)
                maxY = carModel.vertices[i].y;
            if (carModel.vertices[i].z > maxZ)
                maxZ = carModel.vertices[i].z;
        }

        float centerX = (minX + maxX) * 0.5f;
        float centerY = (minY + maxY) * 0.5f;
        float centerZ = (minZ + maxZ) * 0.5f;

        printf("Model raw bounds: (%.2f,%.2f,%.2f) to (%.2f,%.2f,%.2f)\n",
               minX,
               minY,
               minZ,
               maxX,
               maxY,
               maxZ);
        printf("Model center: (%.2f, %.2f, %.2f)\n", centerX, centerY, centerZ);

        // Auto-scale if not set via CLI
        float sizeX = maxX - minX;
        float sizeY = maxY - minY;
        float sizeZ = maxZ - minZ;
        float maxDim = fmaxf(sizeX, fmaxf(sizeY, sizeZ));

        if (!scaleFromCLI) {
            // Scale so largest dimension fits in ~2 world units. On a
            // 256x128x128 grid this gives ~24 cells across the body
            // height with ~3% blockage.
            g_modelScale = 2.4f / maxDim;
        }
        printf("Auto scale: %.6f (max dim: %.2f)\n", g_modelScale, maxDim);

        // Center model at origin first; the upstream shift is
        // applied after auto-rotation so it lands on the correct axis.
        g_offsetX = -centerX * g_modelScale;
        g_offsetY = -centerY * g_modelScale;
        g_offsetZ = -centerZ * g_modelScale;

        printf("Auto offset: (%.4f, %.4f, %.4f)\n",
               g_offsetX,
               g_offsetY,
               g_offsetZ);
    }

    // Auto-orient: rotate so the model's longest axis is
    // streamwise (x). If it's already in x, no rotation needed.
    // If the longest axis is z, rotate 90 to swap z->x.
    {
        float sizeX = 0, sizeZ = 0;
        if (carModel.vertexCount > 0) {
            float mnX = FLT_MAX, mxX = -FLT_MAX;
            float mnZ = FLT_MAX, mxZ = -FLT_MAX;
            for (int i = 0; i < carModel.vertexCount; i++) {
                float x = carModel.vertices[i].x * g_modelScale + g_offsetX;
                float z = carModel.vertices[i].z * g_modelScale + g_offsetZ;
                if (x < mnX)
                    mnX = x;
                if (x > mxX)
                    mxX = x;
                if (z < mnZ)
                    mnZ = z;
                if (z > mxZ)
                    mxZ = z;
            }
            sizeX = mxX - mnX;
            sizeZ = mxZ - mnZ;
        }
        if (sizeZ > sizeX) {
            g_carRotationY = 90.0f;
            printf("Auto-rotation: 90 deg (longest axis was z)\n");
        } else {
            g_carRotationY = 0.0f;
            printf("Auto-rotation: 0 deg (longest axis already x)\n");
        }
    }

    // Shift model upstream: place center at x=-2 in post-rotation
    // space.  The transform is rotate(scale*v + offset), so we need
    // the pre-rotation shift (dx, dz) whose image under the rotation
    // is (-2, 0).
    {
        float radY = g_carRotationY * (float)M_PI / 180.0f;
        g_offsetX += -2.0f * cosf(radY);
        g_offsetZ += 2.0f * sinf(radY);
    }

    CarBounds carBounds = computeModelBounds(&carModel,
                                             g_modelScale,
                                             g_offsetX,
                                             g_offsetY,
                                             g_offsetZ,
                                             g_carRotationY);

    printf("Creating triangle buffer...\n");
    int numTriangles = 0;
    GPUTriangle *triangleData = createTriangleBuffer(&carModel,
                                                     g_modelScale,
                                                     g_offsetX,
                                                     g_offsetY,
                                                     g_offsetZ,
                                                     g_carRotationY,
                                                     &numTriangles);

    GLuint triangleBuffer = 0;
    if (triangleData && numTriangles > 0) {
        glGenBuffers(1, &triangleBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, triangleBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     numTriangles * sizeof(GPUTriangle),
                     triangleData,
                     GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, triangleBuffer);
        printf("Uploaded %d triangles to GPU\n", numTriangles);
    }

    // Build spatial acceleration grid for per-triangle collision
    GLuint gridCellStartBuf = 0, gridCellCountBuf = 0, gridTriIdxBuf = 0;
    CollisionGrid collGrid = {0};

    if (triangleData && numTriangles > 0) {
        collGrid = buildCollisionGrid(triangleData,
                                      numTriangles,
                                      carBounds.minX,
                                      carBounds.minY,
                                      carBounds.minZ,
                                      carBounds.maxX,
                                      carBounds.maxY,
                                      carBounds.maxZ);

        glGenBuffers(1, &gridCellStartBuf);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridCellStartBuf);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     collGrid.totalCells * sizeof(int),
                     collGrid.cellStart,
                     GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, gridCellStartBuf);

        glGenBuffers(1, &gridCellCountBuf);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridCellCountBuf);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     collGrid.totalCells * sizeof(int),
                     collGrid.cellCount,
                     GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, gridCellCountBuf);

        glGenBuffers(1, &gridTriIdxBuf);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridTriIdxBuf);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     collGrid.totalIndices * sizeof(int),
                     collGrid.triIndices,
                     GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, gridTriIdxBuf);

        printf("Collision grid: %dx%dx%d, %d index entries (%.1f KB)\n",
               COLL_GRID_RES,
               COLL_GRID_RES,
               COLL_GRID_RES,
               collGrid.totalIndices,
               collGrid.totalIndices * sizeof(int) / 1024.0f);
    }

    // Initialize LBM grid
    int lbmSizeX = gridX;
    int lbmSizeY = gridY;
    int lbmSizeZ = gridZ;

    // Lattice velocity fixed at Ma ~0.087 for stability.
    // (cs = 1/sqrt(3) ~ 0.577, so Ma = 0.05/0.577 = 0.087)
    float latticeVelocity = 0.05f;

    // Derive Reynolds number from wind speed when --reynolds is
    // not given. This makes the wind speed slider actually change
    // the flow physics instead of only affecting particle viz.
    // Scale: 200 * windSpeed, so wind=1 -> Re=200, wind=5 -> Re=1000.
    // The tau clamp caps the effective Re on coarse grids.
    float scaleX = lbmSizeX / 8.0f;
    float charLength = (carBounds.maxX - carBounds.minX) * scaleX;

    if (reynoldsNumber <= 0) {
        reynoldsNumber = 200.0f * windSpeed;
        if (reynoldsNumber < 50.0f)
            reynoldsNumber = 50.0f;
    }

    float lbmViscosity = (latticeVelocity * charLength) / reynoldsNumber;
    float tau = 3.0f * lbmViscosity + 0.5f;
    if (tau < 0.52f) {
        lbmViscosity = (0.52f - 0.5f) / 3.0f;
        tau = 0.52f;
        float actualRe = (latticeVelocity * charLength) / lbmViscosity;
        printf("  Re capped at %.0f (tau would be too low for "
               "Re=%.0f)\n",
               actualRe,
               reynoldsNumber);
        reynoldsNumber = actualRe;
    }
    printf("Reynolds number: %.0f\n", reynoldsNumber);
    printf("  Char length: %.1f lattice units\n", charLength);
    printf("  Viscosity: %.6f\n", lbmViscosity);
    printf("  tau: %.4f\n", tau);
    printf("  CFL: %.4f\n", latticeVelocity);

    printf("Initializing LBM grid...\n");
    lbmGrid = LBM_Create(lbmSizeX, lbmSizeY, lbmSizeZ, lbmViscosity);

    if (lbmGrid) {
        if (triangleData && numTriangles > 0) {
            // Use actual mesh for LBM solid
            LBM_SetSolidMesh(lbmGrid,
                             (float *)triangleData,
                             numTriangles,
                             carBounds.minX,
                             carBounds.minY,
                             carBounds.minZ,
                             carBounds.maxX,
                             carBounds.maxY,
                             carBounds.maxZ);
        } else {
            // Fallback to AABB
            LBM_SetSolidAABB(lbmGrid,
                             carBounds.minX,
                             carBounds.minY,
                             carBounds.minZ,
                             carBounds.maxX,
                             carBounds.maxY,
                             carBounds.maxZ);
        }
        // Periodic lateral boundaries -- correct for external aero
        // when blockage ratio < 5%.
        lbmGrid->periodicYZ = 1;

        // Enable Smagorinsky SGS turbulence model by default.
        // Adds local eddy viscosity so we can simulate at
        // effectively higher Re on coarse grids.
        lbmGrid->useSmagorinsky = 1;
        lbmGrid->smagorinskyCs = smagorinskyCs;
        printf("Smagorinsky SGS enabled (Cs=%.2f), periodic YZ\n",
               lbmGrid->smagorinskyCs);

        if (useMRT) {
            lbmGrid->useMRT = 1;
            lbmGrid->useRegularized = 0;
            printf("MRT collision enabled\n");
        } else if (lbmGrid->tau < 0.6f) {
            // Auto-enable regularized when tau is low.
            lbmGrid->useRegularized = 1;
            printf("Regularized collision enabled "
                   "(tau=%.3f)\n",
                   lbmGrid->tau);
        }

        LBM_InitializeFlow(lbmGrid, latticeVelocity, 0.0f, 0.0f);

        // Print effective Reynolds number
        {
            float sX = lbmGrid->sizeX / 8.0f;
            float charL = (carBounds.maxX - carBounds.minX) * sX;
            float latU = latticeVelocity;
            float nu = (lbmGrid->tau - 0.5f) / 3.0f;
            float re = (nu > 1e-10f) ? latU * charL / nu : 0;
            printf("Effective Re = %.0f "
                   "(U=%.3f, L=%.1f, nu=%.4f)\n",
                   re,
                   latU,
                   charL,
                   nu);
        }
        printf("  Sample interval: %d lattice steps\n", lbmSubsteps);
        printf("LBM initialized successfully\n");
    } else {
        printf("Warning: LBM initialization failed, using simple wind\n");
        useLBM = 0;
    }

    // Load ML surrogate model for instant Cd prediction.
    // Fails gracefully if weight files are missing.
    MLPredictor *mlModel = ML_Load("model.bin", "model_norm.bin");
    if (!mlModel)
        mlModel = ML_Load("ml/model.bin", "ml/model_norm.bin");
    if (!mlModel)
        mlModel = ML_Load("../ml/model.bin", "../ml/model_norm.bin");
    if (mlModel) {
        // Map model path to model_id (0=car, 1=ahmed25, 2=ahmed35)
        float modelId = 0.0f;
        if (strstr(modelPath, "ahmed_25") || strstr(modelPath, "ahmed25"))
            modelId = 1.0f;
        else if (strstr(modelPath, "ahmed_35") || strstr(modelPath, "ahmed35"))
            modelId = 2.0f;

        float mlCd, mlCl;
        ML_Predict(mlModel, windSpeed, reynoldsNumber, modelId, &mlCd, &mlCl);
        printf("ML estimate: Cd=%.3f Cl=%.3f (instant)\n", mlCd, mlCl);
    } else {
        printf("ML model not loaded (no weight files), "
               "skipping instant prediction\n");
    }

    // Get compute shader uniform locations
    glUseProgram(computeShaderProgram);
    GLint dtLoc = glGetUniformLocation(computeShaderProgram, "dt");
    GLint windLoc = glGetUniformLocation(computeShaderProgram, "wind");
    GLint carMinLoc = glGetUniformLocation(computeShaderProgram, "carMin");
    GLint carMaxLoc = glGetUniformLocation(computeShaderProgram, "carMax");
    GLint carCenterLoc =
        glGetUniformLocation(computeShaderProgram, "carCenter");
    GLint collisionModeLoc =
        glGetUniformLocation(computeShaderProgram, "collisionMode");
    GLint numTrianglesLoc =
        glGetUniformLocation(computeShaderProgram, "numTriangles");
    GLint useLBMLoc = glGetUniformLocation(computeShaderProgram, "useLBM");
    GLint lbmGridSizeLoc =
        glGetUniformLocation(computeShaderProgram, "lbmGridSize");
    GLint timeLoc = glGetUniformLocation(computeShaderProgram, "time");
    GLint gridMinLoc = glGetUniformLocation(computeShaderProgram, "gridMin");
    GLint gridCellSizeLoc =
        glGetUniformLocation(computeShaderProgram, "gridCellSize");
    GLint gridResLoc = glGetUniformLocation(computeShaderProgram, "gridRes");
    GLint computeVizModeLoc =
        glGetUniformLocation(computeShaderProgram, "vizMode");

    printf("Compute shader uniform locations:\n");
    printf("  dt=%d, wind=%d, carMin=%d, carMax=%d, carCenter=%d\n",
           dtLoc,
           windLoc,
           carMinLoc,
           carMaxLoc,
           carCenterLoc);
    printf("  collisionMode=%d, numTriangles=%d, useLBM=%d, lbmGridSize=%d\n",
           collisionModeLoc,
           numTrianglesLoc,
           useLBMLoc,
           lbmGridSizeLoc);

    printf("Allocating %d particles on heap...\n", GPU_PARTICLES);
    Particle *particles = (Particle *)malloc(GPU_PARTICLES * sizeof(Particle));
    if (!particles) {
        printf("Failed to allocate particle memory!\n");
        freeModel(&carModel);
        glDeleteProgram(particleShaderProgram);
        glDeleteProgram(computeShaderProgram);
        GLContext_Destroy(glCtx);
        return 1;
    }

    for (int i = 0; i < GPU_PARTICLES; i++) {
        // Spread particles across the full domain so the flow looks
        // continuous from the first frame instead of a single wave.
        particles[i].x = -4.0f + ((float)rand() / RAND_MAX) * 8.0f;
        particles[i].y = ((float)rand() / RAND_MAX - 0.5f) * 2.6f;
        particles[i].z = ((float)rand() / RAND_MAX - 0.5f) * 2.6f;
        particles[i].padding1 = 0.0f;
        particles[i].vx = 0.3f + ((float)rand() / RAND_MAX) * 0.1f;
        particles[i].vy = 0.0f;
        particles[i].vz = 0.0f;
        // Stagger lifetimes so particles respawn at different times
        // instead of all at once.
        particles[i].life = 0.2f + ((float)rand() / RAND_MAX) * 0.8f;
    }

    printf("Creating particle buffer...\n");
    GLuint particleBuffer;
    glGenBuffers(1, &particleBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 GPU_PARTICLES * sizeof(Particle),
                 particles,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particleBuffer);
    checkGLError("After creating particle buffer");

    GLuint particleVAO;
    glGenVertexArrays(1, &particleVAO);
    glBindVertexArray(particleVAO);

    glBindBuffer(GL_ARRAY_BUFFER, particleBuffer);

    // Attr 0: position (vec3 at offset 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void *)0);

    // Attr 1: flowSpeed (float at offset 12 -- stored in padding1)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,
                          1,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(Particle),
                          (void *)(3 * sizeof(float)));

    // Attr 2: velocity (vec3 at offset 16)
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(Particle),
                          (void *)(4 * sizeof(float)));

    // Attr 3: life (float at offset 28)
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3,
                          1,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(Particle),
                          (void *)(7 * sizeof(float)));

    glBindVertexArray(0);
    checkGLError("After creating VAO");

// Streamline trail buffer and rendering setup
#define TRAIL_LEN 32
    GLuint trailBuffer;
    {
        size_t trailSize = GPU_PARTICLES * TRAIL_LEN * 4 * sizeof(float);
        void *trailZeros = calloc(1, trailSize);
        glGenBuffers(1, &trailBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, trailBuffer);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER, trailSize, trailZeros, GL_DYNAMIC_COPY);
        free(trailZeros);
        printf("Trail buffer: %.1f MB (%d particles x %d points)\n",
               trailSize / (1024.0 * 1024.0),
               GPU_PARTICLES,
               TRAIL_LEN);
    }

    // Trail VAO (reads vec4 from trailBuffer)
    GLuint trailVAO;
    glGenVertexArrays(1, &trailVAO);
    glBindVertexArray(trailVAO);
    glBindBuffer(GL_ARRAY_BUFFER, trailBuffer);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glBindVertexArray(0);

    // Trail index buffer with primitive restart
    GLuint trailEBO;
    {
        int idxCount = GPU_PARTICLES * (TRAIL_LEN + 1);
        GLuint *indices = (GLuint *)malloc(idxCount * sizeof(GLuint));
        int k = 0;
        for (int i = 0; i < GPU_PARTICLES; i++) {
            for (int j = 0; j < TRAIL_LEN; j++)
                indices[k++] = i * TRAIL_LEN + j;
            indices[k++] = 0xFFFFFFFF; // restart
        }
        glGenBuffers(1, &trailEBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, trailEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     k * sizeof(GLuint),
                     indices,
                     GL_STATIC_DRAW);
        free(indices);
    }

    // Trail shaders
    GLuint trailUpdateShader = createComputeShader("shaders/trail_update.comp");
    GLuint trailRenderProgram =
        createShaderProgram("shaders/trail.vert", "shaders/trail.frag");

    checkGLError("After trail setup");

    // RK4 streamline buffer and rendering setup
#define STREAMLINE_SEEDS 256
#define STREAMLINE_LEN 64
    GLuint streamlineBuffer;
    {
        size_t slSize =
            STREAMLINE_SEEDS * STREAMLINE_LEN * 4 * sizeof(float);
        void *slZeros = calloc(1, slSize);
        glGenBuffers(1, &streamlineBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, streamlineBuffer);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER, slSize, slZeros, GL_DYNAMIC_COPY);
        free(slZeros);
    }

    // Streamline VAO (same layout as trail: vec4 per vertex)
    GLuint streamlineVAO;
    glGenVertexArrays(1, &streamlineVAO);
    glBindVertexArray(streamlineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, streamlineBuffer);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glBindVertexArray(0);

    // Streamline EBO with primitive restart
    GLuint streamlineEBO;
    {
        int idxCount = STREAMLINE_SEEDS * (STREAMLINE_LEN + 1);
        GLuint *indices = (GLuint *)malloc(idxCount * sizeof(GLuint));
        int k = 0;
        for (int i = 0; i < STREAMLINE_SEEDS; i++) {
            for (int j = 0; j < STREAMLINE_LEN; j++)
                indices[k++] = i * STREAMLINE_LEN + j;
            indices[k++] = 0xFFFFFFFF;
        }
        glGenBuffers(1, &streamlineEBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, streamlineEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     k * sizeof(GLuint),
                     indices,
                     GL_STATIC_DRAW);
        free(indices);
    }

    GLuint streamlineTraceShader =
        createComputeShader("shaders/streamline_trace.comp");

    checkGLError("After streamline setup");

    printf("Creating fluid cube...\n");
    FluidCube *fluidCube = NULL;
    if (carModel.vertexCount > 0) {
        fluidCube = FluidCubeCreate(
            WIDTH / 10, HEIGHT / 10, 20, 0.001f, 0.0f, 0.001f, &carModel);
    }

    // Print GPU memory estimate
    {
        size_t lbmBytes = 0;
        if (lbmGrid) {
            size_t cells =
                (size_t)lbmGrid->sizeX * lbmGrid->sizeY * lbmGrid->sizeZ;
            lbmBytes += cells * 19 * sizeof(float) * 2; // f + fNew
            lbmBytes += cells * 4 * sizeof(float);      // velocity
            lbmBytes += cells * sizeof(int);            // solid
            lbmBytes += 4 * sizeof(int);                // force
        }
        size_t triBytes = numTriangles * sizeof(GPUTriangle);
        size_t particleBytes = GPU_PARTICLES * sizeof(Particle);
        size_t totalBytes = lbmBytes + triBytes + particleBytes;

        printf("\nGPU memory estimate:\n");
        if (lbmGrid) {
            printf("  LBM buffers:      %.1f MB (%dx%dx%d grid)\n",
                   lbmBytes / (1024.0 * 1024.0),
                   lbmGrid->sizeX,
                   lbmGrid->sizeY,
                   lbmGrid->sizeZ);
        }
        printf("  Triangle mesh:    %.1f MB (%d triangles)\n",
               triBytes / (1024.0 * 1024.0),
               numTriangles);
        printf("  Particle system:  %.1f MB (%d particles)\n",
               particleBytes / (1024.0 * 1024.0),
               GPU_PARTICLES);
        printf("  Total:            %.1f MB\n", totalBytes / (1024.0 * 1024.0));
    }

    printf("\n========================================\n");
    printf("Initialization complete. Starting main loop...\n");
    printf("========================================\n\n");
    printf("CONTROLS\n");
    printf("----------------------------------------\n");
    printf("Mouse drag:     Rotate camera\n");
    printf("Shift+drag:     Pan camera\n");
    printf("Middle drag:    Pan camera\n");
    printf("Scroll wheel:   Zoom in/out\n");
    printf("A/D:            Rotate left/right\n");
    printf("W/S:            Rotate up/down\n");
    printf("Q/E:            Zoom in/out (step)\n");
    printf("R:              Reset camera\n");
    printf("F1:             Front view\n");
    printf("F2:             Side view\n");
    printf("F3:             Top view\n");
    printf("F4:             Isometric view\n");
    printf("UP/DOWN:        Adjust wind speed\n");
    printf("LEFT/RIGHT:     Adjust max speed scale\n");
    printf("\nCOLLISION MODES:\n");
    printf("0:              Collision OFF\n");
    printf("1:              AABB collision (fast)\n");
    printf("2:              Per-triangle collision (accurate)\n");
    printf("\nVISUALIZATION MODES:\n");
    printf("V:              Cycle visualization mode\n");
    printf("3-9:            Select specific mode\n");
    printf("\nL:              Toggle LBM flow field\n");
    printf("SPACE:          Pause / resume simulation\n");
    printf("PERIOD:         Step one frame (while paused)\n");
    printf("ESC:            Quit\n");
    printf("----------------------------------------\n\n");

    int running = 1;
    Uint32 lastTime = SDL_GetTicks();
    int frameCount = 0;
    int outputFrameCount = 0;
    int maxFrames = (renderDuration > 0) ? renderDuration * 60 : 0;

// Convergence detection for auto-stop
#define CD_HISTORY_SIZE 100
    float cdHistory[CD_HISTORY_SIZE];
    int cdHistoryCount = 0;
    int converged = 0;
    float cdEma = 0.0f;

    // Cl time series for Strouhal extraction (dynamically grown)
    int clCapacity = 256;
    int clCount = 0;
    float *clSeries = (float *)malloc(clCapacity * sizeof(float));

    while (running) {
        if (!paused || stepOnce)
            frameCount++;

        glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = 0;
            } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    mouseDown = 1;
                    lastMouseX = event.button.x;
                    lastMouseY = event.button.y;
                } else if (event.button.button == SDL_BUTTON_MIDDLE) {
                    middleMouseDown = 1;
                    lastMouseX = event.button.x;
                    lastMouseY = event.button.y;
                }
            } else if (event.type == SDL_MOUSEBUTTONUP) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    mouseDown = 0;
                } else if (event.button.button == SDL_BUTTON_MIDDLE) {
                    middleMouseDown = 0;
                }
            } else if (event.type == SDL_MOUSEMOTION) {
                int dx = event.motion.x - lastMouseX;
                int dy = event.motion.y - lastMouseY;
                if (mouseDown &&
                    (SDL_GetModState() & KMOD_SHIFT)) {
                    // Shift+left-drag: pan
                    float panScale = cameraDistance * 0.002f;
                    cameraTargetX -=
                        dx * panScale * cosf(cameraAngleY);
                    cameraTargetZ +=
                        dx * panScale * sinf(cameraAngleY);
                    cameraTargetY += dy * panScale;
                    lastMouseX = event.motion.x;
                    lastMouseY = event.motion.y;
                } else if (mouseDown) {
                    cameraAngleY += dx * 0.005f;
                    cameraAngleX += dy * 0.005f;

                    if (cameraAngleX > 1.5f)
                        cameraAngleX = 1.5f;
                    if (cameraAngleX < -1.5f)
                        cameraAngleX = -1.5f;

                    lastMouseX = event.motion.x;
                    lastMouseY = event.motion.y;
                } else if (middleMouseDown) {
                    // Middle-click drag: pan
                    float panScale = cameraDistance * 0.002f;
                    cameraTargetX -=
                        dx * panScale * cosf(cameraAngleY);
                    cameraTargetZ +=
                        dx * panScale * sinf(cameraAngleY);
                    cameraTargetY += dy * panScale;
                    lastMouseX = event.motion.x;
                    lastMouseY = event.motion.y;
                }
            } else if (event.type == SDL_MOUSEWHEEL) {
                cameraDistance -= event.wheel.y * 0.5f;
                if (cameraDistance < 1.0f)
                    cameraDistance = 1.0f;
                if (cameraDistance > 20.0f)
                    cameraDistance = 20.0f;
            } else if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                case SDLK_ESCAPE:
                    running = 0;
                    break;
                case SDLK_0:
                    collisionMode = 0;
                    printf("Collision: OFF\n");
                    break;
                case SDLK_1:
                    collisionMode = 1;
                    printf("Collision: AABB (fast)\n");
                    break;
                case SDLK_2:
                    collisionMode = 2;
                    printf(
                        "Collision: Per-Triangle (accurate) - %d triangles\n",
                        numTriangles);
                    break;
                case SDLK_3:
                    visualizationMode = 0;
                    printf("Visualization: %s\n",
                           vizModeNames[visualizationMode]);
                    break;
                case SDLK_4:
                    visualizationMode = 1;
                    printf("Visualization: %s\n",
                           vizModeNames[visualizationMode]);
                    break;
                case SDLK_5:
                    visualizationMode = 2;
                    printf("Visualization: %s\n",
                           vizModeNames[visualizationMode]);
                    break;
                case SDLK_6:
                    visualizationMode = 3;
                    printf("Visualization: %s\n",
                           vizModeNames[visualizationMode]);
                    break;
                case SDLK_7:
                    visualizationMode = 4;
                    printf("Visualization: %s\n",
                           vizModeNames[visualizationMode]);
                    break;
                case SDLK_8:
                    visualizationMode = 5;
                    printf("Visualization: %s\n",
                           vizModeNames[visualizationMode]);
                    break;
                case SDLK_9:
                    visualizationMode = 6;
                    printf("Visualization: %s\n",
                           vizModeNames[visualizationMode]);
                    break;
                case SDLK_v:
                    visualizationMode = (visualizationMode + 1) % numVizModes;
                    printf("Visualization: %s\n",
                           vizModeNames[visualizationMode]);
                    break;
                case SDLK_UP:
                    windSpeed += 0.5f;
                    printf("Wind speed: %.1f\n", windSpeed);
                    break;
                case SDLK_DOWN:
                    windSpeed -= 0.5f;
                    if (windSpeed < 0.0f)
                        windSpeed = 0.0f;
                    printf("Wind speed: %.1f\n", windSpeed);
                    break;
                case SDLK_LEFT:
                    maxSpeed -= 0.2f;
                    if (maxSpeed < 0.2f)
                        maxSpeed = 0.2f;
                    printf("Max speed scale: %.1f\n", maxSpeed);
                    break;
                case SDLK_RIGHT:
                    maxSpeed += 0.2f;
                    if (maxSpeed > 10.0f)
                        maxSpeed = 10.0f;
                    printf("Max speed scale: %.1f\n", maxSpeed);
                    break;
                case SDLK_a:
                    cameraAngleY -= 0.1f;
                    break;
                case SDLK_d:
                    cameraAngleY += 0.1f;
                    break;
                case SDLK_w:
                    cameraAngleX -= 0.1f;
                    if (cameraAngleX < -1.5f)
                        cameraAngleX = -1.5f;
                    break;
                case SDLK_s:
                    cameraAngleX += 0.1f;
                    if (cameraAngleX > 1.5f)
                        cameraAngleX = 1.5f;
                    break;
                case SDLK_q:
                    cameraDistance -= 0.5f;
                    if (cameraDistance < 1.0f)
                        cameraDistance = 1.0f;
                    break;
                case SDLK_e:
                    cameraDistance += 0.5f;
                    if (cameraDistance > 20.0f)
                        cameraDistance = 20.0f;
                    break;
                case SDLK_r:
                    cameraAngleY = 0.0f;
                    cameraAngleX = 0.3f;
                    cameraDistance = 6.0f;
                    cameraTargetX = -1.5f;
                    cameraTargetY = 0.0f;
                    cameraTargetZ = 0.0f;
                    printf("Camera reset\n");
                    break;
                case SDLK_F1:
                    // Front view
                    cameraAngleY = 0.0f;
                    cameraAngleX = 0.0f;
                    cameraDistance = 6.0f;
                    printf("Camera: front\n");
                    break;
                case SDLK_F2:
                    // Side view
                    cameraAngleY = 1.5708f;
                    cameraAngleX = 0.0f;
                    cameraDistance = 6.0f;
                    printf("Camera: side\n");
                    break;
                case SDLK_F3:
                    // Top view
                    cameraAngleY = 0.0f;
                    cameraAngleX = 1.5f;
                    cameraDistance = 8.0f;
                    printf("Camera: top\n");
                    break;
                case SDLK_F4:
                    // Isometric view
                    cameraAngleY = 0.6f;
                    cameraAngleX = 0.5f;
                    cameraDistance = 7.0f;
                    printf("Camera: isometric\n");
                    break;
                case SDLK_l:
                    useLBM = !useLBM;
                    printf("LBM: %s\n", useLBM ? "ON" : "OFF");
                    break;
                case SDLK_SPACE:
                    paused = !paused;
                    printf("Simulation %s at frame %d\n",
                           paused ? "paused" : "resumed",
                           frameCount);
                    break;
                case SDLK_PERIOD:
                    if (paused)
                        stepOnce = 1;
                    break;
                }
            }
        }

        calculateViewMatrix(view,
                            cameraAngleY,
                            cameraAngleX,
                            cameraDistance,
                            cameraTargetX,
                            cameraTargetY,
                            cameraTargetZ);

        Uint32 currentTime = SDL_GetTicks();
        float deltaTime = (currentTime - lastTime) / 1000.0f;
        lastTime = currentTime;

        // Cap deltaTime so particles don't explode after the skip phase
        // or any long stall. 60fps = 16.7ms, allow up to ~2 frames.
        if (deltaTime > 0.05f)
            deltaTime = 0.016f;

        // Run LBM simulation FIRST (skip when paused unless single-stepping)
        if (lbmGrid && useLBM && (!paused || stepOnce)) {
            // Ramp inlet velocity over first 300 frames (~5s) to avoid
            // impulsive acoustic startup that destabilizes low-tau runs.
            int rampFrames = 300;
            float rampFactor = (frameCount < rampFrames)
                                   ? (float)frameCount / (float)rampFrames
                                   : 1.0f;
            float currentInletVel = latticeVelocity * rampFactor;

            for (int i = 0; i < lbmSubsteps; i++) {
                LBM_Step(lbmGrid, currentInletVel, 0.0f, 0.0f);
            }

            // VTK field dump at specified interval
            if (strlen(vtkOutputPath) > 0 &&
                frameCount % vtkInterval == 0) {
                writeVTI(lbmGrid, vtkOutputPath, frameCount);
            }
        }

        // Fast LBM-only convergence: skip particles and rendering
        // until the flow is developed enough for Cd measurement.
        // This is ~5x faster than rendering every frame.
        int rampEnd = 300;
        int flowThroughFrames =
            lbmGrid
                ? (int)(lbmGrid->sizeX / (latticeVelocity * lbmSubsteps * 2))
                : 150;
        int cdStartFrame = rampEnd + 3 * flowThroughFrames;

        // For short headless runs, cap the skip so we still produce
        // some frames and Cd output rather than exiting with nothing.
        if (maxFrames > 0 && cdStartFrame >= maxFrames) {
            cdStartFrame = maxFrames * 2 / 3;
            if (cdStartFrame < rampEnd && maxFrames > rampEnd)
                cdStartFrame = rampEnd;
        }
        int skipRendering = (renderDuration > 0 && frameCount < cdStartFrame);

        // When the skip phase ends, reinitialize particles at the
        // inlet so they fill the domain naturally as the video plays.
        // Without this, particles sit frozen during the skip and get
        // immediately swept out when rendering resumes.
        if (!skipRendering && frameCount == cdStartFrame &&
            renderDuration > 0) {
            Particle *fresh =
                (Particle *)malloc(GPU_PARTICLES * sizeof(Particle));
            if (fresh) {
                for (int i = 0; i < GPU_PARTICLES; i++) {
                    fresh[i].x = -4.0f + ((float)rand() / RAND_MAX) * 8.0f;
                    fresh[i].y = ((float)rand() / RAND_MAX - 0.5f) * 2.6f;
                    fresh[i].z = ((float)rand() / RAND_MAX - 0.5f) * 2.6f;
                    fresh[i].padding1 = 0.0f;
                    fresh[i].vx = 0.3f + ((float)rand() / RAND_MAX) * 0.1f;
                    fresh[i].vy = 0.0f;
                    fresh[i].vz = 0.0f;
                    fresh[i].life = 0.2f + ((float)rand() / RAND_MAX) * 0.8f;
                }
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleBuffer);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                                0,
                                GPU_PARTICLES * sizeof(Particle),
                                fresh);
                free(fresh);
            }
        }

        if (!skipRendering) {
            // Now set up particle compute shader
            glUseProgram(computeShaderProgram);

            // Bind buffers
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particleBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, triangleBuffer);

            // Bind LBM velocity and solid buffers if using LBM
            if (lbmGrid && useLBM) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER,
                                 4,
                                 LBM_GetVelocityBuffer(lbmGrid));
                glBindBufferBase(
                    GL_SHADER_STORAGE_BUFFER, 5, lbmGrid->solidBuffer);
            }

            // Set uniforms
            if (dtLoc != -1)
                glUniform1f(dtLoc, deltaTime);
            if (windLoc != -1)
                glUniform3f(windLoc, windSpeed * 0.5f, 0.0f, 0.0f);
            if (carMinLoc != -1)
                glUniform3f(
                    carMinLoc, carBounds.minX, carBounds.minY, carBounds.minZ);
            if (carMaxLoc != -1)
                glUniform3f(
                    carMaxLoc, carBounds.maxX, carBounds.maxY, carBounds.maxZ);
            if (carCenterLoc != -1)
                glUniform3f(carCenterLoc,
                            carBounds.centerX,
                            carBounds.centerY,
                            carBounds.centerZ);
            if (collisionModeLoc != -1)
                glUniform1i(collisionModeLoc, collisionMode);
            if (numTrianglesLoc != -1)
                glUniform1i(numTrianglesLoc, numTriangles);

            // LBM uniforms
            if (useLBMLoc != -1)
                glUniform1i(useLBMLoc, (useLBM && lbmGrid) ? 1 : 0);
            if (lbmGridSizeLoc != -1 && lbmGrid) {
                glUniform3i(lbmGridSizeLoc,
                            lbmGrid->sizeX,
                            lbmGrid->sizeY,
                            lbmGrid->sizeZ);
            }

            // Collision grid uniforms
            if (gridMinLoc != -1)
                glUniform3f(
                    gridMinLoc, collGrid.minX, collGrid.minY, collGrid.minZ);
            if (gridCellSizeLoc != -1)
                glUniform3f(gridCellSizeLoc,
                            collGrid.cellSizeX,
                            collGrid.cellSizeY,
                            collGrid.cellSizeZ);
            if (gridResLoc != -1)
                glUniform3i(
                    gridResLoc, COLL_GRID_RES, COLL_GRID_RES, COLL_GRID_RES);

            // Bind grid SSBOs
            if (gridCellStartBuf)
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, gridCellStartBuf);
            if (gridCellCountBuf)
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, gridCellCountBuf);
            if (gridTriIdxBuf)
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, gridTriIdxBuf);

            // Visualization mode for pressure coloring in compute
            if (computeVizModeLoc != -1)
                glUniform1i(computeVizModeLoc, visualizationMode);

            // Time uniform for randomness
            if (timeLoc != -1)
                glUniform1f(timeLoc, (float)SDL_GetTicks() / 1000.0f);

            // Dispatch particle compute (skip when paused unless
            // single-stepping)
            if (!paused || stepOnce) {
                glDispatchCompute((GPU_PARTICLES + 255) / 256, 1, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

                // Update pathline trails after particle positions are written
                if (trailUpdateShader && visualizationMode == 7) {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, trailBuffer);
                    glUseProgram(trailUpdateShader);
                    glUniform1i(
                        glGetUniformLocation(trailUpdateShader, "trailLen"),
                        TRAIL_LEN);
                    glUniform1i(
                        glGetUniformLocation(trailUpdateShader, "numParticles"),
                        GPU_PARTICLES);
                    glDispatchCompute((GPU_PARTICLES + 255) / 256, 1, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                    GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
                }

                // RK4 streamline trace (recompute every frame)
                if (streamlineTraceShader && visualizationMode == 9 &&
                    lbmGrid && useLBM) {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4,
                                     LBM_GetVelocityBuffer(lbmGrid));
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11,
                                     streamlineBuffer);
                    glUseProgram(streamlineTraceShader);
                    glUniform3i(
                        glGetUniformLocation(streamlineTraceShader,
                                             "lbmGridSize"),
                        lbmGrid->sizeX, lbmGrid->sizeY, lbmGrid->sizeZ);
                    glUniform1i(
                        glGetUniformLocation(streamlineTraceShader, "numSeeds"),
                        STREAMLINE_SEEDS);
                    glUniform1i(
                        glGetUniformLocation(streamlineTraceShader, "traceLen"),
                        STREAMLINE_LEN);
                    glDispatchCompute(
                        (STREAMLINE_SEEDS + 255) / 256, 1, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                    GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
                }
            }

            if (visualizationMode == 7 && trailRenderProgram) {
                // Pathline mode: render trails as line strips
                glUseProgram(trailRenderProgram);
                glUniformMatrix4fv(
                    glGetUniformLocation(trailRenderProgram, "projection"),
                    1,
                    GL_FALSE,
                    projection);
                glUniformMatrix4fv(
                    glGetUniformLocation(trailRenderProgram, "view"),
                    1,
                    GL_FALSE,
                    view);
                glUniform1i(
                    glGetUniformLocation(trailRenderProgram, "trailLen"),
                    TRAIL_LEN);
                glUniform1f(
                    glGetUniformLocation(trailRenderProgram, "maxSpeed"),
                    maxSpeed);

                glEnable(GL_PRIMITIVE_RESTART);
                glPrimitiveRestartIndex(0xFFFFFFFF);

                glBindVertexArray(trailVAO);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, trailEBO);
                glDrawElements(GL_LINE_STRIP,
                               GPU_PARTICLES * (TRAIL_LEN + 1),
                               GL_UNSIGNED_INT,
                               0);

                glDisable(GL_PRIMITIVE_RESTART);
                checkGLError("After rendering pathlines");
            } else if (visualizationMode == 9 && trailRenderProgram) {
                // Streamline mode: render RK4 traces as line strips
                glUseProgram(trailRenderProgram);
                glUniformMatrix4fv(
                    glGetUniformLocation(trailRenderProgram, "projection"),
                    1,
                    GL_FALSE,
                    projection);
                glUniformMatrix4fv(
                    glGetUniformLocation(trailRenderProgram, "view"),
                    1,
                    GL_FALSE,
                    view);
                glUniform1i(
                    glGetUniformLocation(trailRenderProgram, "trailLen"),
                    STREAMLINE_LEN);
                glUniform1f(
                    glGetUniformLocation(trailRenderProgram, "maxSpeed"),
                    maxSpeed);

                glEnable(GL_PRIMITIVE_RESTART);
                glPrimitiveRestartIndex(0xFFFFFFFF);

                glBindVertexArray(streamlineVAO);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, streamlineEBO);
                glDrawElements(GL_LINE_STRIP,
                               STREAMLINE_SEEDS * (STREAMLINE_LEN + 1),
                               GL_UNSIGNED_INT,
                               0);

                glDisable(GL_PRIMITIVE_RESTART);
                checkGLError("After rendering streamlines");
            } else {
                // Normal particle rendering (modes 0-6)
                glUseProgram(particleShaderProgram);
                if (viewLoc != -1)
                    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);
                if (vizModeLoc != -1)
                    glUniform1i(vizModeLoc, visualizationMode);
                if (maxSpeedLoc != -1)
                    glUniform1f(maxSpeedLoc, maxSpeed);

                glBindVertexArray(particleVAO);
                glDrawArrays(GL_POINTS, 0, GPU_PARTICLES);
                checkGLError("After rendering particles");
            }

            // Render car model
            if (carModel.faceCount > 0) {
                glUseProgram(0);

                glMatrixMode(GL_PROJECTION);
                glPushMatrix();
                glLoadMatrixf(projection);

                glMatrixMode(GL_MODELVIEW);
                glPushMatrix();
                glLoadMatrixf(view);

                renderModel(&carModel, SCALE);

                glPopMatrix();
                glMatrixMode(GL_PROJECTION);
                glPopMatrix();
                glMatrixMode(GL_MODELVIEW);

                checkGLError("After rendering model");
            }
        } // end skipRendering

        // Compute and display drag coefficient every 20 frames
        // once the flow has developed (cdStartFrame computed above).
        if (frameCount >= cdStartFrame && frameCount % 20 == 0 && lbmGrid &&
            useLBM) {
            // Compute force with pressure/friction decomposition
            float fx, fy, fz, px, py, pz;
            LBM_ComputeDragForceDecomposed(lbmGrid, &fx, &fy, &fz,
                                           &px, &py, &pz);

            // Reference area ~ car frontal area in lattice units
            float scaleY = lbmGrid->sizeY / 4.0f;
            float scaleZ = lbmGrid->sizeZ / 4.0f;
            float refArea = (carBounds.maxY - carBounds.minY) * scaleY *
                            (carBounds.maxZ - carBounds.minZ) * scaleZ;
            float dynP = 0.5f * latticeVelocity * latticeVelocity;
            float denom = dynP * refArea;
            float Cd = (denom > 1e-10f) ? fabsf(fx) / denom : 0.0f;
            float Cl = (denom > 1e-10f) ? fabsf(fy) / denom : 0.0f;
            float CdPressure = (denom > 1e-10f) ? fabsf(px) / denom : 0.0f;
            float CdFriction = Cd - CdPressure;

            // Exponential moving average for stable reporting
            if (cdHistoryCount == 0) {
                cdEma = Cd;
            } else {
                float alpha = 0.1f;
                cdEma = alpha * Cd + (1.0f - alpha) * cdEma;
            }

            int totalSteps = frameCount * lbmSubsteps;
            float tStar = (charLength > 0)
                              ? (float)totalSteps * latticeVelocity / charLength
                              : 0.0f;
            float flowThroughs =
                (lbmGrid && lbmGrid->sizeX > 0)
                    ? (float)totalSteps * latticeVelocity / lbmGrid->sizeX
                    : 0.0f;

            printf("  Drag Force: (%.4f, %.4f, %.4f),"
                   " Cd=%.3f Cl=%.3f (avg=%.3f)"
                   " t*=%.3f flow-throughs=%.2f\n",
                   fx,
                   fy,
                   fz,
                   Cd,
                   Cl,
                   cdEma,
                   tStar,
                   flowThroughs);
            printf("  Cd_pressure=%.3f Cd_friction=%.3f\n",
                   CdPressure,
                   CdFriction);

            // Store Cl for Strouhal extraction
            if (clSeries) {
                if (clCount >= clCapacity) {
                    clCapacity *= 2;
                    clSeries =
                        (float *)realloc(clSeries, clCapacity * sizeof(float));
                }
                if (clSeries)
                    clSeries[clCount++] = Cl;
            }

            // Track Cd for convergence detection
            if (Cd > 0 && Cd < 1000) {
                cdHistory[cdHistoryCount % CD_HISTORY_SIZE] = Cd;
                cdHistoryCount++;

                if (cdHistoryCount >= CD_HISTORY_SIZE && !converged) {
                    float mean = 0;
                    for (int j = 0; j < CD_HISTORY_SIZE; j++)
                        mean += cdHistory[j];
                    mean /= CD_HISTORY_SIZE;

                    float var = 0;
                    for (int j = 0; j < CD_HISTORY_SIZE; j++) {
                        float d = cdHistory[j] - mean;
                        var += d * d;
                    }
                    float relStd =
                        sqrtf(var / CD_HISTORY_SIZE) / (mean + 1e-10f);

                    if (relStd < 0.01f) {
                        converged = 1;
                        printf("  Cd converged (mean=%.3f,"
                               " relStd=%.4f)\n",
                               mean,
                               relStd);
                        // Auto-stop in headless mode
                        if (maxFrames > 0) {
                            // Run 2 more seconds for clean video ending
                            int extra = 120;
                            if (outputFrameCount + extra < maxFrames)
                                maxFrames = outputFrameCount + extra;
                        }
                    }
                }
            }
        }

        if (maxFrames > 0 && outputFrameCount >= maxFrames) {
            printf("Render complete: %d frames\n", outputFrameCount);
            running = 0;
        }

        if (!skipRendering && renderDuration > 0 && strlen(outputPath) > 0) {
            char framePath[512];
            snprintf(framePath,
                     sizeof(framePath),
                     "%s/frame_%05d.ppm",
                     outputPath,
                     outputFrameCount);
            saveFrameToPPM(framePath, WIDTH, HEIGHT);
            outputFrameCount++;
        }

        stepOnce = 0;
        if (!skipRendering)
            GLContext_SwapBuffers(glCtx);
    }

    // Strouhal number extraction from Cl time series.
    // St = f_peak * L / U, where f_peak is the dominant shedding
    // frequency found via DFT of the Cl signal.
    if (clSeries && clCount >= 12 && useLBM) {
        // Discard first 35% as transient
        int start = clCount * 35 / 100;
        int n = clCount - start;
        float *sig = clSeries + start;

        // Remove mean
        float mean = 0;
        for (int i = 0; i < n; i++)
            mean += sig[i];
        mean /= n;

        // Hann window + mean removal
        float *win = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (n - 1)));
            win[i] = (sig[i] - mean) * w;
        }

        // DFT over one-sided spectrum (skip DC).
        // Each Cl sample corresponds to lbmSubsteps lattice timesteps.
        float fs = 1.0f / lbmSubsteps;
        int nfreqs = n / 2;
        float peakPow = 0;
        float peakFreq = 0;
        for (int k = 1; k < nfreqs; k++) {
            float re = 0, im = 0;
            for (int i = 0; i < n; i++) {
                float angle = 2.0f * (float)M_PI * k * i / n;
                re += win[i] * cosf(angle);
                im -= win[i] * sinf(angle);
            }
            float pw = re * re + im * im;
            if (pw > peakPow) {
                peakPow = pw;
                peakFreq = (float)k * fs / n;
            }
        }

        float charL = charLength; // lattice units, set during LBM init
        float st =
            (latticeVelocity > 1e-10f) ? peakFreq * charL / latticeVelocity : 0;
        printf("St=%.4f (f=%.6f, L=%.1f, U=%.3f)\n",
               st,
               peakFreq,
               charL,
               latticeVelocity);
        free(win);
    }

    printf("Cleaning up...\n");
    free(particles);
    if (triangleData)
        free(triangleData);
    free(clSeries);
    free(collGrid.cellStart);
    free(collGrid.cellCount);
    free(collGrid.triIndices);
    freeModel(&carModel);
    if (fluidCube)
        FluidCubeFree(fluidCube);
    if (lbmGrid)
        LBM_Free(lbmGrid);
    if (mlModel)
        ML_Free(mlModel);
    glDeleteVertexArrays(1, &particleVAO);
    glDeleteBuffers(1, &particleBuffer);
    glDeleteVertexArrays(1, &trailVAO);
    glDeleteBuffers(1, &trailBuffer);
    glDeleteBuffers(1, &trailEBO);
    glDeleteVertexArrays(1, &streamlineVAO);
    glDeleteBuffers(1, &streamlineBuffer);
    glDeleteBuffers(1, &streamlineEBO);
    if (streamlineTraceShader)
        glDeleteProgram(streamlineTraceShader);
    if (trailUpdateShader)
        glDeleteProgram(trailUpdateShader);
    if (trailRenderProgram)
        glDeleteProgram(trailRenderProgram);
    if (triangleBuffer)
        glDeleteBuffers(1, &triangleBuffer);
    glDeleteProgram(particleShaderProgram);
    glDeleteProgram(computeShaderProgram);
    GLContext_Destroy(glCtx);

    printf("Cleanup complete. Exiting.\n");
    printf("Thanks for Running my project! If you liked it please consider "
           "contributing or leaving a star on GitHub!");
    return 0;
}
