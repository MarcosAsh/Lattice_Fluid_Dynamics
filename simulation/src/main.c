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

#include "../lib/gl_context.h"
#include "../lib/lbm.h"
#include "../lib/fluid_cube.h"
#include "../lib/particle_system.h"
#include "../obj-file-loader/lib/model_loader.h"
#include "../lib/render_model.h"
#include "../lib/opengl_utils.h"
#include "../lib/config.h"

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
float cameraTargetX = 0.0f;
float cameraTargetY = 0.0f;
float cameraTargetZ = 0.0f;

// Mouse control variables
int mouseDown = 0;
int lastMouseX = 0;
int lastMouseY = 0;

// Visualization mode
int visualizationMode = 1;
float maxSpeed = 2.0f;

// LBM settings
int useLBM = 1;
LBMGrid *lbmGrid = NULL;
int lbmSubsteps = 5;

// Pause / single-step
int paused = 0;
int stepOnce = 0;

const char *vizModeNames[] = {"Depth",
                              "Velocity Magnitude",
                              "Velocity Direction",
                              "Particle Lifetime",
                              "Turbulence/Pressure",
                              "Flow Progress",
                              "Vorticity"};
const int numVizModes = 7;

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
    forward[0] /= forwardLength;
    forward[1] /= forwardLength;
    forward[2] /= forwardLength;

    float up[3] = {0.0f, 1.0f, 0.0f};

    float side[3] = {forward[1] * up[2] - forward[2] * up[1],
                     forward[2] * up[0] - forward[0] * up[2],
                     forward[0] * up[1] - forward[1] * up[0]};
    float sideLength =
        sqrtf(side[0] * side[0] + side[1] * side[1] + side[2] * side[2]);
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

int main(int argc, char *argv[]) {
    printf("Starting 3D Fluid Simulation...\n");
    srand(time(NULL));

    // Parse command line arguments
    float windSpeed = 1.0f;
    int visualizationMode = 1;
    int collisionMode = 1;
    int renderDuration = 0;
    char outputPath[256] = "";
    char modelPath[512] = "assets/3d-files/car-model.obj";
    int slantAngle = 0;
    float reynoldsNumber = 0.0f;
    int gridX = 128, gridY = 64, gridZ = 64;

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
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}

    };

    int opt;
    while ((opt = getopt_long(
                argc, argv, "w:v:c:d:o:m:a:r:s:g:h", long_options, NULL)) != -1) {
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
            if (collisionMode > 2)
                collisionMode = 2;
            break;
        case 'd':
            renderDuration = atoi(optarg);
            if (renderDuration < 0)
                renderDuration = 0;
            break;
        case 'o':
            strncpy(outputPath, optarg, 255);
            break;
        case 'm':
            strncpy(modelPath, optarg, 511);
            break;
        case 'a':
            slantAngle = atoi(optarg);
            if (slantAngle == 25) {
                strncpy(modelPath, "assets/3d-files/ahmed-25deg.obj", 511);
            } else if (slantAngle == 35) {
                strncpy(modelPath, "assets/3d-files/ahmed-35deg.obj", 511);
            }
            break;
        case 's':
            g_modelScale = atof(optarg);
            break;
        case 'r':
            reynoldsNumber = atof(optarg);
            if (reynoldsNumber < 0)
                reynoldsNumber = 0;
            break;
        case 'g':
            if (sscanf(optarg, "%dx%dx%d",
                        &gridX, &gridY, &gridZ) != 3) {
                /* Try single number: NxN/2xN/2 */
                int n = atoi(optarg);
                if (n > 0) {
                    gridX = n;
                    gridY = n / 2;
                    gridZ = n / 2;
                }
            }
            break;
        case 'h':
        default:
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -w, --wind=SPEED      Wind speed 0-5 (default: 1.0)\n");
            printf("  -v, --viz=MODE        Visualization mode 0-6 (default: "
                   "1)\n");
            printf("  -c, --collision=MODE  Collision 0=off, 1=AABB, 2=mesh "
                   "(default: 1)\n");
            printf("  -d, --duration=SECS   Render duration (0=interactive, "
                   "default: 0)\n");
            printf("  -o, --output=PATH     Output directory for frames\n");
            printf("  -m, --model=PATH      Path to OBJ model file\n");
            printf(
                "  -a, --angle=DEGREES   Ahmed body slant angle (25 or 35)\n");
            printf(
                "  -s, --scale=SCALE     Model scale factor (default: 0.05)\n");
            printf("  -r, --reynolds=RE     Target Reynolds number (0=fixed "
                   "viscosity)\n");
            printf("  -g, --grid=XxYxZ      Grid size (default: 128x64x64)\n");
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

        // Scale so largest dimension fits in ~2 units
        g_modelScale = 2.0f / maxDim;
        printf("Auto scale: %.6f (max dim: %.2f)\n", g_modelScale, maxDim);

        // Auto-center: offset so model center is at origin
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
                float x = carModel.vertices[i].x * g_modelScale
                          + g_offsetX;
                float z = carModel.vertices[i].z * g_modelScale
                          + g_offsetZ;
                if (x < mnX) mnX = x;
                if (x > mxX) mxX = x;
                if (z < mnZ) mnZ = z;
                if (z > mxZ) mxZ = z;
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

    // Initialize LBM grid
    int lbmSizeX = gridX;
    int lbmSizeY = gridY;
    int lbmSizeZ = gridZ;

    // Map physical wind speed to a safe lattice velocity.
    // LBM is stable when the Mach number Ma = U/cs < 0.2
    // (cs = 1/sqrt(3) ~ 0.577). We cap the lattice velocity
    // at 0.08 and scale viscosity so higher wind speeds just
    // increase Re instead of blowing up the simulation.
    float maxLatticeVel = 0.08f;
    float latticeVelocity = windSpeed * 0.1f;
    if (latticeVelocity > maxLatticeVel)
        latticeVelocity = maxLatticeVel;

    // Viscosity scales with wind speed so Re increases
    // with wind: Re = U_phys * L / nu_phys. In lattice
    // units, we keep U_lat fixed and lower nu to raise Re.
    float baseViscosity = 0.02f;
    float lbmViscosity = baseViscosity;
    if (windSpeed > 0.8f) {
        lbmViscosity = baseViscosity * 0.8f / windSpeed;
        float minViscosity = 0.0167f; // tau_min=0.55 -> nu=(0.55-0.5)/3
        if (lbmViscosity < minViscosity)
            lbmViscosity = minViscosity;
    }

    if (reynoldsNumber > 0) {
        float scaleX = lbmSizeX / 8.0f;
        float charLength =
            (carBounds.maxX - carBounds.minX) * scaleX;
        lbmViscosity =
            (latticeVelocity * charLength) / reynoldsNumber;
        float tau = 3.0f * lbmViscosity + 0.5f;
        printf("Reynolds number: %.0f\n", reynoldsNumber);
        printf("  Char length: %.1f lattice units\n",
               charLength);
        printf("  Viscosity: %.6f\n", lbmViscosity);
        printf("  tau: %.4f\n", tau);
        if (tau <= 0.5f) {
            printf("  WARNING: tau <= 0.5, unstable. "
                   "Lower Re or wind speed.\n");
        }
    }

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
        // Enable Smagorinsky SGS turbulence model by default.
        // Adds local eddy viscosity so we can simulate at
        // effectively higher Re on coarse grids.
        lbmGrid->useSmagorinsky = 1;
        lbmGrid->smagorinskyCs = 0.15f;
        printf("Smagorinsky SGS enabled (Cs=%.2f)\n",
               lbmGrid->smagorinskyCs);

        // Auto-enable regularized when tau is low.
        if (lbmGrid->tau < 0.6f) {
            lbmGrid->useRegularized = 1;
            printf("Regularized collision enabled "
                   "(tau=%.3f)\n",
                   lbmGrid->tau);
        }

        LBM_InitializeFlow(lbmGrid, latticeVelocity, 0.0f, 0.0f);

        // Print effective Reynolds number
        {
            float sX = lbmGrid->sizeX / 8.0f;
            float charL =
                (carBounds.maxX - carBounds.minX) * sX;
            float latU = latticeVelocity;
            float nu = (lbmGrid->tau - 0.5f) / 3.0f;
            float re = (nu > 1e-10f) ? latU * charL / nu : 0;
            printf("Effective Re = %.0f "
                   "(U=%.3f, L=%.1f, nu=%.4f)\n",
                   re, latU, charL, nu);
        }
        printf("LBM initialized successfully\n");
    } else {
        printf("Warning: LBM initialization failed, using simple wind\n");
        useLBM = 0;
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
        particles[i].x = -4.0f + ((float)rand() / RAND_MAX) * 0.5f;
        particles[i].y = ((float)rand() / RAND_MAX - 0.5f) * 3.0f;
        particles[i].z = ((float)rand() / RAND_MAX - 0.5f) * 3.0f;
        particles[i].padding1 = 0.0f;
        particles[i].vx = 0.5f + ((float)rand() / RAND_MAX) * 0.2f;
        particles[i].vy = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
        particles[i].vz = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
        particles[i].life = 1.0f;
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

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void *)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(Particle),
                          (void *)(4 * sizeof(float)));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2,
                          1,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(Particle),
                          (void *)(7 * sizeof(float)));

    glBindVertexArray(0);
    checkGLError("After creating VAO");

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
    printf("Scroll wheel:   Zoom in/out\n");
    printf("A/D:            Rotate left/right\n");
    printf("W/S:            Rotate up/down\n");
    printf("Q/E:            Zoom in/out (step)\n");
    printf("R:              Reset camera\n");
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
    int maxFrames = (renderDuration > 0) ? renderDuration * 60 : 0;

    // Convergence detection for auto-stop
    #define CD_HISTORY_SIZE 10
    float cdHistory[CD_HISTORY_SIZE];
    int cdHistoryCount = 0;
    int converged = 0;

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
                }
            } else if (event.type == SDL_MOUSEBUTTONUP) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    mouseDown = 0;
                }
            } else if (event.type == SDL_MOUSEMOTION) {
                if (mouseDown) {
                    int dx = event.motion.x - lastMouseX;
                    int dy = event.motion.y - lastMouseY;

                    cameraAngleY += dx * 0.005f;
                    cameraAngleX += dy * 0.005f;

                    if (cameraAngleX > 1.5f)
                        cameraAngleX = 1.5f;
                    if (cameraAngleX < -1.5f)
                        cameraAngleX = -1.5f;

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
                    printf("Camera reset\n");
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
        }

        // Now set up particle compute shader
        glUseProgram(computeShaderProgram);

        // Bind buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particleBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, triangleBuffer);

        // Bind LBM velocity buffer if using LBM
        if (lbmGrid && useLBM) {
            glBindBufferBase(
                GL_SHADER_STORAGE_BUFFER, 4, LBM_GetVelocityBuffer(lbmGrid));
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
            glUniform3i(
                lbmGridSizeLoc, lbmGrid->sizeX, lbmGrid->sizeY, lbmGrid->sizeZ);
        }

        // Time uniform for randomness
        if (timeLoc != -1)
            glUniform1f(timeLoc, (float)SDL_GetTicks() / 1000.0f);

        // Dispatch particle compute (skip when paused unless single-stepping)
        if (!paused || stepOnce) {
            glDispatchCompute((GPU_PARTICLES + 255) / 256, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                            GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
        }

        // Render particles
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

        // Compute and display drag coefficient every 60 frames.
        // Skip the first 180 frames (~3s at 60fps) to avoid startup
        // oscillation.
        if (frameCount > 360 && frameCount % 60 == 0 && lbmGrid && useLBM) {
            float fx, fy, fz;
            LBM_ComputeDragForce(lbmGrid, &fx, &fy, &fz);

            // Reference area ~ car frontal area in lattice units
            float scaleY = lbmGrid->sizeY / 4.0f;
            float scaleZ = lbmGrid->sizeZ / 4.0f;
            float refArea = (carBounds.maxY - carBounds.minY) * scaleY *
                            (carBounds.maxZ - carBounds.minZ) * scaleZ;
            float Cd =
                LBM_ComputeDragCoefficient(lbmGrid, latticeVelocity, refArea);
            float Cl =
                LBM_ComputeLiftCoefficient(lbmGrid, latticeVelocity, refArea);

            printf("  Drag Force: (%.4f, %.4f, %.4f),"
                   " Cd=%.3f Cl=%.3f\n",
                   fx, fy, fz, Cd, Cl);

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
                    float relStd = sqrtf(var / CD_HISTORY_SIZE) / (mean + 1e-10f);

                    if (relStd < 0.02f) {
                        converged = 1;
                        printf("  Cd converged (mean=%.3f,"
                               " relStd=%.4f)\n",
                               mean, relStd);
                        // Auto-stop in headless mode
                        if (maxFrames > 0) {
                            // Run 2 more seconds for clean video ending
                            int extra = 120;
                            if (frameCount + extra < maxFrames)
                                maxFrames = frameCount + extra;
                        }
                    }
                }
            }
        }

        if (maxFrames > 0 && frameCount >= maxFrames) {
            printf("Render complete: %d frames\n", frameCount);
            running = 0;
        }

        if (renderDuration > 0 && strlen(outputPath) > 0) {
            char framePath[512];
            snprintf(framePath,
                     sizeof(framePath),
                     "%s/frame_%05d.ppm",
                     outputPath,
                     frameCount);
            saveFrameToPPM(framePath, WIDTH, HEIGHT);
        }

        stepOnce = 0;
        GLContext_SwapBuffers(glCtx);
    }

    printf("Cleaning up...\n");
    free(particles);
    if (triangleData)
        free(triangleData);
    freeModel(&carModel);
    if (fluidCube)
        FluidCubeFree(fluidCube);
    if (lbmGrid)
        LBM_Free(lbmGrid);
    glDeleteVertexArrays(1, &particleVAO);
    glDeleteBuffers(1, &particleBuffer);
    if (triangleBuffer)
        glDeleteBuffers(1, &triangleBuffer);
    glDeleteProgram(particleShaderProgram);
    glDeleteProgram(computeShaderProgram);
    GLContext_Destroy(glCtx);

    printf("Cleanup complete. Exiting.\n");
    return 0;
}
