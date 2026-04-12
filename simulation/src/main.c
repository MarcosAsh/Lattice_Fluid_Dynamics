#define _USE_MATH_DEFINES
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <glad/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include "../lib/gl_context.h"
#include "../lib/lbm.h"
#include "../lib/particle_system.h"
#include "../obj-file-loader/lib/model_loader.h"
#include "../lib/render_model.h"
#include "../lib/opengl_utils.h"
#include "../lib/config.h"
#include "../lib/ml_predict.h"
#include "../lib/superres.h"

#include "../lib/cli.h"
#include "../lib/drag_metrics.h"
#include "../lib/event_handlers.h"
#include "../lib/gl_helpers.h"
#include "../lib/model_bounds.h"
#include "../lib/view_matrix.h"
#include "../lib/vti_export.h"

#define GPU_PARTICLES MAX_PARTICLES

// Global model transform (definitions)
float g_modelScale = 1.0f;
float g_offsetX = 0.0f;
float g_offsetY = -0.1f;
float g_offsetZ = -0.9f;
float g_carRotationY = 360.0f;

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

int main(int argc, char *argv[]) {
    printf("Starting 3D Fluid Simulation...\n");
    srand(time(NULL));

    CliOptions opts;
    if (cli_parse(argc, argv, &opts) != 0)
        return 0;

    float windSpeed = opts.windSpeed;
    int visualizationMode = opts.visualizationMode;
    int collisionMode = opts.collisionMode;
    int renderDuration = opts.renderDuration;
    char outputPath[256];
    strncpy(outputPath, opts.outputPath, sizeof(outputPath));
    outputPath[sizeof(outputPath) - 1] = '\0';
    char modelPath[512];
    strncpy(modelPath, opts.modelPath, sizeof(modelPath));
    modelPath[sizeof(modelPath) - 1] = '\0';
    int slantAngle = opts.slantAngle;
    float reynoldsNumber = opts.reynoldsNumber;
    int scaleFromCLI = opts.scaleFromCLI;
    int gridX = opts.gridX, gridY = opts.gridY, gridZ = opts.gridZ;
    float smagorinskyCs = opts.smagorinskyCs;
    int useMRT = opts.useMRT;
    char vtkOutputPath[256];
    strncpy(vtkOutputPath, opts.vtkOutputPath, sizeof(vtkOutputPath));
    vtkOutputPath[sizeof(vtkOutputPath) - 1] = '\0';
    int vtkInterval = opts.vtkInterval;
    int useSuperRes = opts.useSuperRes;
    char srWeightsPath[256];
    strncpy(srWeightsPath, opts.srWeightsPath, sizeof(srWeightsPath));
    srWeightsPath[sizeof(srWeightsPath) - 1] = '\0';
    char srNormPath[256];
    strncpy(srNormPath, opts.srNormPath, sizeof(srNormPath));
    srNormPath[sizeof(srNormPath) - 1] = '\0';

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

    if (renderDuration == 0)
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
        printf("Trying fallback path ../assets/...\n");
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
    // not given. Scale: 200 * windSpeed, so wind=1 -> Re=200.
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
    // tau < 0.52 is numerically unstable. At 0.52 < tau < 0.65,
    // the force has period-2 oscillation but the EMA averages it
    // out after ~50 samples. MRT + Smagorinsky help stability.
    if (tau < 0.52f) {
        lbmViscosity = (0.52f - 0.5f) / 3.0f;
        tau = 0.52f;
        float actualRe = (latticeVelocity * charLength) / lbmViscosity;
        printf("  Re capped at %.0f (tau=%.2f needed for stable Cd, "
               "requested Re=%.0f)\n",
               actualRe, tau, reynoldsNumber);
        reynoldsNumber = actualRe;
    }
    printf("Reynolds number: %.0f\n", reynoldsNumber);
    printf("  Char length: %.1f lattice units\n", charLength);
    printf("  Viscosity: %.6f\n", lbmViscosity);
    printf("  tau: %.4f\n", tau);
    printf("  CFL: %.4f\n", latticeVelocity);

    float refArea = 0.0f, bboxArea = 0.0f;
    float epsilon = 0.0f, blockageFactor = 1.0f;

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

        // Ground plane is available via LBM_AddGroundPlane() but
        // disabled by default: it conflicts with periodic z-BC and
        // needs sufficient domain height (> 3x body height) plus
        // ground clearance to avoid dominating the drag signal.

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

        // MRT is on by default -- much more stable than BGK at
        // low tau, which is typical for external aero grids.
        lbmGrid->useMRT = 1;
        printf("MRT collision enabled\n");

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

        // Compute reference area and blockage BEFORE the main loop
        // (glFinish in the readback disrupts force computation if
        // called mid-loop).
        {
            float sY = lbmGrid->sizeY / 4.0f;
            float sZ = lbmGrid->sizeZ / 4.0f;
            bboxArea = (carBounds.maxY - carBounds.minY) * sY *
                       (carBounds.maxZ - carBounds.minZ) * sZ;
            refArea = LBM_ComputeProjectedArea(lbmGrid, 0);
            if (refArea < 1.0f)
                refArea = bboxArea;
            epsilon = refArea / ((float)lbmGrid->sizeY * lbmGrid->sizeZ);
            blockageFactor = (1.0f - epsilon) * (1.0f - epsilon);
        }

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

    // Super-resolution upscaler (optional)
    SuperResUpscaler *srUpscaler = NULL;
    if (useSuperRes) {
        srUpscaler =
            SR_Create(lbmSizeX, lbmSizeY, lbmSizeZ, srWeightsPath, srNormPath);
        if (srUpscaler) {
            int fineX, fineY, fineZ;
            SR_GetFineSize(srUpscaler, &fineX, &fineY, &fineZ);
            printf("Super-resolution: %dx%dx%d -> %dx%dx%d\n",
                   lbmSizeX,
                   lbmSizeY,
                   lbmSizeZ,
                   fineX,
                   fineY,
                   fineZ);
        } else {
            printf("Super-resolution: failed to initialize, "
                   "falling back to native resolution\n");
            useSuperRes = 0;
        }
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
#define STREAMLINE_LEN   64
    GLuint streamlineBuffer;
    {
        size_t slSize = STREAMLINE_SEEDS * STREAMLINE_LEN * 4 * sizeof(float);
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

    // Convergence detection for auto-stop. CD_HISTORY_SIZE and
    // CD_SAMPLE_INTERVAL live in drag_metrics.h.
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
            handle_sdl_event(&event,
                             &running,
                             &collisionMode,
                             &visualizationMode,
                             &windSpeed,
                             numTriangles,
                             frameCount);
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

            // Super-resolution upscale after LBM step
            if (srUpscaler) {
                SR_Upscale(srUpscaler, LBM_GetVelocityBuffer(lbmGrid));
            }

            // VTK field dump at specified interval
            if (strlen(vtkOutputPath) > 0 && frameCount % vtkInterval == 0) {
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

            // Bind LBM velocity and solid buffers if using LBM.
            // When super-resolution is active, particles sample the
            // upscaled fine velocity buffer instead.
            if (lbmGrid && useLBM) {
                GLuint velBuf = srUpscaler
                                    ? SR_GetFineVelocityBuffer(srUpscaler)
                                    : LBM_GetVelocityBuffer(lbmGrid);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, velBuf);
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
                if (srUpscaler) {
                    int fx, fy, fz;
                    SR_GetFineSize(srUpscaler, &fx, &fy, &fz);
                    glUniform3i(lbmGridSizeLoc, fx, fy, fz);
                } else {
                    glUniform3i(lbmGridSizeLoc,
                                lbmGrid->sizeX,
                                lbmGrid->sizeY,
                                lbmGrid->sizeZ);
                }
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
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,
                                     4,
                                     LBM_GetVelocityBuffer(lbmGrid));
                    glBindBufferBase(
                        GL_SHADER_STORAGE_BUFFER, 11, streamlineBuffer);
                    glUseProgram(streamlineTraceShader);
                    glUniform3i(glGetUniformLocation(streamlineTraceShader,
                                                     "lbmGridSize"),
                                lbmGrid->sizeX,
                                lbmGrid->sizeY,
                                lbmGrid->sizeZ);
                    glUniform1i(
                        glGetUniformLocation(streamlineTraceShader, "numSeeds"),
                        STREAMLINE_SEEDS);
                    glUniform1i(
                        glGetUniformLocation(streamlineTraceShader, "traceLen"),
                        STREAMLINE_LEN);
                    glDispatchCompute((STREAMLINE_SEEDS + 255) / 256, 1, 1);
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

                renderModel(&carModel);

                glPopMatrix();
                glMatrixMode(GL_PROJECTION);
                glPopMatrix();
                glMatrixMode(GL_MODELVIEW);

                checkGLError("After rendering model");
            }
        } // end skipRendering

        // Compute and display drag coefficient every 20 frames
        // once the flow has developed (cdStartFrame computed above).
        if (frameCount >= cdStartFrame &&
            frameCount % CD_SAMPLE_INTERVAL == 0 && lbmGrid && useLBM) {
            // Compute force with pressure/friction decomposition
            float fx, fy, fz, px, py, pz;
            LBM_ComputeDragForceDecomposed(
                lbmGrid, &fx, &fy, &fz, &px, &py, &pz);

            // refArea, bboxArea, epsilon, blockageFactor computed at
            // init (see LBM initialization block above)

            float rho_0 = 1.0f;
            float dynP = 0.5f * rho_0 * latticeVelocity * latticeVelocity;
            float denom = dynP * refArea;
            float Cd = (denom > 1e-10f) ? fabsf(fx) / denom : 0.0f;
            float Cl = (denom > 1e-10f) ? fy / denom : 0.0f;
            float CdPressure = (denom > 1e-10f) ? fabsf(px) / denom : 0.0f;
            float CdFriction = Cd - CdPressure;
            if (CdFriction < 0.0f)
                CdFriction = 0.0f;

            // Blockage correction (Pope & Harper 1966)
            float CdCorr = Cd * blockageFactor;

            // One-time diagnostic
            static int cdDiagPrinted = 0;
            if (!cdDiagPrinted) {
                float scaleY = lbmGrid->sizeY / 4.0f;
                float scaleZ = lbmGrid->sizeZ / 4.0f;
                float bodyLatY = (carBounds.maxY - carBounds.minY) * scaleY;
                float bodyLatZ = (carBounds.maxZ - carBounds.minZ) * scaleZ;
                printf("  Cd calculation breakdown:\n");
                printf("    Re = %.0f, U_lattice = %.4f, rho_0 = %.1f\n",
                       reynoldsNumber,
                       latticeVelocity,
                       rho_0);
                printf("    Body (lattice): %.1f x %.1f x %.1f cells\n",
                       charLength,
                       bodyLatY,
                       bodyLatZ);
                printf("    Projected area = %.1f cells^2 "
                       "(bbox = %.1f)\n",
                       refArea,
                       bboxArea);
                printf("    Blockage = %.1f%%, correction = %.3f "
                       "(Pope & Harper 1966)\n",
                       epsilon * 100.0f,
                       blockageFactor);
                if (fminf(bodyLatY, bodyLatZ) < 10.0f)
                    printf("    WARNING: body < 10 cells across -- Cd is "
                           "unreliable at this resolution\n");
                if (epsilon > 0.05f)
                    printf("    WARNING: blockage > 5%% -- confinement "
                           "inflates Cd\n");
                printf("    NOTE: published Ahmed body Cd ~0.3 is at "
                       "Re > 500k. Low-Re simulations give higher Cd.\n");
                cdDiagPrinted = 1;
            }

            // Exponential moving average tracks corrected Cd
            if (cdHistoryCount == 0) {
                cdEma = CdCorr;
            } else {
                float alpha = 0.1f;
                cdEma = alpha * CdCorr + (1.0f - alpha) * cdEma;
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
                   " Cd=%.3f Cd_corr=%.3f Cl=%.3f (avg=%.3f)"
                   " t*=%.3f flow-throughs=%.2f\n",
                   fx,
                   fy,
                   fz,
                   Cd,
                   CdCorr,
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
                    float *tmp =
                        (float *)realloc(clSeries, clCapacity * sizeof(float));
                    if (tmp) {
                        clSeries = tmp;
                    } else {
                        clCapacity /= 2;
                    }
                }
                if (clCount < clCapacity)
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

    if (useLBM) {
        compute_strouhal(clSeries,
                         clCount,
                         lbmSubsteps,
                         charLength,
                         latticeVelocity);
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
    if (lbmGrid)
        LBM_Free(lbmGrid);
    if (mlModel)
        ML_Free(mlModel);
    if (srUpscaler)
        SR_Free(srUpscaler);
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
