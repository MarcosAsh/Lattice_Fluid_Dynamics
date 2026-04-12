#include "../lib/cli.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Globals defined in main.c that the CLI needs to poke directly.
extern float g_modelScale;
extern const int numVizModes;

int cli_parse(int argc, char *argv[], CliOptions *opts) {
    opts->windSpeed = 1.0f;
    opts->visualizationMode = 1;
    opts->collisionMode = 2;
    opts->renderDuration = 0;
    opts->outputPath[0] = '\0';
    strncpy(opts->modelPath,
            "assets/3d-files/car-model.obj",
            sizeof(opts->modelPath) - 1);
    opts->modelPath[sizeof(opts->modelPath) - 1] = '\0';
    opts->slantAngle = 0;
    opts->reynoldsNumber = 0.0f;
    opts->scaleFromCLI = 0;
    opts->gridX = 128;
    opts->gridY = 64;
    opts->gridZ = 64;
    opts->smagorinskyCs = 0.1f;
    opts->useMRT = 0;
    opts->vtkOutputPath[0] = '\0';
    opts->vtkInterval = 100;
    opts->useSuperRes = 0;
    strncpy(opts->srWeightsPath,
            "assets/sr_model.bin",
            sizeof(opts->srWeightsPath) - 1);
    opts->srWeightsPath[sizeof(opts->srWeightsPath) - 1] = '\0';
    strncpy(opts->srNormPath,
            "assets/sr_model_norm.bin",
            sizeof(opts->srNormPath) - 1);
    opts->srNormPath[sizeof(opts->srNormPath) - 1] = '\0';

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
        {"superres", no_argument, 0, 'R'},
        {"sr-weights", required_argument, 0, 'W'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}

    };

    int opt;
    while ((opt = getopt_long(
                argc, argv, "w:v:c:d:o:m:a:r:s:g:S:Mh", long_options, NULL)) !=
           -1) {
        switch (opt) {
        case 'w':
            opts->windSpeed = atof(optarg);
            if (opts->windSpeed < 0)
                opts->windSpeed = 0;
            if (opts->windSpeed > 5)
                opts->windSpeed = 5;
            break;
        case 'v':
            opts->visualizationMode = atoi(optarg);
            if (opts->visualizationMode < 0)
                opts->visualizationMode = 0;
            if (opts->visualizationMode >= numVizModes)
                opts->visualizationMode = numVizModes - 1;
            break;
        case 'c':
            opts->collisionMode = atoi(optarg);
            if (opts->collisionMode < 0)
                opts->collisionMode = 0;
            if (opts->collisionMode > 3)
                opts->collisionMode = 3;
            break;
        case 'd':
            opts->renderDuration = atoi(optarg);
            if (opts->renderDuration < 0)
                opts->renderDuration = 0;
            break;
        case 'o':
            strncpy(opts->outputPath, optarg, sizeof(opts->outputPath) - 1);
            opts->outputPath[sizeof(opts->outputPath) - 1] = '\0';
            break;
        case 'm':
            strncpy(opts->modelPath, optarg, sizeof(opts->modelPath) - 1);
            opts->modelPath[sizeof(opts->modelPath) - 1] = '\0';
            break;
        case 'a':
            opts->slantAngle = atoi(optarg);
            if (opts->slantAngle == 25) {
                strncpy(opts->modelPath,
                        "assets/3d-files/ahmed_25deg_m.obj",
                        sizeof(opts->modelPath) - 1);
            } else if (opts->slantAngle == 35) {
                strncpy(opts->modelPath,
                        "assets/3d-files/ahmed_35deg_m.obj",
                        sizeof(opts->modelPath) - 1);
            }
            break;
        case 's':
            g_modelScale = atof(optarg);
            opts->scaleFromCLI = 1;
            break;
        case 'r':
            opts->reynoldsNumber = atof(optarg);
            if (opts->reynoldsNumber < 0)
                opts->reynoldsNumber = 0;
            break;
        case 'S':
            opts->smagorinskyCs = atof(optarg);
            if (opts->smagorinskyCs < 0.0f)
                opts->smagorinskyCs = 0.0f;
            if (opts->smagorinskyCs > 0.5f)
                opts->smagorinskyCs = 0.5f;
            break;
        case 'g':
            if (sscanf(optarg, "%dx%dx%d", &opts->gridX, &opts->gridY, &opts->gridZ) != 3) {
                /* Try single number: NxN/2xN/2 */
                int n = atoi(optarg);
                if (n > 0) {
                    opts->gridX = n;
                    opts->gridY = n / 2;
                    opts->gridZ = n / 2;
                }
            }
            break;
        case 'M':
            opts->useMRT = 1;
            break;
        case 'V':
            strncpy(opts->vtkOutputPath, optarg, sizeof(opts->vtkOutputPath) - 1);
            opts->vtkOutputPath[sizeof(opts->vtkOutputPath) - 1] = '\0';
            break;
        case 'I':
            opts->vtkInterval = atoi(optarg);
            if (opts->vtkInterval < 1)
                opts->vtkInterval = 1;
            break;
        case 'R':
            opts->useSuperRes = 1;
            break;
        case 'W':
            strncpy(opts->srWeightsPath, optarg, sizeof(opts->srWeightsPath) - 1);
            opts->srWeightsPath[sizeof(opts->srWeightsPath) - 1] = '\0';
            break;
        case 'h':
        default:
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -w, --wind=SPEED      Wind speed 0-5 (default: 1.0)\n");
            printf("  -v, --viz=MODE        Visualization mode 0-9 (default: "
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
            printf("  --superres            Enable 2x super-resolution "
                   "upscaling\n");
            printf("  --sr-weights=PATH     Super-resolution weights "
                   "(default: assets/sr_model.bin)\n");
            printf("  -h, --help            Show this help\n");
            return 1;
        }
    }

    return 0;
}
