#include "../lib/cli.h"
#include "../lib/drag_metrics.h"
#include "lattice_version.h"

#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GRID_MAX 4096

enum {
    OPT_CONVERGE = 1000,
    OPT_CONVERGE_WINDOW,
    OPT_MAX_DURATION,
    OPT_MANIFEST,
    OPT_OPERATOR,
    OPT_STRICT_RE,
    OPT_TAU_FLOOR,
    OPT_CHECKPOINT,
    OPT_CHECKPOINT_INTERVAL,
    OPT_RESTART,
    OPT_SURFACE,
};

static int bad_value(const char *name, const char *arg) {
    fprintf(stderr, "invalid value for --%s: '%s'\n", name, arg);
    return 0;
}

static int parse_int(const char *name, const char *arg, int *out) {
    char *end;
    errno = 0;
    long v = strtol(arg, &end, 10);
    if (end == arg || *end != '\0' || errno == ERANGE || v < INT_MIN ||
        v > INT_MAX)
        return bad_value(name, arg);
    *out = (int)v;
    return 1;
}

static int parse_float(const char *name, const char *arg, float *out) {
    char *end;
    errno = 0;
    float v = strtof(arg, &end);
    if (end == arg || *end != '\0' || errno == ERANGE || !isfinite(v))
        return bad_value(name, arg);
    *out = v;
    return 1;
}

// Accepts XxYxZ or a single N meaning NxN/2xN/2
static int parse_grid(const char *arg, CliOptions *opts) {
    int dims[3];
    int n = 0;
    const char *p = arg;
    while (n < 3) {
        char *end;
        errno = 0;
        long v = strtol(p, &end, 10);
        if (end == p || errno == ERANGE || v < 1 || v > GRID_MAX)
            return bad_value("grid", arg);
        dims[n++] = (int)v;
        if (*end == '\0')
            break;
        if (*end != 'x' || n == 3)
            return bad_value("grid", arg);
        p = end + 1;
    }
    if (n == 1) {
        if (dims[0] < 2)
            return bad_value("grid", arg);
        dims[1] = dims[0] / 2;
        dims[2] = dims[0] / 2;
    } else if (n != 3) {
        return bad_value("grid", arg);
    }
    opts->gridX = dims[0];
    opts->gridY = dims[1];
    opts->gridZ = dims[2];
    return 1;
}

static int copy_path(const char *name, const char *arg, char *dst, size_t n) {
    if (strlen(arg) >= n) {
        fprintf(stderr, "--%s path too long (max %zu)\n", name, n - 1);
        return 0;
    }
    memcpy(dst, arg, strlen(arg) + 1);
    return 1;
}

const char *cli_version(void) {
    return LATTICE_VERSION " (" LATTICE_GIT_DESCRIBE ")";
}

const BuildInfo *cli_build_info(void) {
    static const BuildInfo info = {LATTICE_VERSION,
                                   LATTICE_GIT_DESCRIBE,
                                   LATTICE_GIT_COMMIT,
                                   LATTICE_GIT_DIRTY};
    return &info;
}

const char *cli_operator_name(int op) {
    static const char *names[] = {
        "bgk", "regularized", "mrt", "entropic", "recursive"};
    return (op >= CLI_OP_BGK && op <= CLI_OP_RECURSIVE) ? names[op] : "unknown";
}

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
    opts->modelScale = 1.0f;
    opts->gridX = 128;
    opts->gridY = 64;
    opts->gridZ = 64;
    opts->smagorinskyCs = 0.1f;
    opts->collisionOperator = CLI_OP_MRT;
    opts->turbulenceIntensity = 0.0f;
    opts->spongeCells = -1;
    opts->vtkOutputPath[0] = '\0';
    opts->vtkInterval = 100;
    opts->vtkStats = 0;
    opts->useSuperRes = 0;
    strncpy(opts->srWeightsPath,
            "assets/sr_model.bin",
            sizeof(opts->srWeightsPath) - 1);
    opts->srWeightsPath[sizeof(opts->srWeightsPath) - 1] = '\0';
    strncpy(opts->srNormPath,
            "assets/sr_model_norm.bin",
            sizeof(opts->srNormPath) - 1);
    opts->srNormPath[sizeof(opts->srNormPath) - 1] = '\0';
    opts->manifestPath[0] = '\0';
    opts->convergeTol = 0.0f;
    opts->convergeWindow = 20;
    opts->maxDuration = 0;
    opts->strictReynolds = 0;
    opts->tauFloor = 0.52f;
    opts->checkpointPath[0] = '\0';
    opts->checkpointInterval = 0;
    opts->restartPath[0] = '\0';
    opts->surfacePath[0] = '\0';

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
        {"entropic", no_argument, 0, 'E'},
        {"collision-operator", required_argument, 0, OPT_OPERATOR},
        {"turbulence", required_argument, 0, 't'},
        {"sponge", required_argument, 0, 'P'},
        {"vtk-output", required_argument, 0, 'V'},
        {"vtk-interval", required_argument, 0, 'I'},
        {"vtk-stats", no_argument, 0, 'T'},
        {"superres", no_argument, 0, 'R'},
        {"sr-weights", required_argument, 0, 'W'},
        {"manifest", required_argument, 0, OPT_MANIFEST},
        {"converge", required_argument, 0, OPT_CONVERGE},
        {"converge-window", required_argument, 0, OPT_CONVERGE_WINDOW},
        {"max-duration", required_argument, 0, OPT_MAX_DURATION},
        {"strict-reynolds", no_argument, 0, OPT_STRICT_RE},
        {"tau-floor", required_argument, 0, OPT_TAU_FLOOR},
        {"checkpoint", required_argument, 0, OPT_CHECKPOINT},
        {"checkpoint-interval", required_argument, 0, OPT_CHECKPOINT_INTERVAL},
        {"restart", required_argument, 0, OPT_RESTART},
        {"surface-output", required_argument, 0, OPT_SURFACE},
        {"version", no_argument, 0, 'X'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}

    };

    int opt;
    while (
        (opt = getopt_long(
             argc, argv, "w:v:c:d:o:m:a:r:s:g:S:Mt:Eh", long_options, NULL)) !=
        -1) {
        switch (opt) {
        case 'w':
            if (!parse_float("wind", optarg, &opts->windSpeed))
                return -1;
            if (opts->windSpeed < 0)
                opts->windSpeed = 0;
            if (opts->windSpeed > 5)
                opts->windSpeed = 5;
            break;
        case 'v':
            if (!parse_int("viz", optarg, &opts->visualizationMode))
                return -1;
            if (opts->visualizationMode < 0)
                opts->visualizationMode = 0;
            if (opts->visualizationMode >= CLI_NUM_VIZ_MODES)
                opts->visualizationMode = CLI_NUM_VIZ_MODES - 1;
            break;
        case 'c':
            if (!parse_int("collision", optarg, &opts->collisionMode))
                return -1;
            if (opts->collisionMode < 0)
                opts->collisionMode = 0;
            if (opts->collisionMode > 3)
                opts->collisionMode = 3;
            break;
        case 'd':
            if (!parse_int("duration", optarg, &opts->renderDuration))
                return -1;
            if (opts->renderDuration < 0)
                opts->renderDuration = 0;
            break;
        case 'o':
            if (!copy_path("output",
                           optarg,
                           opts->outputPath,
                           sizeof(opts->outputPath)))
                return -1;
            break;
        case 'm':
            if (!copy_path(
                    "model", optarg, opts->modelPath, sizeof(opts->modelPath)))
                return -1;
            break;
        case 'a':
            if (!parse_int("angle", optarg, &opts->slantAngle))
                return -1;
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
            if (!parse_float("scale", optarg, &opts->modelScale))
                return -1;
            if (opts->modelScale <= 0.0f) {
                bad_value("scale", optarg);
                return -1;
            }
            opts->scaleFromCLI = 1;
            break;
        case 'r':
            if (!parse_float("reynolds", optarg, &opts->reynoldsNumber))
                return -1;
            if (opts->reynoldsNumber < 0)
                opts->reynoldsNumber = 0;
            break;
        case 'S':
            if (!parse_float("smagorinsky", optarg, &opts->smagorinskyCs))
                return -1;
            if (opts->smagorinskyCs < 0.0f)
                opts->smagorinskyCs = 0.0f;
            if (opts->smagorinskyCs > 0.5f)
                opts->smagorinskyCs = 0.5f;
            break;
        case 'g':
            if (!parse_grid(optarg, opts))
                return -1;
            break;
        case 'M':
            opts->collisionOperator = CLI_OP_MRT;
            break;
        case 'E':
            opts->collisionOperator = CLI_OP_ENTROPIC;
            break;
        case OPT_STRICT_RE:
            opts->strictReynolds = 1;
            break;
        case OPT_TAU_FLOOR:
            if (!parse_float("tau-floor", optarg, &opts->tauFloor))
                return -1;
            if (opts->tauFloor <= 0.5f || opts->tauFloor > 1.0f) {
                bad_value("tau-floor", optarg);
                return -1;
            }
            break;
        case OPT_OPERATOR: {
            int found = -1;
            for (int op = CLI_OP_BGK; op <= CLI_OP_RECURSIVE; op++)
                if (strcmp(optarg, cli_operator_name(op)) == 0)
                    found = op;
            if (strcmp(optarg, "reg") == 0)
                found = CLI_OP_REGULARIZED;
            if (strcmp(optarg, "rr") == 0)
                found = CLI_OP_RECURSIVE;
            if (found < 0) {
                fprintf(stderr,
                        "invalid value for --collision-operator: '%s' "
                        "(bgk, regularized, mrt, entropic, recursive)\n",
                        optarg);
                return -1;
            }
            opts->collisionOperator = found;
            break;
        }
        case 't':
            if (!parse_float("turbulence", optarg, &opts->turbulenceIntensity))
                return -1;
            if (opts->turbulenceIntensity < 0.0f)
                opts->turbulenceIntensity = 0.0f;
            if (opts->turbulenceIntensity > 0.3f)
                opts->turbulenceIntensity = 0.3f;
            break;
        case 'P':
            if (!parse_int("sponge", optarg, &opts->spongeCells))
                return -1;
            if (opts->spongeCells < 0)
                opts->spongeCells = -1;
            break;
        case 'V':
            if (!copy_path("vtk-output",
                           optarg,
                           opts->vtkOutputPath,
                           sizeof(opts->vtkOutputPath)))
                return -1;
            break;
        case 'I':
            if (!parse_int("vtk-interval", optarg, &opts->vtkInterval))
                return -1;
            if (opts->vtkInterval < 1)
                opts->vtkInterval = 1;
            break;
        case 'T':
            opts->vtkStats = 1;
            break;
        case 'R':
            opts->useSuperRes = 1;
            break;
        case 'W':
            if (!copy_path("sr-weights",
                           optarg,
                           opts->srWeightsPath,
                           sizeof(opts->srWeightsPath)))
                return -1;
            break;
        case OPT_MANIFEST:
            if (!copy_path("manifest",
                           optarg,
                           opts->manifestPath,
                           sizeof(opts->manifestPath)))
                return -1;
            break;
        case OPT_CONVERGE:
            if (!parse_float("converge", optarg, &opts->convergeTol))
                return -1;
            if (!(opts->convergeTol > 0.0f && opts->convergeTol < 1.0f)) {
                bad_value("converge", optarg);
                return -1;
            }
            break;
        case OPT_CONVERGE_WINDOW:
            if (!parse_int("converge-window", optarg, &opts->convergeWindow))
                return -1;
            if (opts->convergeWindow < 4 || opts->convergeWindow > 100000) {
                bad_value("converge-window", optarg);
                return -1;
            }
            break;
        case OPT_MAX_DURATION:
            if (!parse_int("max-duration", optarg, &opts->maxDuration))
                return -1;
            if (opts->maxDuration < 1) {
                bad_value("max-duration", optarg);
                return -1;
            }
            break;
        case OPT_CHECKPOINT:
            if (!copy_path("checkpoint",
                           optarg,
                           opts->checkpointPath,
                           sizeof(opts->checkpointPath)))
                return -1;
            break;
        case OPT_CHECKPOINT_INTERVAL:
            if (!parse_int(
                    "checkpoint-interval", optarg, &opts->checkpointInterval))
                return -1;
            if (opts->checkpointInterval < 1) {
                bad_value("checkpoint-interval", optarg);
                return -1;
            }
            break;
        case OPT_RESTART:
            if (!copy_path("restart",
                           optarg,
                           opts->restartPath,
                           sizeof(opts->restartPath)))
                return -1;
            break;
        case OPT_SURFACE:
            if (!copy_path("surface-output",
                           optarg,
                           opts->surfacePath,
                           sizeof(opts->surfacePath)))
                return -1;
            break;
        case 'X':
            printf("3d_fluid_simulation_car %s\n", cli_version());
            return 1;
        case '?':
            fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
            return -1;
        case 'h':
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -w, --wind=SPEED      Wind speed 0-5 (default: 1.0)\n");
            printf("  -v, --viz=MODE        Visualization mode 0-9 (default: "
                   "1)\n");
            printf("  -c, --collision=MODE  Collision 0=off, 1=AABB, 2=mesh, "
                   "3=voxel "
                   "(default: 2)\n");
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
            printf("  --collision-operator=OP\n"
                   "                        bgk, regularized, mrt, entropic or "
                   "recursive\n"
                   "                        (default: mrt)\n");
            printf(
                "  -M, --mrt             Same as --collision-operator=mrt\n");
            printf(
                "  -E, --entropic        Same as --collision-operator=entropic,"
                "\n"
                "                        also turns Smagorinsky off\n");
            printf("  -t, --turbulence=TI   Inlet turbulence intensity "
                   "0-0.3 (default: 0,\n"
                   "                        wind tunnels are ~0.01-0.02)\n");
            printf("  --sponge=CELLS        Outlet acoustic sponge width "
                   "(default: auto =\n"
                   "                        gridX/8, 0 disables)\n");
            printf("  --vtk-output=PATH     Directory for VTK field dumps\n");
            printf("  --vtk-interval=N      Frames between VTK dumps "
                   "(default: 100)\n");
            printf("  --vtk-stats           Add time-averaged velocity, "
                   "RMS and TKE to VTK output\n");
            printf("  --superres            Enable 2x super-resolution "
                   "upscaling\n");
            printf("  --sr-weights=PATH     Super-resolution weights "
                   "(default: assets/sr_model.bin)\n");
            printf("  --manifest=PATH       Run manifest JSON (default: "
                   "OUTPUT/manifest.json\n"
                   "                        when --output is set)\n");
            printf("  --converge=TOL        Stop a headless run once the "
                   "95%% confidence\n"
                   "                        half-width of mean Cd is under "
                   "TOL of the mean\n"
                   "                        and the series after the "
                   "transient is stationary\n");
            printf("  --converge-window=N   Cd samples between convergence "
                   "checks (default: 20,\n"
                   "                        one sample every %d frames)\n",
                   CD_SAMPLE_INTERVAL);
            printf("  --max-duration=SECS   Frame cap for --converge "
                   "(overrides --duration)\n");
            printf("  --strict-reynolds     Exit with an error if the grid "
                   "cannot reach the\n"
                   "                        requested Re at tau >= 0.52\n");
            printf("  --tau-floor=TAU       Developer: lowest tau before Re "
                   "is capped (default:\n"
                   "                        0.52, must be in (0.5, 1])\n");
            printf("  --checkpoint=PATH     Write the solver state to PATH at "
                   "the end of the run\n");
            printf("  --checkpoint-interval=FRAMES\n"
                   "                        Also write it every FRAMES "
                   "frames\n");
            printf("  --restart=PATH        Continue the run saved in PATH\n");
            printf("  --surface-output=PATH Time-averaged Cp and wall force on "
                   "the body, VTK\n"
                   "                        and CSV\n");
            printf("  --version             Print version and exit\n");
            printf("  -h, --help            Show this help\n");
            return 1;
        default:
            return -1;
        }
    }

    if (!opts->manifestPath[0] && opts->outputPath[0]) {
        int n = snprintf(opts->manifestPath,
                         sizeof(opts->manifestPath),
                         "%s/manifest.json",
                         opts->outputPath);
        if (n < 0 || (size_t)n >= sizeof(opts->manifestPath)) {
            fprintf(stderr, "--output path too long for the manifest\n");
            return -1;
        }
    }

    if (opts->checkpointInterval > 0 && !opts->checkpointPath[0]) {
        fprintf(stderr, "--checkpoint-interval needs --checkpoint\n");
        return -1;
    }

    if (opts->maxDuration > 0) {
        if (opts->convergeTol <= 0.0f) {
            fprintf(stderr, "--max-duration needs --converge\n");
            return -1;
        }
        opts->renderDuration = opts->maxDuration;
    }
    if (opts->convergeTol > 0.0f && opts->renderDuration <= 0) {
        fprintf(stderr,
                "--converge needs a headless run (--duration or "
                "--max-duration)\n");
        return -1;
    }

    return 0;
}
