#ifndef CLI_H
#define CLI_H

#define CLI_NUM_VIZ_MODES 10

enum { CLI_OP_BGK, CLI_OP_REGULARIZED, CLI_OP_MRT, CLI_OP_ENTROPIC };

// "bgk", "regularized", "mrt" or "entropic"
const char *cli_operator_name(int op);

typedef struct {
    float windSpeed;
    int visualizationMode;
    int collisionMode;
    int renderDuration;
    char outputPath[256];
    char modelPath[512];
    int slantAngle;
    float reynoldsNumber;
    int scaleFromCLI;
    float modelScale;
    int gridX, gridY, gridZ;
    float smagorinskyCs;
    int collisionOperator; // CLI_OP_*
    float turbulenceIntensity;
    int spongeCells; // -1 = auto (sizeX/8), 0 = off
    char vtkOutputPath[256];
    int vtkInterval;
    int vtkStats;
    int useSuperRes;
    char srWeightsPath[256];
    char srNormPath[256];
    char manifestPath[512]; // defaults to <output>/manifest.json
    float convergeTol;      // 0 = off
    int convergeWindow;
    int maxDuration;
    int strictReynolds; // fail instead of capping Re
} CliOptions;

// Parse command-line options. Returns 0 on success, 1 if --help was
// requested (caller should exit cleanly) and -1 on invalid input.
int cli_parse(int argc, char *argv[], CliOptions *opts);

// Project version and git describe, e.g. "1.5.0 (v1.5.0-3-gabc123)"
const char *cli_version(void);

typedef struct {
    const char *version;
    const char *gitDescribe;
    const char *gitCommit;
    int gitDirty;
} BuildInfo;

const BuildInfo *cli_build_info(void);

#endif // CLI_H
