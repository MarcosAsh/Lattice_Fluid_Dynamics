#ifndef CLI_H
#define CLI_H

#define CLI_NUM_VIZ_MODES 10

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
    int useMRT;
    int useEntropic;
    float turbulenceIntensity;
    int spongeCells; // -1 = auto (sizeX/8), 0 = off
    char vtkOutputPath[256];
    int vtkInterval;
    int vtkStats;
    int useSuperRes;
    char srWeightsPath[256];
    char srNormPath[256];
    float convergeTol; // 0 = off
    int convergeWindow;
    int maxDuration;
} CliOptions;

// Parse command-line options. Returns 0 on success, 1 if --help was
// requested (caller should exit cleanly) and -1 on invalid input.
int cli_parse(int argc, char *argv[], CliOptions *opts);

// Project version and git describe, e.g. "1.5.0 (v1.5.0-3-gabc123)"
const char *cli_version(void);

#endif // CLI_H
