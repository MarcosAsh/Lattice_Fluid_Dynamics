#ifndef CLI_H
#define CLI_H

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
    int gridX, gridY, gridZ;
    float smagorinskyCs;
    int useMRT;
    char vtkOutputPath[256];
    int vtkInterval;
    int useSuperRes;
    char srWeightsPath[256];
    char srNormPath[256];
} CliOptions;

// Parse command-line options. Returns 0 on success, 1 if --help was
// requested (caller should exit cleanly).
int cli_parse(int argc, char *argv[], CliOptions *opts);

#endif // CLI_H
