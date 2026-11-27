#ifndef MANIFEST_H
#define MANIFEST_H

#include "cli.h"

#define MANIFEST_SCHEMA_VERSION 1

// Everything a run records in its manifest. NAN floats become null.
typedef struct {
    const CliOptions *opts;
    char startTimeUtc[32];
    double wallSeconds;
    const char *glVendor;
    const char *glRenderer;
    const char *glVersion;

    // Resolved settings
    float modelScale;
    int spongeCells;
    const char *collisionOperator;
    int smagorinsky;
    int periodicYZ;
    float viscosity;
    float tau;
    float latticeVelocity;
    float reynolds;
    float reynoldsEffective;
    float charLength;

    // Model
    const char *modelPath;
    int triangles;
    long solidCells;
    float refArea;
    float blockageFactor;

    // Run
    int frames;
    int outputFrames;
    long lbmSteps;
    int substeps;
    int sampleIntervalFrames;
    float emaAlpha;

    // Results
    int cdSamples;
    float cd;
    float cdCorr;
    float cdEma;
    float cdMean;
    float cdCorrMean;
    float cdPressure;
    float cdFriction;
    float cl;
    float clMean;
    float tStar;
    float flowThroughs;
    float strouhal;
    float sheddingFreq;

    // Convergence stopping
    int converged;
    int stopFrame;
    double convMean;
    double convSem;
    double convRelChange;
} RunManifest;

// Fills build info and defaults, NAN for results not yet known
void manifest_init(RunManifest *m, const CliOptions *opts);

// Writes path atomically via a temp file. Returns 0 on success.
int manifest_write(const char *path, const RunManifest *m);

#endif // MANIFEST_H
