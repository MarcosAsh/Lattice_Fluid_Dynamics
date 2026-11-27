#ifndef CONVERGENCE_H
#define CONVERGENCE_H

#include <stdio.h>

// Steady-state statistics of the Cd series and a stopping rule on them
#define CONV_METHOD         "mser5+lugsail_obm+hw_cvm+trend"
#define CONV_METHOD_VERSION 1

#define CONV_MIN_SAMPLES 30
#define CONV_MIN_KEPT    200
#define CONV_MIN_ESS     50
#define CONV_CVM_95      0.461

typedef struct {
    int valid;          // the fields below are defined
    const char *reason; // why not valid or not stationary, or a warning
    int samples;        // finite samples seen
    int rejected;       // NaN or Inf samples dropped
    int truncated;      // leading samples discarded as transient
    int kept;
    int truncationOk; // MSER-5 minimum lies in the first half
    int stationary;   // Cramer-von Mises test passes
    double cvm;       // Cramer-von Mises statistic of the bridge
    double mean;
    double sd;
    double sem; // from lugsail overlapping batch means
    int batchSize;
    double dof;
    double ess;
    double halfWidth; // 95 percent confidence half-width
    double relHalfWidth;
    double lag1;
    double drift; // rise of the least-squares line over the kept part
    double relDrift;
} ConvStats;

typedef struct {
    float tol; // max relative half-width, 0 = statistics only
    int checkEvery;
    float *samples;
    int count;
    int capacity;
    int rejected;
    int allocFailed;
    int converged;
} ConvergenceDetector;

// Returns 0 on success, -1 on bad arguments or allocation failure
int conv_init(ConvergenceDetector *d, float tol, int checkEvery);
void conv_free(ConvergenceDetector *d);

// Returns 1 on the sample that reaches convergence
int conv_add(ConvergenceDetector *d, float value);

void conv_stats(const ConvergenceDetector *d, ConvStats *s);
void conv_series_stats(const float *x, int n, ConvStats *s);

// NULL when s meets the stopping rule for tol, else the reason
const char *conv_stop_reason(const ConvStats *s, float tol);

// Stdout summary, free of the Cd= the cloud worker parses
void conv_print(FILE *f, const ConvStats *s);

// Two-sided 95 percent Student t quantile
double conv_t975(double dof);

#endif // CONVERGENCE_H
