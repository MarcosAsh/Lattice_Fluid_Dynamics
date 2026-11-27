#ifndef CONVERGENCE_H
#define CONVERGENCE_H

// Stops a run once the windowed mean of a series settles: the means of
// the last two consecutive windows differ by less than tol (relative)
// at two window boundaries in a row.
typedef struct {
    float tol;
    int window;
    float *samples;
    int count;
    int capacity;
    int stableChecks;
    int converged;
    double relChange;
} ConvergenceDetector;

#define CONV_MIN_WINDOW 4

// Returns 0 on success, -1 on bad arguments or allocation failure
int conv_init(ConvergenceDetector *d, float tol, int window);
void conv_free(ConvergenceDetector *d);

// Adds a sample. Returns 1 on the sample where convergence is reached.
int conv_add(ConvergenceDetector *d, float value);

// Mean and standard error of the last two windows (or all samples)
void conv_stats(const ConvergenceDetector *d, double *mean, double *sem);

#endif // CONVERGENCE_H
