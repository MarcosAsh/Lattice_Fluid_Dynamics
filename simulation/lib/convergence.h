#ifndef CONVERGENCE_H
#define CONVERGENCE_H

// Converged once the last two window means differ by under tol twice running
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

// Returns 1 on the sample that reaches convergence
int conv_add(ConvergenceDetector *d, float value);

// Mean and standard error of the last two windows (or all samples)
void conv_stats(const ConvergenceDetector *d, double *mean, double *sem);

#endif // CONVERGENCE_H
