#include "../lib/convergence.h"

#include <math.h>
#include <stdlib.h>

#define SEM_BATCHES 8

int conv_init(ConvergenceDetector *d, float tol, int window) {
    *d = (ConvergenceDetector){0};
    if (!(tol > 0.0f) || window < CONV_MIN_WINDOW)
        return -1;
    d->tol = tol;
    d->window = window;
    d->capacity = 4 * window;
    d->samples = (float *)malloc((size_t)d->capacity * sizeof(float));
    d->relChange = NAN;
    return d->samples ? 0 : -1;
}

void conv_free(ConvergenceDetector *d) {
    free(d->samples);
    d->samples = NULL;
    d->count = d->capacity = 0;
}

static double mean_of(const float *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += x[i];
    return s / n;
}

int conv_add(ConvergenceDetector *d, float value) {
    if (!d->samples)
        return 0;
    if (d->count == d->capacity) {
        int cap = d->capacity * 2;
        float *p = (float *)realloc(d->samples, (size_t)cap * sizeof(float));
        if (!p)
            return 0;
        d->samples = p;
        d->capacity = cap;
    }
    d->samples[d->count++] = value;

    int n = d->window;
    if (d->converged || d->count < 2 * n || d->count % n != 0)
        return 0;

    double prev = mean_of(d->samples + d->count - 2 * n, n);
    double last = mean_of(d->samples + d->count - n, n);
    d->relChange = fabs(last - prev) / fmax(fabs(prev), 1e-12);
    d->stableChecks = d->relChange < d->tol ? d->stableChecks + 1 : 0;
    if (d->stableChecks >= 2) {
        d->converged = 1;
        return 1;
    }
    return 0;
}

void conv_stats(const ConvergenceDetector *d, double *mean, double *sem) {
    *mean = NAN;
    *sem = NAN;
    int m = d->count < 2 * d->window ? d->count : 2 * d->window;
    if (m < 1)
        return;
    const float *x = d->samples + d->count - m;
    *mean = mean_of(x, m);

    // SEM by batch means over SEM_BATCHES contiguous batches
    int b = m / SEM_BATCHES;
    if (b < 1)
        return;
    const float *y = x + (m - b * SEM_BATCHES);
    double bm[SEM_BATCHES];
    double grand = 0.0;
    for (int i = 0; i < SEM_BATCHES; i++) {
        bm[i] = mean_of(y + i * b, b);
        grand += bm[i];
    }
    grand /= SEM_BATCHES;
    double var = 0.0;
    for (int i = 0; i < SEM_BATCHES; i++)
        var += (bm[i] - grand) * (bm[i] - grand);
    var /= SEM_BATCHES - 1;
    *sem = sqrt(var / SEM_BATCHES);
}
