#include "../lib/convergence.h"

#include <math.h>
#include <stdlib.h>

#define MSER_BATCH    5
#define MSER_MIN_TAIL 5

int conv_init(ConvergenceDetector *d, float tol, int checkEvery) {
    *d = (ConvergenceDetector){0};
    if (!(tol >= 0.0f && tol < 1.0f) || checkEvery < 1)
        return -1;
    d->tol = tol;
    d->checkEvery = checkEvery;
    d->capacity = 256;
    d->samples = (float *)malloc((size_t)d->capacity * sizeof(float));
    return d->samples ? 0 : -1;
}

void conv_free(ConvergenceDetector *d) {
    free(d->samples);
    d->samples = NULL;
    d->count = d->capacity = 0;
}

int conv_add(ConvergenceDetector *d, float value) {
    if (!d->samples)
        return 0;
    if (!isfinite(value)) {
        d->rejected++;
        return 0;
    }
    if (d->count == d->capacity) {
        int cap = d->capacity * 2;
        float *p = (float *)realloc(d->samples, (size_t)cap * sizeof(float));
        if (!p) {
            d->allocFailed = 1;
            return 0;
        }
        d->samples = p;
        d->capacity = cap;
    }
    d->samples[d->count++] = value;

    if (!(d->tol > 0.0f) || d->converged || d->count < CONV_MIN_SAMPLES ||
        d->count % d->checkEvery != 0)
        return 0;
    ConvStats s;
    conv_series_stats(d->samples, d->count, &s);
    if (conv_stop_reason(&s, d->tol))
        return 0;
    d->converged = 1;
    return 1;
}

// Truncation point of MSER-5 (White 1997, Franklin and White 2008)
static int mser5(const float *x, int n, int *ok) {
    int m = n / MSER_BATCH;
    double mean = 0.0, m2 = 0.0, best = INFINITY;
    int bestD = 0;
    // Welford over the batch means, from the end backwards
    for (int d = m - 1; d >= 0; d--) {
        double z = 0.0;
        for (int i = 0; i < MSER_BATCH; i++)
            z += x[d * MSER_BATCH + i];
        z /= MSER_BATCH;
        int k = m - d;
        double delta = z - mean;
        mean += delta / k;
        m2 += delta * (z - mean);
        if (k < MSER_MIN_TAIL)
            continue;
        double v = m2 / ((double)k * k);
        if (v <= best) {
            best = v;
            bestD = d;
        }
    }
    // A minimum in the second half means the run is too short
    *ok = bestD <= m / 2;
    return (*ok ? bestD : m / 2) * MSER_BATCH;
}

// Overlapping batch means variance (Meketon and Schmeiser 1984)
static double obm_variance(const float *y, int n, double mean, int b) {
    double sum = 0.0, acc = 0.0;
    for (int i = 0; i < b; i++)
        sum += y[i] - mean;
    for (int j = 0;; j++) {
        double dev = sum / b;
        acc += dev * dev;
        if (j + b >= n)
            break;
        sum += (y[j + b] - mean) - (y[j] - mean);
    }
    return (double)n * b / ((double)(n - b) * (n - b + 1)) * acc;
}

// Schruben's bridge statistic as used by Heidelberger and Welch 1983
static double cvm_statistic(const float *y, int n, double mean, double var) {
    if (!(var > 0.0))
        return 0.0;
    double s = 0.0, acc = 0.0;
    for (int i = 0; i < n; i++) {
        s += y[i] - mean;
        acc += s * s;
    }
    return acc / ((double)n * n * var);
}

void conv_series_stats(const float *x, int n, ConvStats *s) {
    *s = (ConvStats){0};
    s->samples = n;
    s->cvm = s->mean = s->sd = s->sem = s->dof = s->ess = NAN;
    s->halfWidth = s->relHalfWidth = s->lag1 = NAN;
    s->drift = s->relDrift = NAN;
    if (!x || n < CONV_MIN_SAMPLES) {
        s->reason = "too few samples";
        return;
    }

    s->truncated = mser5(x, n, &s->truncationOk);
    const float *y = x + s->truncated;
    int k = n - s->truncated;
    s->kept = k;

    double mean = 0.0, m2 = 0.0;
    for (int i = 0; i < k; i++) {
        double delta = y[i] - mean;
        mean += delta / (i + 1);
        m2 += delta * (y[i] - mean);
    }
    double lag = 0.0;
    for (int i = 0; i + 1 < k; i++)
        lag += (y[i] - mean) * (y[i + 1] - mean);
    s->mean = mean;
    s->sd = sqrt(m2 / (k - 1));
    s->lag1 = m2 > 0.0 ? lag / m2 : NAN;

    // Batch size sqrt(n) (Flegal and Jones 2010), a multiple of 6 so a
    // period-2 component cancels in batches of b and b/3
    int b = 6 * (int)(sqrt((double)k) / 6.0);
    if (b < 6)
        b = 6;
    double var = obm_variance(y, k, mean, b);
    // Lugsail r=3 c=1/2 offsets the small-batch bias (Vats and Flegal 2022)
    double lug = 2.0 * var - obm_variance(y, k, mean, b / 3);
    if (lug > 0.0)
        var = lug;
    s->batchSize = b;
    s->dof = 1.5 * ((double)k / b - 1.0);
    s->sem = sqrt(var / k);
    s->ess = var > 0.0 ? k * (m2 / (k - 1)) / var : k;
    s->halfWidth = conv_t975(s->dof) * s->sem;
    if (fabs(mean) > 0.0)
        s->relHalfWidth = s->halfWidth / fabs(mean);

    // Total change of the fitted line (Scandurra et al. 2025, sec. 4.3)
    double st = 0.0;
    for (int i = 0; i < k; i++)
        st += (i - 0.5 * (k - 1)) * (y[i] - mean);
    double slope = st / ((double)k * ((double)k * k - 1.0) / 12.0);
    s->drift = slope * (k - 1);
    if (fabs(mean) > 0.0)
        s->relDrift = fabs(s->drift) / fabs(mean);

    s->cvm = cvm_statistic(y, k, mean, var);
    // A decaying oscillation keeps the MSER minimum late, so only warn
    s->stationary = s->cvm < CONV_CVM_95;
    if (!s->stationary)
        s->reason = "trend in the mean";
    else if (!s->truncationOk)
        s->reason = "transient may not be over";
    s->valid = 1;
}

void conv_stats(const ConvergenceDetector *d, ConvStats *s) {
    conv_series_stats(d->samples, d->count, s);
    s->rejected = d->rejected;
    if (d->allocFailed) {
        s->valid = 0;
        s->reason = "sample buffer allocation failed";
    }
}

const char *conv_stop_reason(const ConvStats *s, float tol) {
    if (!s->valid || !s->stationary)
        return s->reason;
    if (s->kept < CONV_MIN_KEPT)
        return "too few samples after the transient";
    if (s->ess < CONV_MIN_ESS)
        return "too few effective samples";
    if (!(tol > 0.0f))
        return "no tolerance set";
    if (!(s->relHalfWidth < tol))
        return "confidence interval too wide";
    if (!(s->relDrift < tol))
        return "mean still drifting";
    return NULL;
}

void conv_print(FILE *f, const ConvStats *s) {
    if (!s->valid) {
        fprintf(f,
                "Drag statistics: not available after %d samples (%s)\n",
                s->samples,
                s->reason ? s->reason : "unknown");
        return;
    }
    fprintf(f,
            "Drag statistics: mean %.5f +/- %.5f (95%% CI, %.2f%% of mean)\n",
            s->mean,
            s->halfWidth,
            100.0 * s->relHalfWidth);
    fprintf(f,
            "  sem %.5f, ESS %.1f, kept %d of %d samples after cutting %d, "
            "lag-1 %.2f\n",
            s->sem,
            s->ess,
            s->kept,
            s->samples,
            s->truncated,
            s->lag1);
    fprintf(f,
            "  drift %.2f%% of mean, stationary %s%s%s\n",
            100.0 * s->relDrift,
            s->stationary ? "yes" : "no",
            s->reason ? ", " : "",
            s->reason ? s->reason : "");
    if (s->lag1 < -0.5)
        fprintf(f, "  strong period-2 oscillation in the force samples\n");
    if (s->rejected > 0)
        fprintf(f, "  %d NaN or Inf samples rejected\n", s->rejected);
}

double conv_t975(double dof) {
    static const double tab[] = {12.706205,
                                 4.302653,
                                 3.182446,
                                 2.776445,
                                 2.570582,
                                 2.446912,
                                 2.364624,
                                 2.306004,
                                 2.262157,
                                 2.228139};
    if (!(dof >= 1.0))
        dof = 1.0;
    if (dof < 10.0) {
        int lo = (int)dof;
        double f = (1.0 / lo - 1.0 / dof) / (1.0 / lo - 1.0 / (lo + 1));
        return tab[lo - 1] + f * (tab[lo] - tab[lo - 1]);
    }
    // Cornish-Fisher expansion (Abramowitz and Stegun 26.7.5)
    const double z = 1.959963984540054;
    double z2 = z * z, v = dof;
    double g1 = (z2 + 1.0) * z / 4.0;
    double g2 = ((5.0 * z2 + 16.0) * z2 + 3.0) * z / 96.0;
    double g3 = (((3.0 * z2 + 19.0) * z2 + 17.0) * z2 - 15.0) * z / 384.0;
    double g4 =
        ((((79.0 * z2 + 776.0) * z2 + 1482.0) * z2 - 1920.0) * z2 - 945.0) * z /
        92160.0;
    return z + g1 / v + g2 / (v * v) + g3 / (v * v * v) + g4 / (v * v * v * v);
}
