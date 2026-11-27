/*
 * Convergence statistics on synthetic Cd series with known answers.
 * No GL needed.
 */

#include "../lib/convergence.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int tests_run = 0;
static int tests_failed = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        tests_run++;                                                           \
        if (!(cond)) {                                                         \
            tests_failed++;                                                    \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__);                   \
        }                                                                      \
    } while (0)

#define TRIALS 400

// splitmix64 (Steele, Lea and Flood 2014)
static uint64_t rng_state;

static void seed(uint64_t s) {
    rng_state = s;
}

static double uniform(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return ((double)(z >> 11) + 0.5) / 9007199254740992.0;
}

static double gauss(void) {
    return sqrt(-2.0 * log(uniform())) * cos(6.283185307179586 * uniform());
}

typedef enum { WHITE, AR1, PERIOD2, TRANSIENT, DRIFT, GENTLE, RINGDOWN } Kind;

static void fill(float *x, int n, Kind kind, uint64_t s) {
    seed(s);
    double a = gauss() / sqrt(1.0 - 0.81);
    for (int i = 0; i < n; i++) {
        switch (kind) {
        case WHITE:
            x[i] = (float)(0.4 + 0.02 * gauss());
            break;
        case AR1:
            a = 0.9 * a + gauss();
            x[i] = (float)(0.4 + 0.01 * a);
            break;
        case PERIOD2:
            x[i] = (float)(5.0 + (i % 2 ? -3.5 : 3.5) + gauss());
            break;
        case TRANSIENT:
            x[i] = (float)(1.0 + 2.0 * exp(-i / 50.0) + 0.05 * gauss());
            break;
        case DRIFT:
            x[i] = (float)(1.0 + 0.001 * i + 0.05 * gauss());
            break;
        case GENTLE:
            x[i] = (float)(1.0 + 0.0003 * i + 0.05 * gauss());
            break;
        case RINGDOWN:
            x[i] = (float)(3.5 + exp(-i / 80.0) * (0.3 + (i % 2 ? -3 : 3)) +
                           0.002 * gauss());
            break;
        }
    }
}

typedef struct {
    double coverage, essRatio, semRatio, stationary;
} Trials;

// essRef is the true ESS per sample, sd the true sem * sqrt(n)
static Trials trials(Kind kind, int n, double mu, double essRef, double sd) {
    float *x = (float *)malloc((size_t)n * sizeof(float));
    Trials t = {0};
    for (int k = 0; k < TRIALS; k++) {
        fill(x, n, kind, 1000u + (uint64_t)k);
        ConvStats s;
        conv_series_stats(x, n, &s);
        t.coverage += fabs(s.mean - mu) <= s.halfWidth;
        t.essRatio += s.ess / (essRef * s.kept);
        t.semRatio += s.sem * sqrt((double)s.kept) / sd;
        t.stationary += s.stationary;
    }
    free(x);
    t.coverage /= TRIALS;
    t.essRatio /= TRIALS;
    t.semRatio /= TRIALS;
    t.stationary /= TRIALS;
    printf("  coverage %.3f, ESS/true %.3f, sem/true %.3f, stationary %.3f\n",
           t.coverage,
           t.essRatio,
           t.semRatio,
           t.stationary);
    return t;
}

// Fraction of stopped runs whose interval covers mu, -1 if many never stop
static double sequential(Kind kind, int n, double mu, float tol) {
    float *x = (float *)malloc((size_t)n * sizeof(float));
    int covered = 0, stopped = 0;
    long total = 0;
    for (int k = 0; k < TRIALS; k++) {
        fill(x, n, kind, 5000u + (uint64_t)k);
        ConvergenceDetector d;
        conv_init(&d, tol, 20);
        for (int i = 0; i < n; i++) {
            if (conv_add(&d, x[i])) {
                ConvStats s;
                conv_stats(&d, &s);
                covered += fabs(s.mean - mu) <= s.halfWidth;
                stopped++;
                total += d.count;
                break;
            }
        }
        conv_free(&d);
    }
    free(x);
    printf("  stopped %d/%d at %ld samples on average, coverage %.3f\n",
           stopped,
           TRIALS,
           stopped ? total / stopped : 0,
           stopped ? (double)covered / stopped : 0.0);
    return stopped >= TRIALS * 0.95 ? (double)covered / stopped : -1.0;
}

static void test_bad_arguments(void) {
    printf("test: bad arguments are rejected\n");
    ConvergenceDetector d;
    CHECK(conv_init(&d, -0.01f, 20) != 0, "negative tolerance");
    CHECK(conv_init(&d, NAN, 20) != 0, "NaN tolerance");
    CHECK(conv_init(&d, 1.0f, 20) != 0, "tolerance of one");
    CHECK(conv_init(&d, 0.01f, 0) != 0, "zero check interval");
    CHECK(conv_init(&d, 0.0f, 1) == 0, "statistics only");
    conv_free(&d);
}

static void test_t_quantile(void) {
    printf("test: Student t quantiles\n");
    CHECK(fabs(conv_t975(1.0) - 12.706205) < 1e-6, "nu=1");
    CHECK(fabs(conv_t975(5.0) - 2.570582) < 1e-6, "nu=5");
    CHECK(fabs(conv_t975(10.0) - 2.228139) < 1e-4, "nu=10");
    CHECK(fabs(conv_t975(30.0) - 2.042272) < 1e-5, "nu=30");
    CHECK(fabs(conv_t975(100.0) - 1.983972) < 1e-5, "nu=100");
    CHECK(fabs(conv_t975(1e9) - 1.959964) < 1e-5, "normal limit");
    CHECK(conv_t975(5.5) < conv_t975(5.0) && conv_t975(5.5) > conv_t975(6.0),
          "interpolates between integers");
}

static void test_white_noise(void) {
    printf("test: white noise\n");
    Trials t = trials(WHITE, 4000, 0.4, 1.0, 0.02);
    CHECK(t.coverage >= 0.90 && t.coverage <= 0.99, "coverage near 95%");
    CHECK(t.essRatio > 0.9 && t.essRatio < 1.1, "ESS close to n");
    CHECK(t.semRatio > 0.9 && t.semRatio < 1.1, "sem close to sd/sqrt(n)");
    CHECK(t.stationary > 0.9, "mostly stationary");
}

static void test_ar1(void) {
    printf("test: AR(1) with phi 0.9\n");
    // ESS per sample is (1-phi)/(1+phi), sem * sqrt(n) is sigma/(1-phi)
    Trials t = trials(AR1, 10000, 0.4, 0.1 / 1.9, 0.01 / 0.1);
    CHECK(t.coverage >= 0.90 && t.coverage <= 0.99, "coverage near 95%");
    CHECK(t.essRatio > 0.8 && t.essRatio < 1.25, "ESS near n(1-phi)/(1+phi)");
    CHECK(t.semRatio > 0.85 && t.semRatio < 1.1, "sem near the true value");
    CHECK(t.stationary > 0.85, "mostly stationary");
}

static void test_period2(void) {
    printf("test: period-2 oscillation plus noise\n");
    // Only the noise moves the mean, so sem * sqrt(n) is the noise sd 1
    Trials t = trials(PERIOD2, 2000, 5.0, 1.0 + 3.5 * 3.5, 1.0);
    CHECK(t.coverage >= 0.90 && t.coverage <= 0.99, "coverage near 95%");
    CHECK(t.semRatio > 0.85 && t.semRatio < 1.15, "sem near the noise sem");

    float x[2000];
    fill(x, 2000, PERIOD2, 7u);
    ConvStats s;
    conv_series_stats(x, 2000, &s);
    CHECK(s.lag1 < -0.8, "lag-1 shows the oscillation");
    CHECK(s.ess > s.kept, "anticorrelation gives ESS above n");
    CHECK(s.sem < 0.5 * s.sd / sqrt((double)s.kept), "naive sem is inflated");
    CHECK(s.stationary, "stationary");
    CHECK(s.truncated < 200, "no transient to cut");

    double cov = sequential(PERIOD2, 4000, 5.0, 0.01f);
    CHECK(cov >= 0.88 && cov <= 1.0, "stopped intervals cover the mean");
}

static void test_transient(void) {
    printf("test: decaying transient then noise\n");
    // 2 exp(-i/50) falls below the noise sd 0.05 at i = 50 ln 40 = 184
    float x[2000];
    int lo = 2000, hi = 0, covered = 0, stationary = 0;
    for (int k = 0; k < 50; k++) {
        fill(x, 2000, TRANSIENT, 300u + (uint64_t)k);
        ConvStats s;
        conv_series_stats(x, 2000, &s);
        lo = s.truncated < lo ? s.truncated : lo;
        hi = s.truncated > hi ? s.truncated : hi;
        covered += fabs(s.mean - 1.0) <= s.halfWidth;
        stationary += s.truncationOk && s.stationary && !s.reason;
    }
    printf("  truncation between %d and %d, %d/50 cover\n", lo, hi, covered);
    CHECK(lo >= 92 && hi <= 368, "truncation near the end of the transient");
    CHECK(covered >= 40, "intervals cover the plateau");
    CHECK(stationary >= 40, "plateau is stationary");

    ConvStats all;
    fill(x, 2000, TRANSIENT, 300u);
    conv_series_stats(x, 2000, &all);
    CHECK(all.mean < 1.0 + all.halfWidth, "transient left out of the mean");
}

static void test_ringdown(void) {
    printf("test: decaying period-2 oscillation, as seen at low tau\n");
    // The leftover decay biases the mean by about one half-width at the stop
    float x[2000];
    int stopped = 0, covered = 0, close = 0;
    for (int k = 0; k < 20; k++) {
        fill(x, 2000, RINGDOWN, 700u + (uint64_t)k);
        ConvStats early;
        conv_series_stats(x, 300, &early);
        CHECK(early.lag1 < -0.9, "lag-1 shows the oscillation");
        ConvergenceDetector d;
        conv_init(&d, 0.02f, 20);
        for (int i = 0; i < 2000 && !d.converged; i++)
            conv_add(&d, x[i]);
        ConvStats s;
        conv_stats(&d, &s);
        stopped += d.converged;
        covered += fabs(s.mean - 3.5) <= s.halfWidth;
        close += fabs(s.mean - 3.5) <= 3.0 * s.halfWidth &&
                 fabs(s.mean - 3.5) < 0.002 * 3.5;
        conv_free(&d);
    }
    printf("  stopped %d/20, %d/20 cover, %d/20 within 3 half-widths\n",
           stopped,
           covered,
           close);
    CHECK(stopped == 20, "stops once the mean has settled");
    CHECK(close == 20, "error stays near the half-width and far under tol");
}

static void test_drift(void) {
    printf("test: linear drift never converges\n");
    float x[5000];
    for (int k = 0; k < 20; k++) {
        fill(x, 5000, DRIFT, 900u + (uint64_t)k);
        ConvergenceDetector d;
        CHECK(conv_init(&d, 0.01f, 5) == 0, "init");
        int stopped = 0;
        for (int i = 0; i < 5000; i++)
            stopped |= conv_add(&d, x[i]);
        ConvStats s;
        conv_stats(&d, &s);
        CHECK(!stopped && !d.converged, "never stops");
        CHECK(!s.stationary && s.reason, "fails stationarity with a reason");
        conv_free(&d);
    }
}

static void test_gentle_drift(void) {
    printf("test: drift near the tolerance rarely stops\n");
    // 200 samples drift by 0.06, six times the tolerance
    float x[4000];
    int stops = 0;
    for (int k = 0; k < 100; k++) {
        fill(x, 4000, GENTLE, 1300u + (uint64_t)k);
        ConvergenceDetector d;
        conv_init(&d, 0.01f, 20);
        for (int i = 0; i < 4000 && !d.converged; i++)
            stops += conv_add(&d, x[i]);
        conv_free(&d);
    }
    printf("  stopped %d/100\n", stops);
    CHECK(stops <= 5, "drift check holds");
}

static void test_constant(void) {
    printf("test: constant series\n");
    ConvergenceDetector d;
    CHECK(conv_init(&d, 0.01f, 10) == 0, "init");
    int at = -1;
    for (int i = 0; i < 1000 && at < 0; i++)
        if (conv_add(&d, 0.4f))
            at = i + 1;
    ConvStats s;
    conv_stats(&d, &s);
    CHECK(at == CONV_MIN_KEPT, "stops at the minimum sample count");
    CHECK(s.valid && s.stationary && s.truncated == 0, "stationary, no cut");
    CHECK(s.mean == (double)0.4f, "mean is the constant");
    CHECK(s.sd == 0.0 && s.sem == 0.0 && s.halfWidth == 0.0, "zero spread");
    CHECK(s.ess == s.kept && isfinite(s.cvm), "finite ESS and statistic");
    conv_free(&d);
}

static void test_short(void) {
    printf("test: short series is refused\n");
    ConvergenceDetector d;
    CHECK(conv_init(&d, 0.5f, 1) == 0, "init");
    int stopped = 0;
    for (int i = 0; i < CONV_MIN_SAMPLES - 1; i++)
        stopped |= conv_add(&d, 0.4f + 0.01f * (float)(i % 3));
    ConvStats s;
    conv_stats(&d, &s);
    CHECK(!stopped && !s.valid, "no statistics");
    CHECK(s.reason != NULL, "says why");
    CHECK(isnan(s.mean) && isnan(s.halfWidth), "results are NaN");
    CHECK(conv_stop_reason(&s, 0.5f) != NULL, "stop rule refuses");
    conv_free(&d);
}

static void test_non_finite(void) {
    printf("test: NaN and Inf samples are rejected\n");
    float x[600];
    fill(x, 600, WHITE, 42u);
    ConvergenceDetector d;
    CHECK(conv_init(&d, 0.0f, 1) == 0, "init");
    for (int i = 0; i < 600; i++) {
        if (i % 50 == 7)
            conv_add(&d, NAN);
        if (i % 75 == 3)
            conv_add(&d, i % 2 ? INFINITY : -INFINITY);
        conv_add(&d, x[i]);
    }
    ConvStats s, ref;
    conv_stats(&d, &s);
    conv_series_stats(x, 600, &ref);
    CHECK(s.rejected == 12 + 8, "rejected samples counted");
    CHECK(s.samples == 600, "finite samples kept");
    CHECK(s.mean == ref.mean && s.sem == ref.sem, "same as the clean series");
    CHECK(isfinite(s.mean) && isfinite(s.halfWidth), "finite results");
    conv_free(&d);
}

static void test_sequential_ar1(void) {
    printf("test: stopping rule on AR(1)\n");
    double cov = sequential(AR1, 20000, 0.4, 0.01f);
    CHECK(cov >= 0.88 && cov <= 1.0, "stopped intervals cover the mean");
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("=== Convergence Tests ===\n\n");
    test_bad_arguments();
    test_t_quantile();
    test_white_noise();
    test_ar1();
    test_period2();
    test_transient();
    test_ringdown();
    test_drift();
    test_gentle_drift();
    test_constant();
    test_short();
    test_non_finite();
    test_sequential_ar1();
    printf("\n%d/%d checks passed\n", tests_run - tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
