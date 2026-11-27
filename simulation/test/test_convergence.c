/*
 * Convergence detector tests on synthetic Cd series. No GL needed.
 */

#include "../lib/convergence.h"

#include <math.h>
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

typedef float (*SeriesFn)(int i);

// Feeds up to n samples, returns the sample count at convergence or -1
static int run(ConvergenceDetector *d, SeriesFn fn, int n) {
    for (int i = 0; i < n; i++) {
        if (conv_add(d, fn(i)))
            return i + 1;
    }
    return -1;
}

static unsigned int lcg = 12345u;
static float noise(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(lcg >> 8) / 16777216.0f - 0.5f) * 2.0f;
}

static float constant(int i) {
    (void)i;
    return 0.4f;
}

static float converging(int i) {
    return 0.35f + 0.5f * expf(-i / 30.0f) + 0.005f * noise();
}

static float fast_oscillation(int i) {
    return 0.35f + 0.05f * sinf(2.0f * 3.14159265f * i / 7.3f);
}

static float slow_oscillation(int i) {
    return 0.35f + 0.1f * sinf(2.0f * 3.14159265f * i / 400.0f);
}

static float drifting(int i) {
    return 0.35f + 0.002f * i;
}

static void test_bad_arguments(void) {
    printf("test: bad arguments are rejected\n");
    ConvergenceDetector d;
    CHECK(conv_init(&d, 0.0f, 20) != 0, "zero tolerance");
    CHECK(conv_init(&d, NAN, 20) != 0, "NaN tolerance");
    CHECK(conv_init(&d, 0.01f, CONV_MIN_WINDOW - 1) != 0, "tiny window");
    conv_free(&d);
}

static void test_constant(void) {
    printf("test: constant series converges after three windows\n");
    ConvergenceDetector d;
    CHECK(conv_init(&d, 0.01f, 10) == 0, "init");
    CHECK(run(&d, constant, 1000) == 30, "stops at the second stable check");
    double mean, sem;
    conv_stats(&d, &mean, &sem);
    CHECK(fabs(mean - 0.4) < 1e-6, "mean is the constant");
    CHECK(sem < 1e-9, "zero standard error");
    conv_free(&d);
}

static void test_converging(void) {
    printf("test: decaying transient converges to its plateau\n");
    ConvergenceDetector d;
    CHECK(conv_init(&d, 0.005f, 20) == 0, "init");
    int at = run(&d, converging, 2000);
    CHECK(at > 60, "not fooled by the transient");
    CHECK(at > 0 && at < 2000, "converges");
    double mean, sem;
    conv_stats(&d, &mean, &sem);
    CHECK(fabs(mean - 0.35) < 0.01, "mean near plateau");
    CHECK(sem > 0.0 && sem < 0.005, "small positive standard error");
    conv_free(&d);
}

static void test_fast_oscillation(void) {
    printf("test: fast oscillation converges in the mean\n");
    ConvergenceDetector d;
    CHECK(conv_init(&d, 0.01f, 40) == 0, "init");
    CHECK(run(&d, fast_oscillation, 2000) > 0, "converges");
    double mean, sem;
    conv_stats(&d, &mean, &sem);
    CHECK(fabs(mean - 0.35) < 0.005, "mean near centre");
    conv_free(&d);
}

static void test_slow_oscillation(void) {
    printf("test: slow oscillation does not converge\n");
    ConvergenceDetector d;
    CHECK(conv_init(&d, 0.01f, 20) == 0, "init");
    CHECK(run(&d, slow_oscillation, 1000) == -1, "never settles");
    CHECK(!d.converged, "flag stays clear");
    conv_free(&d);
}

static void test_drifting(void) {
    printf("test: drifting series does not converge\n");
    ConvergenceDetector d;
    CHECK(conv_init(&d, 0.01f, 20) == 0, "init");
    CHECK(run(&d, drifting, 1000) == -1, "never settles");
    CHECK(d.relChange > 0.01, "relative change stays above tolerance");
    conv_free(&d);
}

int main(void) {
    printf("=== Convergence Tests ===\n\n");
    test_bad_arguments();
    test_constant();
    test_converging();
    test_fast_oscillation();
    test_slow_oscillation();
    test_drifting();
    printf("\n%d/%d checks passed\n", tests_run - tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
