/*
 * CLI parsing tests. No GL context needed.
 */

#include "../lib/cli.h"

#include <getopt.h>
#include <stdio.h>
#include <string.h>

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

// Parse up to two arguments after a fake argv[0]
static int parse(CliOptions *opts, const char *arg0, const char *arg1) {
    char prog[] = "3d_fluid_simulation_car";
    char a0[600] = "", a1[600] = "";
    char *argv[4] = {prog, NULL, NULL, NULL};
    int argc = 1;
    if (arg0) {
        snprintf(a0, sizeof(a0), "%s", arg0);
        argv[argc++] = a0;
    }
    if (arg1) {
        snprintf(a1, sizeof(a1), "%s", arg1);
        argv[argc++] = a1;
    }
    optind = 0;
    opterr = 0;
    return cli_parse(argc, argv, opts);
}

static void test_defaults(void) {
    printf("test: defaults\n");
    CliOptions o;
    CHECK(parse(&o, NULL, NULL) == 0, "no args parses");
    CHECK(o.windSpeed == 1.0f, "default wind");
    CHECK(o.collisionMode == 2, "default collision");
    CHECK(o.gridX == 128 && o.gridY == 64 && o.gridZ == 64, "default grid");
    CHECK(o.spongeCells == -1, "default sponge is auto");
    CHECK(o.scaleFromCLI == 0, "scale not set");
}

static void test_valid_values(void) {
    printf("test: valid values\n");
    CliOptions o;
    CHECK(parse(&o, "--wind=2.5", NULL) == 0 && o.windSpeed == 2.5f,
          "float option");
    CHECK(parse(&o, "--wind=9", NULL) == 0 && o.windSpeed == 5.0f,
          "wind clamps to 5");
    CHECK(parse(&o, "--viz=99", NULL) == 0 &&
              o.visualizationMode == CLI_NUM_VIZ_MODES - 1,
          "viz clamps to last mode");
    CHECK(parse(&o, "--grid=64x32x16", NULL) == 0 && o.gridX == 64 &&
              o.gridY == 32 && o.gridZ == 16,
          "XxYxZ grid");
    CHECK(parse(&o, "--grid=128", NULL) == 0 && o.gridX == 128 &&
              o.gridY == 64 && o.gridZ == 64,
          "single number grid");
    CHECK(parse(&o, "-g", "32x16x16") == 0 && o.gridX == 32,
          "short grid option");
    CHECK(parse(&o, "--scale=0.5", NULL) == 0 && o.scaleFromCLI == 1 &&
              o.modelScale == 0.5f,
          "scale sets flag and value");
    CHECK(parse(&o, "--angle=25", NULL) == 0 &&
              strstr(o.modelPath, "ahmed_25deg") != NULL,
          "angle picks Ahmed model");
    CHECK(parse(&o, "--output=/tmp/frames", NULL) == 0 &&
              strcmp(o.outputPath, "/tmp/frames") == 0,
          "output path");
}

static void test_invalid_values(void) {
    printf("test: invalid values are rejected\n");
    CliOptions o;
    const char *bad[] = {
        "--wind=abc",
        "--wind=1.5x",
        "--wind=nan",
        "--wind=",
        "--duration=5.0",
        "--duration=99999999999",
        "--viz=two",
        "--grid=64x32",
        "--grid=0x32x32",
        "--grid=-64x32x32",
        "--grid=64x32x32x8",
        "--grid=64x32x",
        "--grid=1",
        "--scale=0",
        "--scale=-1",
        "--sponge=auto",
        "--bogus",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        char msg[128];
        snprintf(msg, sizeof(msg), "rejects %s", bad[i]);
        CHECK(parse(&o, bad[i], NULL) == -1, msg);
    }

    char longPath[600];
    memset(longPath, 'a', sizeof(longPath) - 1);
    longPath[sizeof(longPath) - 1] = '\0';
    memcpy(longPath, "--model=", 8);
    CHECK(parse(&o, longPath, NULL) == -1, "rejects over-long path");
}

static void test_converge_options(void) {
    printf("test: convergence options\n");
    CliOptions o;
    CHECK(parse(&o, NULL, NULL) == 0 && o.convergeTol == 0.0f &&
              o.convergeWindow == 20 && o.maxDuration == 0,
          "off by default");
    CHECK(parse(&o, "--converge=0.01", "--max-duration=30") == 0 &&
              o.convergeTol == 0.01f && o.renderDuration == 30,
          "max-duration sets the headless cap");
    CHECK(parse(&o, "--converge=0.01", "--duration=10") == 0 &&
              o.renderDuration == 10,
          "duration works as the cap");
    CHECK(parse(&o, "--converge=0.01", "--converge-window=8") == -1,
          "converge needs a headless run");
    CHECK(parse(&o, "--max-duration=30", NULL) == -1,
          "max-duration needs converge");
    CHECK(parse(&o, "--converge=0", "--duration=10") == -1, "zero tolerance");
    CHECK(parse(&o, "--converge=1.5", "--duration=10") == -1,
          "tolerance below 1");
    CHECK(parse(&o, "--converge-window=3", NULL) == -1, "window too small");
}

static void test_help(void) {
    printf("test: --help and --version ask for a clean exit\n");
    CliOptions o;
    CHECK(parse(&o, "--help", NULL) == 1, "--help returns 1");
    CHECK(parse(&o, "--version", NULL) == 1, "--version returns 1");
    const char *v = cli_version();
    CHECK(strstr(v, " (") != NULL && v[strlen(v) - 1] == ')',
          "version string has a git part");
}

int main(void) {
    printf("=== CLI Tests ===\n\n");
    test_defaults();
    test_valid_values();
    test_invalid_values();
    test_converge_options();
    test_help();
    printf("\n%d/%d checks passed\n", tests_run - tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
