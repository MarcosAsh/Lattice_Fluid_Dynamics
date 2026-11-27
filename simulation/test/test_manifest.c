/*
 * Manifest tests. No GL needed. Leaves test_manifest.json for parsing.
 */

#include "../lib/manifest.h"

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    static char buf[65536];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

int main(void) {
    printf("=== Manifest Tests ===\n\n");

    char prog[] = "3d_fluid_simulation_car";
    char *argv[] = {prog, NULL};
    optind = 0;
    CliOptions opts;
    if (cli_parse(1, argv, &opts) != 0) {
        printf("  FAIL: default options\n");
        return 1;
    }

    RunManifest m;
    manifest_init(&m, &opts);
    snprintf(m.startTimeUtc, sizeof(m.startTimeUtc), "2026-01-02T03:04:05Z");
    m.modelPath = "dir\\with \"quotes\"\nand\ttabs\x01";
    m.glRenderer = "llvmpipe";
    m.collisionOperator = "mrt";
    m.tau = 0.52f;
    m.latticeVelocity = 0.05f;
    m.cdSamples = 3;
    m.cd = 0.5f;
    m.cl = INFINITY;
    m.solidCells = 1234;

    const char *path = "test_manifest.json";
    printf("test: writes and renames atomically\n");
    CHECK(manifest_write(path, &m) == 0, "write succeeds");
    FILE *tmp = fopen("test_manifest.json.tmp", "rb");
    CHECK(tmp == NULL, "no temp file left");
    if (tmp)
        fclose(tmp);

    char *s = slurp(path);
    CHECK(s != NULL, "file readable");
    if (s) {
        printf("test: fields and escaping\n");
        CHECK(strstr(s, "\"schema_version\": 1,") != NULL, "schema version");
        CHECK(strstr(s, "\"commit\": \"") != NULL, "git commit");
        CHECK(strstr(s, "\"dirty\": ") != NULL, "dirty flag");
        CHECK(strstr(s, "\"start_time_utc\": \"2026-01-02T03:04:05Z\"") != NULL,
              "start time");
        CHECK(strstr(s,
                     "\"path\": \"dir\\\\with \\\"quotes\\\"\\nand\\ttabs"
                     "\\u0001\"") != NULL,
              "string escaping");
        CHECK(strstr(s, "\"grid\": [128, 64, 64]") != NULL, "grid array");
        CHECK(strstr(s, "\"collision_operator\": \"mrt\"") != NULL,
              "collision operator");
        CHECK(strstr(s, "\"mach\": 0.0866") != NULL, "Mach number");
        CHECK(strstr(s, "\"solid_cells\": 1234") != NULL, "solid cells");
        CHECK(strstr(s, "\"cd_samples\": 3") != NULL, "sample count");
        CHECK(strstr(s, "\"cl\": null") != NULL, "infinity becomes null");
        CHECK(strstr(s, "\"strouhal\": null") != NULL, "unset is null");
        CHECK(strstr(s, "\"output\": null") != NULL, "empty path is null");
        CHECK(strstr(s, "\"stop_frame\": null") != NULL, "no stop frame");
        CHECK(strstr(s, "\"sem_method\": \"batch_means_8\"") != NULL,
              "SEM method");
        CHECK(strstr(s, ": nan") == NULL && strstr(s, ": inf") == NULL &&
                  strstr(s, ": -inf") == NULL,
              "no bare nan or inf");
    }

    printf("test: unwritable path fails\n");
    CHECK(manifest_write("/nonexistent-dir/m.json", &m) != 0,
          "returns an error");

    printf("\n%d/%d checks passed\n", tests_run - tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
