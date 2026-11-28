/*
 * Checkpoint file tests. No GL needed.
 */

#include "../lib/checkpoint.h"

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

#define TAG_A CKPT_TAG('A', 'A', 'A', 'A')
#define TAG_B CKPT_TAG('B', 'B', 'B', 'B')
#define NB    100003

static char path[512];

static CkptHeader sample_header(void) {
    CkptHeader h;
    ckpt_header_init(&h);
    h.nx = 32;
    h.ny = 16;
    h.nz = 16;
    snprintf(h.op, sizeof(h.op), "mrt");
    h.smagorinsky = 1;
    h.smagorinskyCs = 0.1f;
    h.tau = 0.52f;
    h.u = 0.05f;
    h.spongeCells = 4;
    h.spongeTauMax = 1.8f;
    h.periodicYZ = 1;
    h.solidHash = 0x1234;
    h.qHash = 0x5678;
    h.step = 4242;
    snprintf(h.model, sizeof(h.model), "assets/3d-files/ahmed_25deg_m.obj");
    snprintf(h.commit, sizeof(h.commit), "abc123");
    return h;
}

// Header, a 3-int section and an NB-float section written in pieces
static int write_sample(const char *p) {
    char err[CKPT_ERR_LEN];
    CkptHeader h = sample_header();
    CkptWriter *w = ckpt_create(p, &h, err, sizeof(err));
    if (!w)
        return -1;
    int a[3] = {1, 2, 3};
    float *b = (float *)malloc(NB * sizeof(float));
    for (int i = 0; i < NB; i++)
        b[i] = (float)i * 0.5f;
    int rc = ckpt_begin_section(w, TAG_A, sizeof(a));
    rc |= ckpt_write(w, a, sizeof(a));
    rc |= ckpt_begin_section(w, TAG_B, (uint64_t)NB * sizeof(float));
    for (int i = 0; i < NB; i += 7000) {
        int n = NB - i < 7000 ? NB - i : 7000;
        rc |= ckpt_write(w, b + i, (size_t)n * sizeof(float));
    }
    free(b);
    if (rc) {
        ckpt_abort(w);
        return -1;
    }
    return ckpt_commit(w, err, sizeof(err));
}

static unsigned char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    *n = ftell(f);
    rewind(f);
    unsigned char *buf = (unsigned char *)malloc(*n);
    if (fread(buf, 1, *n, f) != (size_t)*n)
        *n = -1;
    fclose(f);
    return buf;
}

static void spill(const char *p, const unsigned char *buf, long n) {
    FILE *f = fopen(p, "wb");
    fwrite(buf, 1, n, f);
    fclose(f);
}

// Opening the mutated copy must fail with a message containing want
static void expect_refusal(const unsigned char *orig,
                           long n,
                           long at,
                           unsigned char value,
                           long keep,
                           const char *want,
                           const char *what) {
    char p[600];
    snprintf(p, sizeof(p), "%s.bad", path);
    unsigned char *buf = (unsigned char *)malloc(n);
    memcpy(buf, orig, n);
    if (at >= 0)
        buf[at] = value;
    spill(p, buf, keep >= 0 ? keep : n);
    free(buf);
    char err[CKPT_ERR_LEN] = "";
    CkptHeader h;
    CkptReader *r = ckpt_open(p, &h, err, sizeof(err));
    CHECK(r == NULL, what);
    if (r)
        ckpt_close(r);
    else if (!strstr(err, want))
        printf("    %s: message was '%s'\n", what, err);
    CHECK(r == NULL && strstr(err, want), what);
    remove(p);
}

static void test_hashes(void) {
    printf("test: CRC-32 and FNV-1a known answers\n");
    CHECK(ckpt_crc32(0, "123456789", 9) == 0xCBF43926u, "crc32 check value");
    CHECK(ckpt_fnv1a(CKPT_FNV_OFFSET, "a", 1) == 0xaf63dc4c8601ec8cull,
          "fnv1a of a");
    unsigned char buf[1000];
    for (int i = 0; i < 1000; i++)
        buf[i] = (unsigned char)(i * 37 + 11);
    uint32_t whole = ckpt_crc32(0, buf, sizeof(buf));
    uint32_t parts = ckpt_crc32(ckpt_crc32(0, buf, 333), buf + 333, 667);
    CHECK(whole == parts, "crc32 in pieces equals crc32 of the whole");
}

static void test_round_trip(void) {
    printf("test: write, rename and read back\n");
    CHECK(write_sample(path) == 0, "write");
    char tmp[600];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *t = fopen(tmp, "rb");
    CHECK(t == NULL, "no temp file left");
    if (t)
        fclose(t);

    char err[CKPT_ERR_LEN];
    CkptHeader h, want = sample_header();
    CkptReader *r = ckpt_open(path, &h, err, sizeof(err));
    CHECK(r != NULL, "open");
    if (!r) {
        printf("    %s\n", err);
        return;
    }
    CHECK(h.nx == 32 && h.ny == 16 && h.nz == 16, "grid");
    CHECK(strcmp(h.op, "mrt") == 0 && strcmp(h.storage, "f-w") == 0, "names");
    CHECK(h.step == 4242 && h.qHash == 0x5678, "step and hash");
    CHECK(strcmp(h.model, want.model) == 0, "model path");
    CHECK(ckpt_check(&h, &want, err, sizeof(err)) == 0, "matches itself");

    uint64_t bytes = 0;
    float *b = (float *)malloc(NB * sizeof(float));
    CHECK(ckpt_find(r, TAG_B, &bytes) == 0 && bytes == NB * sizeof(float),
          "find B");
    CHECK(ckpt_read(r, b, NB * sizeof(float)) == 0, "read B");
    int same = 1;
    for (int i = 0; i < NB; i++)
        same &= b[i] == (float)i * 0.5f;
    CHECK(same, "B contents");
    CHECK(ckpt_read(r, b, 4) != 0, "read past the section end refused");
    free(b);
    int a[3] = {0, 0, 0};
    CHECK(ckpt_find(r, TAG_A, &bytes) == 0 && bytes == sizeof(a), "find A");
    CHECK(ckpt_read(r, a, sizeof(a)) == 0 && a[0] == 1 && a[2] == 3,
          "A contents");
    CHECK(ckpt_find(r, CKPT_TAG('Z', 'Z', 'Z', 'Z'), NULL) != 0,
          "missing section not found");
    ckpt_close(r);
}

static void test_refusals(void) {
    printf("test: refusals\n");
    long n = 0;
    unsigned char *orig = slurp(path, &n);
    CHECK(orig != NULL && n > CKPT_HEADER_BYTES, "sample file");
    if (!orig || n <= CKPT_HEADER_BYTES)
        return;

    char err[CKPT_ERR_LEN] = "";
    CkptHeader h;
    CHECK(ckpt_open("/nonexistent/x.ckpt", &h, err, sizeof(err)) == NULL &&
              strstr(err, "cannot open"),
          "missing file");

    expect_refusal(orig, n, 0, 'X', -1, "bad magic", "magic");
    expect_refusal(orig, n, 8, 2, -1, "format version 2", "version");
    // Byte-swapped marker as a big-endian writer would leave it
    unsigned char *sw = (unsigned char *)malloc(n);
    memcpy(sw, orig, n);
    for (int i = 0; i < 4; i++)
        sw[12 + i] = orig[15 - i];
    expect_refusal(sw, n, -1, 0, -1, "byte order", "endianness");
    free(sw);
    expect_refusal(orig, n, 20, 8, -1, "floats are 8 bytes", "float size");
    expect_refusal(orig, n, -1, 0, 500, "header needs", "short header");
    expect_refusal(orig,
                   n,
                   -1,
                   0,
                   CKPT_HEADER_BYTES + 100,
                   "section BBBB",
                   "cut inside a section");
    expect_refusal(orig, n, -1, 0, n - 8, "no end marker", "cut trailer");
    expect_refusal(orig, n, -1, 0, n - 2, "no end marker", "cut checksum");
    expect_refusal(orig,
                   n,
                   CKPT_HEADER_BYTES + 100,
                   0x5a,
                   -1,
                   "checksum mismatch",
                   "flipped data byte");
    expect_refusal(
        orig, n, 40, 99, -1, "checksum mismatch", "flipped header byte");

    // Trailing bytes after the end marker
    unsigned char *longer = (unsigned char *)malloc(n + 4);
    memcpy(longer, orig, n);
    memset(longer + n, 0, 4);
    expect_refusal(
        longer, n + 4, -1, 0, -1, "after the end marker", "trailing data");
    free(longer);
    free(orig);
}

static void test_header_mismatch(void) {
    printf("test: header mismatches\n");
    char err[CKPT_ERR_LEN];
    CkptHeader a = sample_header(), b;

#define MISMATCH(change, want)                                                 \
    do {                                                                       \
        b = a;                                                                 \
        change;                                                                \
        err[0] = 0;                                                            \
        int rc = ckpt_check(&a, &b, err, sizeof(err));                         \
        if (rc == 0 || !strstr(err, want))                                     \
            printf("    %s: '%s'\n", #change, err);                            \
        CHECK(rc != 0 && strstr(err, want), #change);                          \
    } while (0)

    MISMATCH(b.nx = 64, "grid 32x16x16 does not match 64x16x16");
    MISMATCH(b.nz = 8, "grid");
    MISMATCH(snprintf(b.lattice, sizeof(b.lattice), "D3Q27"), "lattice");
    MISMATCH(snprintf(b.storage, sizeof(b.storage), "f"), "storage");
    MISMATCH(snprintf(b.op, sizeof(b.op), "bgk"),
             "collision operator mrt does not match bgk");
    MISMATCH(b.tau = 0.53f, "tau 0.519999981 does not match 0.529999971");
    MISMATCH(b.u = 0.04f, "inlet velocity");
    MISMATCH(b.smagorinsky = 0, "Smagorinsky flag");
    MISMATCH(b.smagorinskyCs = 0.2f, "Smagorinsky constant");
    MISMATCH(b.spongeCells = 0, "sponge width 4 does not match 0");
    MISMATCH(b.spongeTauMax = 1.5f, "sponge tau");
    MISMATCH(b.periodicX = 1, "periodic x");
    MISMATCH(b.periodicYZ = 0, "periodic y/z");
    MISMATCH(b.turbulence = 0.01f, "inlet turbulence");
    MISMATCH(b.bodyForce[1] = 1e-6f, "body force");
    MISMATCH(b.solidHash = 1, "solid mask");
    MISMATCH(b.qHash = 1, "wall distances");

    // Same geometry under another path name is accepted
    b = a;
    snprintf(b.model, sizeof(b.model), "/elsewhere/ahmed.obj");
    b.step = 1;
    snprintf(b.commit, sizeof(b.commit), "def456");
    CHECK(ckpt_check(&a, &b, err, sizeof(err)) == 0,
          "path, step and commit are not compared");
    // Cs is ignored when Smagorinsky is off on both sides
    a.smagorinsky = 0;
    b = a;
    b.smagorinskyCs = 0.3f;
    CHECK(ckpt_check(&a, &b, err, sizeof(err)) == 0, "Cs ignored when off");
}

static void test_writer_misuse(void) {
    printf("test: a failed write keeps the old file\n");
    long n0 = 0, n1 = 0;
    unsigned char *before = slurp(path, &n0);
    char err[CKPT_ERR_LEN];
    CkptHeader h = sample_header();
    CkptWriter *w = ckpt_create(path, &h, err, sizeof(err));
    CHECK(w != NULL, "create");
    if (!w)
        return;
    int x[4] = {0};
    CHECK(ckpt_begin_section(w, TAG_A, 8) == 0, "begin");
    CHECK(ckpt_write(w, x, 16) != 0, "overlong write refused");
    CHECK(ckpt_commit(w, err, sizeof(err)) != 0, "commit after error fails");

    w = ckpt_create(path, &h, err, sizeof(err));
    ckpt_begin_section(w, TAG_A, 16);
    ckpt_write(w, x, 8);
    CHECK(ckpt_commit(w, err, sizeof(err)) != 0 && strstr(err, "short"),
          "short section refused");

    unsigned char *after = slurp(path, &n1);
    CHECK(before && after && n0 == n1 && memcmp(before, after, n0) == 0,
          "old checkpoint untouched");
    char tmp[600];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *t = fopen(tmp, "rb");
    CHECK(t == NULL, "temp file removed");
    if (t)
        fclose(t);
    free(before);
    free(after);
    CHECK(ckpt_create("/nonexistent/dir/x.ckpt", &h, err, sizeof(err)) ==
                  NULL &&
              strstr(err, "cannot create"),
          "unwritable path");
}

int main(void) {
    printf("=== Checkpoint Tests ===\n\n");
    snprintf(path, sizeof(path), "test_checkpoint.ckpt");
    test_hashes();
    test_round_trip();
    test_refusals();
    test_header_mismatch();
    test_writer_misuse();
    remove(path);
    printf("\n=== Results: %d/%d passed ===\n",
           tests_run - tests_failed,
           tests_run);
    return tests_failed ? 1 : 0;
}
