// Lightweight C inference for the Cd/Cl surrogate model.
//
// Architecture (matches ml/train.cpp):
//   Linear(3,256) -> SwiGLU(256,512) -> Linear(256,128)
//   -> SwiGLU(128,256) -> Linear(128,2)
//
// Weight file layout (model.bin):
//   Header: uint32 magic (0x4C545753), uint32 version (1), uint32 count
//   Per param: uint32 ndim, uint32 shape[ndim], float32 data[numel]
//
// Normalizer file (model_norm.bin):
//   float32 mean[3], float32 std_dev[3]

#include "../lib/ml_predict.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC 0x4C545753
#define VERSION 1
#define NUM_PARAMS 12

// Weight dimensions baked into the architecture
#define IN_DIM 3
#define H1 256
#define HID1 512
#define H2 128
#define HID2 256
#define OUT_DIM 2

struct MLPredictor {
    // Normalizer
    float mean[3];
    float std_dev[3];

    // fc1: Linear(3, 256)
    float fc1_W[H1 * IN_DIM];   // [256 x 3]
    float fc1_b[H1];            // [256 x 1]

    // act1: SwiGLU(256, 512)
    float act1_Wg[HID1 * H1];   // [512 x 256]
    float act1_Wu[HID1 * H1];   // [512 x 256]
    float act1_Wd[H1 * HID1];   // [256 x 512]

    // fc2: Linear(256, 128)
    float fc2_W[H2 * H1];       // [128 x 256]
    float fc2_b[H2];            // [128 x 1]

    // act2: SwiGLU(128, 256)
    float act2_Wg[HID2 * H2];   // [256 x 128]
    float act2_Wu[HID2 * H2];   // [256 x 128]
    float act2_Wd[H2 * HID2];   // [128 x 256]

    // fc3: Linear(128, 2)
    float fc3_W[OUT_DIM * H2];   // [2 x 128]
    float fc3_b[OUT_DIM];        // [2 x 1]
};

// --------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------

// Read one parameter blob from the file and copy into dst.
// Validates ndim, shape, and total element count.
static int read_param(FILE *f, float *dst, int expect_rows,
                      int expect_cols) {
    uint32_t ndim;
    if (fread(&ndim, 4, 1, f) != 1)
        return 0;

    uint32_t total = 1;
    for (uint32_t d = 0; d < ndim; d++) {
        uint32_t s;
        if (fread(&s, 4, 1, f) != 1)
            return 0;
        // Validate shape: first dim is rows, second (if present) is cols
        if (d == 0 && (int)s != expect_rows)
            return 0;
        if (d == 1 && (int)s != expect_cols)
            return 0;
        total *= s;
    }

    if ((int)total != expect_rows * expect_cols)
        return 0;

    if (fread(dst, sizeof(float), total, f) != total)
        return 0;

    return 1;
}

// Matrix-vector multiply: y = A * x
// A is [rows x cols] stored row-major, x has length cols, y has length rows.
static void matvec(const float *A, const float *x, float *y, int rows,
                   int cols) {
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const float *row = A + r * cols;
        for (int c = 0; c < cols; c++)
            sum += row[c] * x[c];
        y[r] = sum;
    }
}

// Add bias vector in-place: y[i] += b[i]
static void add_bias(float *y, const float *b, int n) {
    for (int i = 0; i < n; i++)
        y[i] += b[i];
}

static float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// SwiGLU forward pass (single vector, not batched):
//   out = W_down @ (swish(W_gate @ x) * (W_up @ x))
//
// tmp_gate has length hidden_dim, tmp_up has length hidden_dim,
// out has length embed_dim.
static void swiglu_forward(const float *Wg, const float *Wu,
                           const float *Wd, const float *x,
                           float *out, float *tmp_gate,
                           float *tmp_up, int embed_dim,
                           int hidden_dim) {
    // gate = W_gate @ x
    matvec(Wg, x, tmp_gate, hidden_dim, embed_dim);

    // up = W_up @ x
    matvec(Wu, x, tmp_up, hidden_dim, embed_dim);

    // swish(gate) * up, in-place into tmp_gate
    for (int i = 0; i < hidden_dim; i++) {
        float sw = tmp_gate[i] * sigmoidf(tmp_gate[i]);
        tmp_gate[i] = sw * tmp_up[i];
    }

    // out = W_down @ tmp_gate
    matvec(Wd, tmp_gate, out, embed_dim, hidden_dim);
}

// --------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------

MLPredictor *ML_Load(const char *weights_path,
                     const char *norm_path) {
    MLPredictor *m =
        (MLPredictor *)calloc(1, sizeof(MLPredictor));
    if (!m)
        return NULL;

    // -- Load normalizer --
    FILE *nf = fopen(norm_path, "rb");
    if (!nf) {
        fprintf(stderr, "ml_predict: cannot open %s\n", norm_path);
        free(m);
        return NULL;
    }
    if (fread(m->mean, sizeof(float), 3, nf) != 3 ||
        fread(m->std_dev, sizeof(float), 3, nf) != 3) {
        fprintf(stderr, "ml_predict: truncated %s\n", norm_path);
        fclose(nf);
        free(m);
        return NULL;
    }
    fclose(nf);

    // -- Load weights --
    FILE *wf = fopen(weights_path, "rb");
    if (!wf) {
        fprintf(stderr, "ml_predict: cannot open %s\n",
                weights_path);
        free(m);
        return NULL;
    }

    uint32_t magic, version, count;
    if (fread(&magic, 4, 1, wf) != 1 || magic != MAGIC) {
        fprintf(stderr, "ml_predict: bad magic in %s\n",
                weights_path);
        fclose(wf);
        free(m);
        return NULL;
    }
    if (fread(&version, 4, 1, wf) != 1 || version != VERSION) {
        fprintf(stderr, "ml_predict: bad version in %s\n",
                weights_path);
        fclose(wf);
        free(m);
        return NULL;
    }
    if (fread(&count, 4, 1, wf) != 1 ||
        count != NUM_PARAMS) {
        fprintf(stderr,
                "ml_predict: param count mismatch "
                "(got %u, expected %d)\n",
                count, NUM_PARAMS);
        fclose(wf);
        free(m);
        return NULL;
    }

    // Parameters are saved in registration order (see ml/train.cpp):
    //  1. fc1.W  [256x3]     2. fc1.b  [256x1]
    //  3. act1.Wg [512x256]  4. act1.Wu [512x256]
    //  5. act1.Wd [256x512]
    //  6. fc2.W  [128x256]   7. fc2.b  [128x1]
    //  8. act2.Wg [256x128]  9. act2.Wu [256x128]
    // 10. act2.Wd [128x256]
    // 11. fc3.W  [2x128]    12. fc3.b  [2x1]
    int ok = 1;
    ok = ok && read_param(wf, m->fc1_W, H1, IN_DIM);
    ok = ok && read_param(wf, m->fc1_b, H1, 1);
    ok = ok && read_param(wf, m->act1_Wg, HID1, H1);
    ok = ok && read_param(wf, m->act1_Wu, HID1, H1);
    ok = ok && read_param(wf, m->act1_Wd, H1, HID1);
    ok = ok && read_param(wf, m->fc2_W, H2, H1);
    ok = ok && read_param(wf, m->fc2_b, H2, 1);
    ok = ok && read_param(wf, m->act2_Wg, HID2, H2);
    ok = ok && read_param(wf, m->act2_Wu, HID2, H2);
    ok = ok && read_param(wf, m->act2_Wd, H2, HID2);
    ok = ok && read_param(wf, m->fc3_W, OUT_DIM, H2);
    ok = ok && read_param(wf, m->fc3_b, OUT_DIM, 1);

    fclose(wf);

    if (!ok) {
        fprintf(stderr, "ml_predict: failed reading params "
                        "from %s\n",
                weights_path);
        free(m);
        return NULL;
    }

    return m;
}

void ML_Free(MLPredictor *m) {
    free(m);
}

void ML_Predict(MLPredictor *m, float wind_speed,
                float reynolds, float model_id, float *cd,
                float *cl) {
    // Scratch buffers on the stack. Largest intermediate is 512
    // floats (the SwiGLU hidden dimension for act1).
    float buf_a[HID1]; // max(HID1, HID2) = 512
    float buf_b[HID1];
    float h1[H1];      // 256
    float h2[H2];      // 128
    float out[OUT_DIM]; // 2

    // Z-score normalize inputs
    float x[IN_DIM];
    x[0] = (wind_speed - m->mean[0]) / m->std_dev[0];
    x[1] = (reynolds - m->mean[1]) / m->std_dev[1];
    x[2] = (model_id - m->mean[2]) / m->std_dev[2];

    // fc1: h1 = W1 * x + b1          [256]
    matvec(m->fc1_W, x, h1, H1, IN_DIM);
    add_bias(h1, m->fc1_b, H1);

    // act1: SwiGLU(256, 512) -> h1    [256]
    swiglu_forward(m->act1_Wg, m->act1_Wu, m->act1_Wd, h1,
                   h1, buf_a, buf_b, H1, HID1);

    // fc2: h2 = W2 * h1 + b2          [128]
    matvec(m->fc2_W, h1, h2, H2, H1);
    add_bias(h2, m->fc2_b, H2);

    // act2: SwiGLU(128, 256) -> h2    [128]
    swiglu_forward(m->act2_Wg, m->act2_Wu, m->act2_Wd, h2,
                   h2, buf_a, buf_b, H2, HID2);

    // fc3: out = W3 * h2 + b3         [2]
    matvec(m->fc3_W, h2, out, OUT_DIM, H2);
    add_bias(out, m->fc3_b, OUT_DIM);

    *cd = out[0];
    *cl = out[1];
}
