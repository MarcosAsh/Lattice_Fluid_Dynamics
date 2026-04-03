#include "../lib/superres.h"
#include "../lib/opengl_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define LTWS_MAGIC 0x4C545753
#define HIDDEN_DIM 128

struct SuperResUpscaler {
    int coarseX, coarseY, coarseZ;
    int fineX, fineY, fineZ;

    GLuint shaderProgram;
    GLuint fineVelBuffer;
    GLuint weightsBuffer;

    // Uniform locations
    GLint coarseSizeLoc;
    GLint fineSizeLoc;
    GLint normMeanLoc;
    GLint normStdLoc;
    GLint hiddenDimLoc;
    GLint wOff0Loc, bOff0Loc;
    GLint wOff1Loc, bOff1Loc;
    GLint wOff2Loc, bOff2Loc;
    GLint wOff3Loc, bOff3Loc;

    // Weight offsets (float indices into the packed buffer)
    int wOff0, bOff0;
    int wOff1, bOff1;
    int wOff2, bOff2;
    int wOff3, bOff3;

    // Normalizer
    float normMean[4];
    float normStd[4];
};

// Load LTWS weights into a flat float array.
// Returns the array and sets *outCount to the total number of floats.
static float *load_ltws_flat(const char *path, int *outCount) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("SR: cannot open weights file: %s\n", path);
        return NULL;
    }

    uint32_t magic, version, paramCount;
    if (fread(&magic, 4, 1, f) != 1 || magic != LTWS_MAGIC) {
        printf("SR: bad magic in %s\n", path);
        fclose(f);
        return NULL;
    }
    fread(&version, 4, 1, f);
    fread(&paramCount, 4, 1, f);

    // First pass: compute total size
    long dataStart = ftell(f);
    int totalFloats = 0;
    for (uint32_t p = 0; p < paramCount; p++) {
        uint32_t ndim;
        fread(&ndim, 4, 1, f);
        int numel = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint32_t s;
            fread(&s, 4, 1, f);
            numel *= (int)s;
        }
        fseek(f, numel * 4, SEEK_CUR);
        totalFloats += numel;
    }

    // Second pass: read data
    float *data = (float *)malloc(totalFloats * sizeof(float));
    if (!data) {
        fclose(f);
        return NULL;
    }

    fseek(f, dataStart, SEEK_SET);
    int offset = 0;
    for (uint32_t p = 0; p < paramCount; p++) {
        uint32_t ndim;
        fread(&ndim, 4, 1, f);
        int numel = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint32_t s;
            fread(&s, 4, 1, f);
            numel *= (int)s;
        }
        fread(data + offset, sizeof(float), numel, f);
        offset += numel;
    }

    fclose(f);
    *outCount = totalFloats;
    printf("SR: loaded %d weight floats from %s\n", totalFloats, path);
    return data;
}

static int load_normalizer(const char *path, float mean[4], float std[4]) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("SR: cannot open normalizer: %s\n", path);
        // Default: no normalization
        for (int i = 0; i < 4; i++) {
            mean[i] = 0.0f;
            std[i] = 1.0f;
        }
        return 0;
    }
    fread(mean, sizeof(float), 4, f);
    fread(std, sizeof(float), 4, f);
    fclose(f);
    printf("SR: normalizer mean=(%.4f, %.4f, %.4f, %.4f) "
           "std=(%.4f, %.4f, %.4f, %.4f)\n",
           mean[0], mean[1], mean[2], mean[3],
           std[0], std[1], std[2], std[3]);
    return 1;
}

// Compute weight offsets for the 4-layer MLP.
// Architecture: Linear(108, H) + Linear(H, H) + Linear(H, H) + Linear(H, 32)
// PyTorch state dict keys are sorted alphabetically:
//   net.0.bias, net.0.weight, net.2.bias, net.2.weight, ...
// So the order is: b0, W0, b1, W1, b2, W2, b3, W3
static void compute_offsets(SuperResUpscaler *sr) {
    int H = HIDDEN_DIM;
    int off = 0;

    // net.0.bias (H)
    sr->bOff0 = off; off += H;
    // net.0.weight (H, 108)
    sr->wOff0 = off; off += H * 108;
    // net.2.bias (H)
    sr->bOff1 = off; off += H;
    // net.2.weight (H, H)
    sr->wOff1 = off; off += H * H;
    // net.4.bias (H)
    sr->bOff2 = off; off += H;
    // net.4.weight (H, H)
    sr->wOff2 = off; off += H * H;
    // net.6.bias (32)
    sr->bOff3 = off; off += 32;
    // net.6.weight (32, H)
    sr->wOff3 = off; off += 32 * H;

    printf("SR: weight offsets: W0=%d b0=%d W1=%d b1=%d "
           "W2=%d b2=%d W3=%d b3=%d (total=%d)\n",
           sr->wOff0, sr->bOff0, sr->wOff1, sr->bOff1,
           sr->wOff2, sr->bOff2, sr->wOff3, sr->bOff3, off);
}

SuperResUpscaler *SR_Create(int coarseX, int coarseY, int coarseZ,
                            const char *weightsPath,
                            const char *normPath) {
    SuperResUpscaler *sr = (SuperResUpscaler *)calloc(1, sizeof(SuperResUpscaler));
    if (!sr) return NULL;

    sr->coarseX = coarseX;
    sr->coarseY = coarseY;
    sr->coarseZ = coarseZ;
    sr->fineX = coarseX * 2;
    sr->fineY = coarseY * 2;
    sr->fineZ = coarseZ * 2;

    printf("SR: coarse %dx%dx%d -> fine %dx%dx%d\n",
           coarseX, coarseY, coarseZ,
           sr->fineX, sr->fineY, sr->fineZ);

    // Load and compile compute shader
    sr->shaderProgram = createComputeShader("shaders/superres_upscale.comp");
    if (!sr->shaderProgram) {
        printf("SR: failed to compile shader\n");
        free(sr);
        return NULL;
    }

    // Get uniform locations
    glUseProgram(sr->shaderProgram);
    sr->coarseSizeLoc = glGetUniformLocation(sr->shaderProgram, "coarseSize");
    sr->fineSizeLoc = glGetUniformLocation(sr->shaderProgram, "fineSize");
    sr->normMeanLoc = glGetUniformLocation(sr->shaderProgram, "normMean");
    sr->normStdLoc = glGetUniformLocation(sr->shaderProgram, "normStd");
    sr->hiddenDimLoc = glGetUniformLocation(sr->shaderProgram, "hiddenDim");
    sr->wOff0Loc = glGetUniformLocation(sr->shaderProgram, "wOff0");
    sr->bOff0Loc = glGetUniformLocation(sr->shaderProgram, "bOff0");
    sr->wOff1Loc = glGetUniformLocation(sr->shaderProgram, "wOff1");
    sr->bOff1Loc = glGetUniformLocation(sr->shaderProgram, "bOff1");
    sr->wOff2Loc = glGetUniformLocation(sr->shaderProgram, "wOff2");
    sr->bOff2Loc = glGetUniformLocation(sr->shaderProgram, "bOff2");
    sr->wOff3Loc = glGetUniformLocation(sr->shaderProgram, "wOff3");
    sr->bOff3Loc = glGetUniformLocation(sr->shaderProgram, "bOff3");

    // Load weights
    int weightCount = 0;
    float *weightData = load_ltws_flat(weightsPath, &weightCount);
    if (!weightData) {
        glDeleteProgram(sr->shaderProgram);
        free(sr);
        return NULL;
    }

    // Upload weights to SSBO
    glGenBuffers(1, &sr->weightsBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sr->weightsBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 weightCount * sizeof(float), weightData, GL_STATIC_DRAW);
    free(weightData);

    // Compute offsets
    compute_offsets(sr);

    // Load normalizer
    load_normalizer(normPath, sr->normMean, sr->normStd);

    // Allocate fine velocity buffer
    size_t fineSize = (size_t)sr->fineX * sr->fineY * sr->fineZ * 4 * sizeof(float);
    glGenBuffers(1, &sr->fineVelBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sr->fineVelBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, fineSize, NULL, GL_DYNAMIC_COPY);

    printf("SR: fine velocity buffer: %.1f MB\n", fineSize / (1024.0 * 1024.0));
    printf("SR: initialization complete\n");

    return sr;
}

void SR_Free(SuperResUpscaler *sr) {
    if (!sr) return;
    if (sr->shaderProgram) glDeleteProgram(sr->shaderProgram);
    if (sr->fineVelBuffer) glDeleteBuffers(1, &sr->fineVelBuffer);
    if (sr->weightsBuffer) glDeleteBuffers(1, &sr->weightsBuffer);
    free(sr);
}

void SR_Upscale(SuperResUpscaler *sr, GLuint coarseVelBuffer) {
    if (!sr || !sr->shaderProgram) return;

    glUseProgram(sr->shaderProgram);

    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coarseVelBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, sr->fineVelBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, sr->weightsBuffer);

    // Set uniforms
    glUniform3i(sr->coarseSizeLoc, sr->coarseX, sr->coarseY, sr->coarseZ);
    glUniform3i(sr->fineSizeLoc, sr->fineX, sr->fineY, sr->fineZ);
    glUniform4f(sr->normMeanLoc,
                sr->normMean[0], sr->normMean[1],
                sr->normMean[2], sr->normMean[3]);
    glUniform4f(sr->normStdLoc,
                sr->normStd[0], sr->normStd[1],
                sr->normStd[2], sr->normStd[3]);
    glUniform1i(sr->hiddenDimLoc, HIDDEN_DIM);
    glUniform1i(sr->wOff0Loc, sr->wOff0);
    glUniform1i(sr->bOff0Loc, sr->bOff0);
    glUniform1i(sr->wOff1Loc, sr->wOff1);
    glUniform1i(sr->bOff1Loc, sr->bOff1);
    glUniform1i(sr->wOff2Loc, sr->wOff2);
    glUniform1i(sr->bOff2Loc, sr->bOff2);
    glUniform1i(sr->wOff3Loc, sr->wOff3);
    glUniform1i(sr->bOff3Loc, sr->bOff3);

    // Dispatch: one workgroup per 4x4x4 coarse cells
    int gx = (sr->coarseX + 3) / 4;
    int gy = (sr->coarseY + 3) / 4;
    int gz = (sr->coarseZ + 3) / 4;
    glDispatchCompute(gx, gy, gz);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

GLuint SR_GetFineVelocityBuffer(SuperResUpscaler *sr) {
    return sr ? sr->fineVelBuffer : 0;
}

void SR_GetFineSize(SuperResUpscaler *sr, int *fineX, int *fineY, int *fineZ) {
    if (!sr) return;
    if (fineX) *fineX = sr->fineX;
    if (fineY) *fineY = sr->fineY;
    if (fineZ) *fineZ = sr->fineZ;
}
