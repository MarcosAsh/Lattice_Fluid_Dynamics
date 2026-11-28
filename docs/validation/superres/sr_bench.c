// Times SR_Upscale against a trilinear 2x shader and dumps SR output
// Build against an existing build dir B, run from simulation/:
//   gcc -O2 -std=gnu17 -Ilib -Ilib/glad/include sr_bench.c -o sr_bench
//       B/liblattice_gl.a B/liblattice_mesh.a B/liblattice_io.a
//       B/liblattice_glad.a $(pkg-config --libs sdl2 gl egl) -lm -ldl
#include "gl_context.h"
#include "lbm.h"
#include "opengl_utils.h"
#include "superres.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

// Wall time per call with the queue drained before and after
static double time_dispatch(GLuint q, void (*fn)(void *), void *arg, int reps) {
    (void)q;
    fn(arg);
    glFinish();
    double t0 = now_ms();
    for (int r = 0; r < reps; r++)
        fn(arg);
    glFinish();
    return (now_ms() - t0) / reps;
}

typedef struct {
    SuperResUpscaler *sr;
    GLuint coarse, fine, prog;
    int cx, cy, cz;
    LBMGrid *grid;
} Ctx;

static void run_sr(void *a) {
    Ctx *c = a;
    SR_Upscale(c->sr, c->coarse);
}

static void run_tri(void *a) {
    Ctx *c = a;
    glUseProgram(c->prog);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, c->coarse);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, c->fine);
    glUniform3i(
        glGetUniformLocation(c->prog, "coarseSize"), c->cx, c->cy, c->cz);
    glUniform3i(glGetUniformLocation(c->prog, "fineSize"),
                2 * c->cx,
                2 * c->cy,
                2 * c->cz);
    glDispatchCompute(
        (2 * c->cx + 3) / 4, (2 * c->cy + 3) / 4, (2 * c->cz + 3) / 4);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

static void run_lbm(void *a) {
    Ctx *c = a;
    LBM_Step(c->grid, 0.05f, 0.0f, 0.0f);
}

int main(int argc, char **argv) {
    // usage: bench TRISHADER nx ny nz reps [in.f32 out.f32]
    if (argc < 6)
        return 2;
    int cx = atoi(argv[2]), cy = atoi(argv[3]), cz = atoi(argv[4]),
        reps = atoi(argv[5]);
    GLContext *ctx = GLContext_CreateHeadless(64, 64);
    if (!ctx)
        return 1;
    printf("renderer %s\n", glGetString(GL_RENDERER));
    size_t n = (size_t)cx * cy * cz;
    float *in = malloc(n * 4 * sizeof(float));
    if (argc > 6) {
        FILE *f = fopen(argv[6], "rb");
        if (!f || fread(in, sizeof(float), n * 4, f) != n * 4)
            return 3;
        fclose(f);
    } else {
        srand(1);
        for (size_t i = 0; i < n; i++) {
            in[4 * i] = 0.05f * rand() / RAND_MAX;
            in[4 * i + 1] = 0.01f * rand() / RAND_MAX;
            in[4 * i + 2] = 0.01f * rand() / RAND_MAX;
            in[4 * i + 3] = 1.0f;
        }
    }
    Ctx c = {0};
    c.cx = cx;
    c.cy = cy;
    c.cz = cz;
    glGenBuffers(1, &c.coarse);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, c.coarse);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * 16, in, GL_DYNAMIC_COPY);
    glGenBuffers(1, &c.fine);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, c.fine);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * 8 * 16, NULL, GL_DYNAMIC_COPY);
    c.sr = SR_Create(
        cx, cy, cz, "assets/sr_model.bin", "assets/sr_model_norm.bin");
    c.prog = createComputeShader(argv[1]);
    if (!c.sr || !c.prog)
        return 4;
    GLuint q;
    glGenQueries(1, &q);
    double tsr = time_dispatch(q, run_sr, &c, reps);
    double ttri = time_dispatch(q, run_tri, &c, reps);
    c.grid = getenv("NOLBM") ? NULL : LBM_Create(2 * cx, 2 * cy, 2 * cz, 0.01f);
    double tlbm = c.grid ? time_dispatch(q, run_lbm, &c, reps) : -1.0;
    printf("RESULT coarse=%dx%dx%d sr_ms=%.4f trilinear_ms=%.4f "
           "lbm_step_fine_ms=%.4f\n",
           cx,
           cy,
           cz,
           tsr,
           ttri,
           tlbm);
    if (argc > 7) {
        run_sr(&c);
        glFinish();
        float *out = malloc(n * 8 * 16);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, SR_GetFineVelocityBuffer(c.sr));
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * 8 * 16, out);
        FILE *f = fopen(argv[7], "wb");
        fwrite(out, 16, n * 8, f);
        fclose(f);
        run_tri(&c);
        glFinish();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, c.fine);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * 8 * 16, out);
        char p[600];
        snprintf(p, sizeof p, "%s.tri", argv[7]);
        f = fopen(p, "wb");
        fwrite(out, 16, n * 8, f);
        fclose(f);
    }
    return 0;
}
