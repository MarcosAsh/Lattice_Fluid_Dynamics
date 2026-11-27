/*
 * Validation driver for the GPU LBM solver.
 *
 * Runs canonical cases with known answers through the same shaders
 * and LBM_* API the wind tunnel uses, and prints one RESULT line per
 * case for validation/run_validation.py to parse.
 *
 * Run from simulation/ (shaders are loaded by relative path):
 *   ./build/lbm_validate tgv --n=64 --tau=0.6 --op=mrt
 *   ./build/lbm_validate sphere --d=16 --re=100
 */

#include "../lib/gl_context.h"
#include "../lib/lbm.h"
#include "../lib/model_bounds.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const int ex[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
static const int ey[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1};
static const int ez[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1};
static const double wq[19] = {1.0 / 3,
                              1.0 / 18,
                              1.0 / 18,
                              1.0 / 18,
                              1.0 / 18,
                              1.0 / 18,
                              1.0 / 18,
                              1.0 / 36,
                              1.0 / 36,
                              1.0 / 36,
                              1.0 / 36,
                              1.0 / 36,
                              1.0 / 36,
                              1.0 / 36,
                              1.0 / 36,
                              1.0 / 36,
                              1.0 / 36,
                              1.0 / 36,
                              1.0 / 36};

typedef struct {
    const char *name;
    int nx, ny, nz;
    int n;
    double tau;
    double nu;
    double re;
    double u;
    double d;
    double q;
    double smag;
    double up, down, lat;
    int steps;
    int window;
    int sponge;
    int maxSlabZ;
    int periodicYZ;
    int plane; // 0 = xy, 1 = xz
    int mesh;  // 1 = sphere through the OBJ mesh path
    int ramp;  // inlet ramp steps, flow starts at rest when > 0
    char model[512];
    char op[16];
} Opts;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double feq(int i, double rho, double ux, double uy, double uz) {
    double eu = ex[i] * ux + ey[i] * uy + ez[i] * uz;
    double u2 = ux * ux + uy * uy + uz * uz;
    return wq[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
}

static int argval(const char *a, const char *key, const char **val) {
    size_t k = strlen(key);
    if (strncmp(a, key, k) == 0 && a[k] == '=') {
        *val = a + k + 1;
        return 1;
    }
    return 0;
}

static void parse(int argc, char **argv, Opts *o) {
    for (int i = 2; i < argc; i++) {
        const char *v;
        if (argval(argv[i], "--grid", &v))
            sscanf(v, "%dx%dx%d", &o->nx, &o->ny, &o->nz);
        else if (argval(argv[i], "--n", &v))
            o->n = atoi(v);
        else if (argval(argv[i], "--tau", &v))
            o->tau = atof(v);
        else if (argval(argv[i], "--re", &v))
            o->re = atof(v);
        else if (argval(argv[i], "--u", &v))
            o->u = atof(v);
        else if (argval(argv[i], "--d", &v))
            o->d = atof(v);
        else if (argval(argv[i], "--q", &v))
            o->q = atof(v);
        else if (argval(argv[i], "--smag", &v))
            o->smag = atof(v);
        else if (argval(argv[i], "--up", &v))
            o->up = atof(v);
        else if (argval(argv[i], "--down", &v))
            o->down = atof(v);
        else if (argval(argv[i], "--lat", &v))
            o->lat = atof(v);
        else if (argval(argv[i], "--steps", &v))
            o->steps = atoi(v);
        else if (argval(argv[i], "--window", &v))
            o->window = atoi(v);
        else if (argval(argv[i], "--sponge", &v))
            o->sponge = atoi(v);
        else if (argval(argv[i], "--max-slab-z", &v))
            o->maxSlabZ = atoi(v);
        else if (argval(argv[i], "--periodic-yz", &v))
            o->periodicYZ = atoi(v);
        else if (argval(argv[i], "--plane", &v))
            o->plane = strcmp(v, "xz") == 0;
        else if (argval(argv[i], "--mesh", &v))
            o->mesh = atoi(v);
        else if (argval(argv[i], "--ramp", &v))
            o->ramp = atoi(v);
        else if (argval(argv[i], "--model", &v))
            snprintf(o->model, sizeof(o->model), "%s", v);
        else if (argval(argv[i], "--op", &v))
            snprintf(o->op, sizeof(o->op), "%s", v);
        else {
            fprintf(stderr, "unknown option %s\n", argv[i]);
            exit(2);
        }
    }
}

static LBMGrid *make_grid(const Opts *o, int nx, int ny, int nz, double tau) {
    LBMGrid *g = LBM_Create(nx, ny, nz, (float)((tau - 0.5) / 3.0));
    if (!g)
        return NULL;
    // LBM_Create clamps tau to [0.52, 2]
    g->tau = (float)tau;
    g->useMRT = strcmp(o->op, "mrt") == 0;
    g->useRegularized = strcmp(o->op, "reg") == 0;
    g->useEntropic = strcmp(o->op, "ent") == 0;
    g->useSmagorinsky = o->smag > 0;
    g->smagorinskyCs = (float)o->smag;
    g->periodicYZ = o->periodicYZ;
    if (o->maxSlabZ > 0 && o->maxSlabZ < nz) {
        g->slabZ = o->maxSlabZ;
        g->numChunks = (nz + o->maxSlabZ - 1) / o->maxSlabZ;
        printf("forced slab split: %d chunks of %d\n", g->numChunks, g->slabZ);
    }
    return g;
}

static void upload_f(LBMGrid *g, const float *f) {
    size_t bytes = (size_t)19 * g->totalCells * sizeof(float);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->fBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, f);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->fNewBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, f);
}

static void read_buf(GLuint buf, size_t bytes, void *dst) {
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, dst);
}

static double total_mass(LBMGrid *g, const int *solid) {
    size_t n = (size_t)19 * g->totalCells;
    float *f = malloc(n * sizeof(float));
    read_buf(g->fBuffer, n * sizeof(float), f);
    double m = 0;
    for (int c = 0; c < g->totalCells; c++) {
        if (solid && solid[c])
            continue;
        for (int i = 0; i < 19; i++)
            m += f[(size_t)c * 19 + i];
    }
    free(f);
    return m;
}

static int check_diag(LBMGrid *g, const char *where) {
    int resets = 0, nans = 0;
    LBM_ReadDiagnostics(g, &resets, &nans);
    if (resets > 0)
        printf("UNSTABLE %s: %d cell resets (%d NaN)\n", where, resets, nans);
    return resets;
}

static void print_gpu(void) {
    printf("GPU: %s | %s\n",
           (const char *)glGetString(GL_RENDERER),
           (const char *)glGetString(GL_VERSION));
    GLint64 ssbo = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &ssbo);
    printf("max SSBO block: %.0f MB\n", ssbo / 1048576.0);
}

/* Uniform flow without a body, open tunnel and closed periodic box */
static int case_uniform(Opts *o) {
    int nx = o->nx ? o->nx : 64, ny = o->ny ? o->ny : 32,
        nz = o->nz ? o->nz : 32;
    double tau = o->tau > 0 ? o->tau : 0.52;
    double U = o->u > 0 ? o->u : 0.05;
    int steps = o->steps ? o->steps : 4000;
    int worst = 0;

    for (int periodic = 0; periodic < 2; periodic++) {
        LBMGrid *g = make_grid(o, nx, ny, nz, tau);
        if (!g)
            return 1;
        g->periodicX = periodic;
        g->spongeCells = periodic ? 0 : (o->sponge < 0 ? nx / 8 : o->sponge);
        LBM_SetForceAveraging(g, 1);
        LBM_InitializeFlow(g, (float)U, 0, 0);
        double m0 = total_mass(g, NULL);
        double t0 = now_s();
        for (int s = 0; s < steps; s++)
            LBM_Step(g, (float)U, 0, 0);
        glFinish();
        double dt = now_s() - t0;
        int resets = check_diag(g, "uniform");
        double m1 = total_mass(g, NULL);
        float fx, fy, fz;
        LBM_ComputeDragForce(g, &fx, &fy, &fz);

        float *vel = malloc((size_t)4 * g->totalCells * sizeof(float));
        read_buf(
            g->velocityBuffer, (size_t)4 * g->totalCells * sizeof(float), vel);
        double du = 0, dv = 0, drho = 0;
        for (int c = 0; c < g->totalCells; c++) {
            du = fmax(du, fabs(vel[4 * c] - U));
            dv = fmax(dv, fmax(fabs(vel[4 * c + 1]), fabs(vel[4 * c + 2])));
            drho = fmax(drho, fabs(vel[4 * c + 3] - 1.0));
        }
        free(vel);
        printf("RESULT case=uniform bc=%s op=%s grid=%dx%dx%d tau=%.4f U=%.4f "
               "steps=%d mass_rel_drift=%.3e max_du=%.3e max_dv=%.3e "
               "max_drho=%.3e fx=%.3e resets=%d mlups=%.1f\n",
               periodic ? "periodic" : "open",
               o->op,
               nx,
               ny,
               nz,
               tau,
               U,
               steps,
               (m1 - m0) / m0,
               du,
               dv,
               drho,
               fx,
               resets,
               (double)g->totalCells * steps / dt * 1e-6);
        worst += resets;
        LBM_Free(g);
    }
    return worst > 0;
}

/* Taylor-Green vortex, E decays as exp(-4 nu k^2 t) */
static int case_tgv(Opts *o) {
    int n = o->n ? o->n : 64;
    int nz = 8;
    double tau = o->tau > 0 ? o->tau : 0.6;
    double U0 = o->u > 0 ? o->u : 0.02;
    double nu = (tau - 0.5) / 3.0;
    double k = 2.0 * M_PI / n;
    int nx = n, ny = o->plane ? nz : n, nzz = o->plane ? n : nz;
    // Run to E/E0 = exp(-1) unless told otherwise
    int steps = o->steps ? o->steps : (int)(1.0 / (4.0 * nu * k * k));

    LBMGrid *g = make_grid(o, nx, ny, nzz, tau);
    if (!g)
        return 1;
    g->periodicX = 1;
    g->periodicYZ = 1;
    g->spongeCells = 0;

    size_t cells = g->totalCells;
    float *f = malloc(cells * 19 * sizeof(float));
    for (int z = 0; z < nzz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) {
                double a = k * (x + 0.5);
                double b = k * ((o->plane ? z : y) + 0.5);
                double u = U0 * sin(a) * cos(b);
                double v = -U0 * cos(a) * sin(b);
                double p = 0.25 * U0 * U0 * (cos(2 * a) + cos(2 * b));
                double rho = 1.0 + 3.0 * p;
                size_t c = x + (size_t)nx * (y + (size_t)ny * z);
                for (int i = 0; i < 19; i++)
                    f[c * 19 + i] = (float)(o->plane ? feq(i, rho, u, 0, v)
                                                     : feq(i, rho, u, v, 0));
            }
    upload_f(g, f);
    double m0 = total_mass(g, NULL);

    // Sample energy at ~40 points, fit log E over the second half
    int nsamp = 40;
    int every = steps / nsamp > 0 ? steps / nsamp : 1;
    double ts[64], lnE[64];
    int ns = 0;
    float *vel = malloc(cells * 4 * sizeof(float));
    double t0 = now_s();
    double l2 = 0;
    for (int s = 1; s <= steps; s++) {
        LBM_Step(g, 0, 0, 0);
        if (s % every == 0 && ns < 64) {
            read_buf(g->velocityBuffer, cells * 4 * sizeof(float), vel);
            double E = 0;
            for (size_t c = 0; c < cells; c++)
                E += vel[4 * c] * vel[4 * c] + vel[4 * c + 1] * vel[4 * c + 1] +
                     vel[4 * c + 2] * vel[4 * c + 2];
            // velocity buffer holds the pre-collision state of step s-1
            ts[ns] = s - 1;
            lnE[ns] = log(E);
            ns++;
        }
    }
    glFinish();
    double dt = now_s() - t0;
    int resets = check_diag(g, "tgv");

    // L2 error against the analytic field at the last velocity sample
    read_buf(g->velocityBuffer, cells * 4 * sizeof(float), vel);
    double decay = exp(-2.0 * nu * k * k * (steps - 1));
    double num = 0, den = 0;
    for (int z = 0; z < nzz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) {
                double a = k * (x + 0.5);
                double b = k * ((o->plane ? z : y) + 0.5);
                double u = U0 * sin(a) * cos(b) * decay;
                double v = -U0 * cos(a) * sin(b) * decay;
                size_t c = x + (size_t)nx * (y + (size_t)ny * z);
                double gu = vel[4 * c];
                double gv = o->plane ? vel[4 * c + 2] : vel[4 * c + 1];
                num += (gu - u) * (gu - u) + (gv - v) * (gv - v);
                den += u * u + v * v;
            }
    l2 = sqrt(num / den);

    // Least squares slope over the second half, past the initial layer
    int a0 = ns / 2;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int m = 0;
    for (int i = a0; i < ns; i++) {
        sx += ts[i];
        sy += lnE[i];
        sxx += ts[i] * ts[i];
        sxy += ts[i] * lnE[i];
        m++;
    }
    double slope = (m * sxy - sx * sy) / (m * sxx - sx * sx);
    double nuEff = -slope / (4.0 * k * k);

    // Smagorinsky nu_t weighted by S:S is 1.441 Cs^2 k A for this field
    double meanA = 0;
    for (int i = a0; i < ns; i++)
        meanA += sqrt(2.0 * exp(lnE[i]) / (double)cells) / m;
    double nuT = 1.441 * o->smag * o->smag * k * meanA;
    double m1 = total_mass(g, NULL);
    printf("RESULT case=tgv op=%s smag=%.3f n=%d plane=%s tau=%.4f U0=%.4f "
           "steps=%d nu=%.6e nu_eff=%.6e nu_err=%.4e nu_t_pred=%.4e l2=%.4e "
           "mass_rel_drift=%.3e chunks=%d resets=%d mlups=%.1f\n",
           o->op,
           o->smag,
           n,
           o->plane ? "xz" : "xy",
           tau,
           U0,
           steps,
           nu,
           nuEff,
           (nuEff - nu) / nu,
           nuT,
           l2,
           (m1 - m0) / m0,
           g->numChunks,
           resets,
           (double)cells * steps / dt * 1e-6);
    free(vel);
    free(f);
    LBM_Free(g);
    return resets > 0;
}

/* Body-force channel, walls in y at distance q from the last fluid node */
static int case_poiseuille(Opts *o) {
    int h = o->n ? o->n : 16; // fluid nodes across
    double q = o->q > 0 ? o->q : 0.5;
    double tau = o->tau > 0 ? o->tau : 0.8;
    double nu = (tau - 0.5) / 3.0;
    int nx = 4, nz = 4, ny = h + 2;
    double width = (h - 1) + 2.0 * q;
    double umax = o->u > 0 ? o->u : 0.02;
    double F = 8.0 * nu * umax / (width * width);
    int steps = o->steps ? o->steps : (int)(8.0 * width * width / nu);

    LBMGrid *g = make_grid(o, nx, ny, nz, tau);
    if (!g)
        return 1;
    g->periodicX = 1;
    g->periodicYZ = 1;
    g->spongeCells = 0;
    g->bodyForce[0] = (float)F;
    LBM_SetForceAveraging(g, 1);

    size_t cells = g->totalCells;
    int *solid = calloc(cells, sizeof(int));
    float *qv = malloc(cells * 19 * sizeof(float));
    for (size_t i = 0; i < cells * 19; i++)
        qv[i] = -1.0f;
    for (int z = 0; z < nz; z++)
        for (int x = 0; x < nx; x++) {
            solid[x + nx * (0 + ny * z)] = 1;
            solid[x + nx * ((ny - 1) + ny * z)] = 1;
        }
    for (int z = 0; z < nz; z++)
        for (int y = 1; y < ny - 1; y++)
            for (int x = 0; x < nx; x++) {
                size_t c = x + nx * (y + (size_t)ny * z);
                for (int i = 1; i < 19; i++) {
                    int yn = y + ey[i];
                    if (yn == 0 || yn == ny - 1)
                        qv[c * 19 + i] = (float)q;
                }
            }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->solidBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, cells * sizeof(int), solid);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->qBuffer);
    glBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 0, cells * 19 * sizeof(float), qv);
    LBM_InitializeFlow(g, 0, 0, 0);

    double t0 = now_s();
    for (int s = 0; s < steps - 1000; s++)
        LBM_Step(g, 0, 0, 0);
    float fx, fy, fz;
    LBM_ComputeDragForce(g, &fx, &fy, &fz); // flush the window
    for (int s = 0; s < 1000; s++)
        LBM_Step(g, 0, 0, 0);
    LBM_ComputeDragForce(g, &fx, &fy, &fz);
    glFinish();
    double dt = now_s() - t0;
    int resets = check_diag(g, "poiseuille");

    float *vel = malloc(cells * 4 * sizeof(float));
    read_buf(g->velocityBuffer, cells * 4 * sizeof(float), vel);
    double num = 0, den = 0, umeas = 0;
    double ylo = 1.0 - q; // wall position in node coordinates
    for (int y = 1; y < ny - 1; y++) {
        double s = 0;
        for (int z = 0; z < nz; z++)
            for (int x = 0; x < nx; x++)
                s += vel[4 * (x + nx * (y + (size_t)ny * z))];
        s /= nx * nz;
        double eta = y - ylo;
        double ua = F / (2.0 * nu) * eta * (width - eta);
        num += (s - ua) * (s - ua);
        den += ua * ua;
        if (getenv("LBM_VALIDATE_PROFILE"))
            printf("PROFILE y=%d u=%.8e exact=%.8e rel=%.4e\n",
                   y,
                   s,
                   ua,
                   (s - ua) / umax);
        umeas = fmax(umeas, s);
    }
    double fluidCells = (double)nx * nz * h;
    double fBody = F * fluidCells;
    printf("RESULT case=poiseuille op=%s h=%d q=%.3f tau=%.4f steps=%d "
           "umax=%.6e umax_meas=%.6e l2=%.4e wall_fx=%.6e body_fx=%.6e "
           "force_err=%.4e resets=%d mlups=%.1f\n",
           o->op,
           h,
           q,
           tau,
           steps,
           umax,
           umeas,
           sqrt(num / den),
           fx,
           fBody,
           (fx - fBody) / fBody,
           resets,
           (double)cells * steps / dt * 1e-6);
    free(vel);
    free(solid);
    free(qv);
    LBM_Free(g);
    return resets > 0;
}

/* Icosphere in the 12-float triangle layout LBM_SetSolidMesh takes */
static float *icosphere(double cx, double r, int levels, int *ntri) {
    double t = (1.0 + sqrt(5.0)) / 2.0;
    double v0[12][3] = {{-1, t, 0},
                        {1, t, 0},
                        {-1, -t, 0},
                        {1, -t, 0},
                        {0, -1, t},
                        {0, 1, t},
                        {0, -1, -t},
                        {0, 1, -t},
                        {t, 0, -1},
                        {t, 0, 1},
                        {-t, 0, -1},
                        {-t, 0, 1}};
    int f0[20][3] = {{0, 11, 5},  {0, 5, 1},  {0, 1, 7},  {0, 7, 10},
                     {0, 10, 11}, {1, 5, 9},  {5, 11, 4}, {11, 10, 2},
                     {10, 7, 6},  {7, 1, 8},  {3, 9, 4},  {3, 4, 2},
                     {3, 2, 6},   {3, 6, 8},  {3, 8, 9},  {4, 9, 5},
                     {2, 4, 11},  {6, 2, 10}, {8, 6, 7},  {9, 8, 1}};
    int n = 20;
    for (int l = 0; l < levels; l++)
        n *= 4;
    double (*tri)[3][3] = malloc(sizeof(double[3][3]) * n);
    for (int i = 0; i < 20; i++)
        for (int k = 0; k < 3; k++)
            for (int c = 0; c < 3; c++) {
                double *p = v0[f0[i][k]];
                double len = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
                tri[i][k][c] = p[c] / len;
            }
    int cur = 20;
    for (int l = 0; l < levels; l++) {
        for (int i = cur - 1; i >= 0; i--) {
            double a[3], b[3], c[3], m[3][3];
            memcpy(a, tri[i][0], sizeof(a));
            memcpy(b, tri[i][1], sizeof(b));
            memcpy(c, tri[i][2], sizeof(c));
            for (int d = 0; d < 3; d++) {
                m[0][d] = a[d] + b[d];
                m[1][d] = b[d] + c[d];
                m[2][d] = c[d] + a[d];
            }
            for (int k = 0; k < 3; k++) {
                double len = sqrt(m[k][0] * m[k][0] + m[k][1] * m[k][1] +
                                  m[k][2] * m[k][2]);
                for (int d = 0; d < 3; d++)
                    m[k][d] /= len;
            }
            double out[4][3][3];
            memcpy(out[0][0], a, sizeof(a));
            memcpy(out[0][1], m[0], sizeof(a));
            memcpy(out[0][2], m[2], sizeof(a));
            memcpy(out[1][0], m[0], sizeof(a));
            memcpy(out[1][1], b, sizeof(a));
            memcpy(out[1][2], m[1], sizeof(a));
            memcpy(out[2][0], m[2], sizeof(a));
            memcpy(out[2][1], m[1], sizeof(a));
            memcpy(out[2][2], c, sizeof(a));
            memcpy(out[3][0], m[0], sizeof(a));
            memcpy(out[3][1], m[1], sizeof(a));
            memcpy(out[3][2], m[2], sizeof(a));
            memcpy(tri[4 * i], out, sizeof(out));
        }
        cur *= 4;
    }
    float *buf = calloc((size_t)n * 12, sizeof(float));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++) {
            buf[i * 12 + k * 4 + 0] = (float)(cx + r * tri[i][k][0]);
            buf[i * 12 + k * 4 + 1] = (float)(r * tri[i][k][1]);
            buf[i * 12 + k * 4 + 2] = (float)(r * tri[i][k][2]);
        }
    free(tri);
    *ntri = n;
    return buf;
}

/* Bouzidi q from the mesh path against the analytic sphere */
static int case_qcheck(Opts *o) {
    double D = o->d > 0 ? o->d : 16;
    int nx = (int)lround(16 * D), ny = nx / 2, nz = ny;
    double scale = nx / 8.0;
    double cx = 6 * D / scale - 4.0, r = 0.5 * D / scale;
    LBMGrid *g = make_grid(o, nx, ny, nz, 0.6);
    if (!g)
        return 1;
    size_t n = (size_t)g->totalCells * 19;
    float *qa = malloc(n * sizeof(float)), *qm = malloc(n * sizeof(float));
    LBM_SetSolidSphere(g, (float)cx, 0, 0, (float)r);
    read_buf(g->qBuffer, n * sizeof(float), qa);
    int ntri;
    float *tris = icosphere(cx, r, 5, &ntri);
    LBM_SetSolidMesh(g,
                     tris,
                     ntri,
                     (float)(cx - r),
                     (float)-r,
                     (float)-r,
                     (float)(cx + r),
                     (float)r,
                     (float)r);
    read_buf(g->qBuffer, n * sizeof(float), qm);
    float area = LBM_MeshFrontalArea(g, tris, ntri);
    float vox = LBM_ComputeProjectedArea(g, 0);
    long both = 0, onlyA = 0, onlyM = 0, half = 0;
    double sumd = 0, maxd = 0;
    for (size_t i = 0; i < n; i++) {
        int a = qa[i] >= 0, m = qm[i] >= 0;
        if (a && m) {
            double d = fabs(qa[i] - qm[i]);
            both++;
            sumd += d;
            maxd = fmax(maxd, d);
            if (qm[i] == 0.5f && fabs(qa[i] - 0.5) > 0.05)
                half++;
        } else if (a)
            onlyA++;
        else if (m)
            onlyM++;
    }
    double exact = M_PI * D * D / 4.0;
    printf("RESULT case=qcheck D=%.1f tris=%d links=%ld only_analytic=%ld "
           "only_mesh=%ld mean_dq=%.4f max_dq=%.4f stuck_at_half=%ld "
           "mesh_area=%.2f voxel_area=%.1f exact_area=%.2f\n",
           D,
           ntri,
           both,
           onlyA,
           onlyM,
           sumd / both,
           maxd,
           half,
           area,
           vox,
           exact);
    free(tris);
    free(qa);
    free(qm);
    LBM_Free(g);
    return 0;
}

/* Sphere in the production tunnel BCs with analytic Bouzidi q */
static int case_sphere(Opts *o) {
    double D = o->d > 0 ? o->d : 16;
    double re = o->re > 0 ? o->re : 100;
    double U = o->u > 0 ? o->u : 0.05;
    double up = o->up > 0 ? o->up : 6;
    double down = o->down > 0 ? o->down : 10;
    double lat = o->lat > 0 ? o->lat : 8;
    int nx = (int)lround((up + down) * D), ny = (int)lround(lat * D), nz = ny;
    // Box is [-4,4]x[-2,2]x[-2,2] world units, cells must stay cubic
    if (nx != 2 * ny) {
        fprintf(stderr, "sphere: need up+down = 2*lat for cubic cells\n");
        return 2;
    }
    double nu = U * D / re;
    double tau = 0.5 + 3.0 * nu;
    double scale = nx / 8.0;
    int window = o->window ? o->window : (int)lround(D / U); // one D/U
    int maxSteps = o->steps ? o->steps : (int)lround(60 * D / U);

    LBMGrid *g = make_grid(o, nx, ny, nz, tau);
    if (!g)
        return 1;
    g->periodicYZ = o->periodicYZ;
    g->spongeCells = o->sponge < 0 ? nx / 8 : o->sponge;
    LBM_SetForceAveraging(g, 1);

    // Centre sits on a cell corner
    double cx = up * D / scale - 4.0;
    if (o->mesh) {
        int ntri;
        float r = (float)(0.5 * D / scale);
        float *tris = icosphere(cx, r, 5, &ntri);
        LBM_SetSolidMesh(
            g, tris, ntri, (float)cx - r, -r, -r, (float)cx + r, r, r);
        free(tris);
    } else {
        LBM_SetSolidSphere(g, (float)cx, 0.0f, 0.0f, (float)(0.5 * D / scale));
    }
    float area = LBM_ComputeProjectedArea(g, 0);
    double areaExact = M_PI * D * D / 4.0;
    double blockage = areaExact / ((double)ny * nz);
    LBM_InitializeFlow(g, (float)U, 0, 0);

    printf("sphere: grid %dx%dx%d D=%.1f Re=%.1f U=%.4f tau=%.5f "
           "proj_area=%.1f exact=%.1f blockage=%.4f window=%d\n",
           nx,
           ny,
           nz,
           D,
           re,
           U,
           tau,
           area,
           areaExact,
           blockage,
           window);

    double dynA = 0.5 * U * U * areaExact;
    double hist[4096];
    int nw = 0, resets = 0;
    double t0 = now_s();
    int s = 0;
    float fx, fy, fz;
    LBM_ComputeDragForce(g, &fx, &fy, &fz);
    while (s < maxSteps) {
        for (int i = 0; i < window; i++, s++)
            LBM_Step(g, (float)U, 0, 0);
        LBM_ComputeDragForce(g, &fx, &fy, &fz);
        resets += check_diag(g, "sphere");
        double cd = fx / dynA;
        if (nw < 4096)
            hist[nw++] = cd;
        printf("WIN step=%d tstar=%.2f cd=%.6f cl_y=%.6f cl_z=%.6f\n",
               s,
               s * U / D,
               cd,
               fy / dynA,
               fz / dynA);
        fflush(stdout);
        if (isnan(cd) || resets > 0)
            break;
        // Stop when the last 5 windows agree to 0.05%
        if (nw >= 10 && s * U / D > 20) {
            double mn = 1e30, mx = -1e30;
            for (int k = nw - 5; k < nw; k++) {
                mn = fmin(mn, hist[k]);
                mx = fmax(mx, hist[k]);
            }
            if ((mx - mn) / fabs(hist[nw - 1]) < 5e-4)
                break;
        }
    }
    glFinish();
    double dt = now_s() - t0;

    // Mean and spread over the last quarter of windows
    int k0 = nw - (nw / 4 > 5 ? nw / 4 : 5);
    if (k0 < 0)
        k0 = 0;
    double mean = 0, sd = 0;
    for (int k = k0; k < nw; k++)
        mean += hist[k];
    mean /= (nw - k0);
    for (int k = k0; k < nw; k++)
        sd += (hist[k] - mean) * (hist[k] - mean);
    sd = sqrt(sd / (nw - k0));
    double cdVox = mean * areaExact / area;
    printf("RESULT case=sphere path=%s op=%s smag=%.3f D=%.1f re=%.1f U=%.4f "
           "tau=%.5f grid=%dx%dx%d sponge=%d periodic_yz=%d blockage=%.4f "
           "steps=%d tstar=%.1f cd=%.5f cd_sd=%.5f cd_voxel_area=%.5f "
           "proj_area=%.1f resets=%d seconds=%.1f mlups=%.1f\n",
           o->mesh ? "mesh" : "analytic",
           o->op,
           o->smag,
           D,
           re,
           U,
           tau,
           nx,
           ny,
           nz,
           g->spongeCells,
           g->periodicYZ,
           blockage,
           s,
           s * U / D,
           mean,
           sd,
           cdVox,
           area,
           resets,
           dt,
           (double)g->totalCells * s / dt * 1e-6);
    LBM_Free(g);
    return resets > 0;
}

/* OBJ body placed as main.c places it */
static int case_obj(Opts *o) {
    int nx = o->nx ? o->nx : 128, ny = o->ny ? o->ny : 64,
        nz = o->nz ? o->nz : 64;
    double U = o->u > 0 ? o->u : 0.05;
    Model model = loadOBJ(o->model);
    if (model.vertexCount == 0)
        return 2;
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = 0; i < model.vertexCount; i++) {
        float v[3] = {
            model.vertices[i].x, model.vertices[i].y, model.vertices[i].z};
        for (int c = 0; c < 3; c++) {
            mn[c] = fminf(mn[c], v[c]);
            mx[c] = fmaxf(mx[c], v[c]);
        }
    }
    float maxDim = fmaxf(mx[0] - mn[0], fmaxf(mx[1] - mn[1], mx[2] - mn[2]));
    float sc = 2.4f / maxDim;
    float off[3];
    for (int c = 0; c < 3; c++)
        off[c] = -0.5f * (mn[c] + mx[c]) * sc;
    float rot = (mx[2] - mn[2] > mx[0] - mn[0]) ? 90.0f : 0.0f;
    float rad = rot * (float)M_PI / 180.0f;
    off[0] += -2.0f * cosf(rad);
    off[2] += 2.0f * sinf(rad);
    CarBounds b = computeModelBounds(&model, sc, off[0], off[1], off[2], rot);
    int ntri = 0;
    GPUTriangle *tris =
        createTriangleBuffer(&model, sc, off[0], off[1], off[2], rot, &ntri);

    double L = (b.maxX - b.minX) * nx / 8.0;
    double re = o->re > 0 ? o->re : 200;
    double tau = o->tau > 0 ? o->tau : 0.5 + 3.0 * U * L / re;
    re = U * L / ((tau - 0.5) / 3.0);
    LBMGrid *g = make_grid(o, nx, ny, nz, tau);
    if (!g)
        return 1;
    g->spongeCells = o->sponge < 0 ? nx / 8 : o->sponge;
    LBM_SetForceAveraging(g, 1);
    LBM_SetSolidMesh(
        g, (float *)tris, ntri, b.minX, b.minY, b.minZ, b.maxX, b.maxY, b.maxZ);
    float areaVox = LBM_ComputeProjectedArea(g, 0);
    float areaMesh = LBM_MeshFrontalArea(g, (float *)tris, ntri);
    LBM_InitializeFlow(g, o->ramp > 0 ? 0.0f : (float)U, 0, 0);
    printf("obj: %s grid %dx%dx%d L=%.1f Re=%.1f tau=%.5f U=%.4f "
           "area_mesh=%.2f area_voxel=%.1f ramp=%d\n",
           o->model,
           nx,
           ny,
           nz,
           L,
           re,
           tau,
           U,
           areaMesh,
           areaVox,
           o->ramp);

    int window = o->window ? o->window : 20;
    int steps = o->steps ? o->steps : (int)(8 * nx / U);
    double dyn = 0.5 * U * U * areaMesh;
    float *vel = malloc((size_t)4 * g->totalCells * sizeof(float));
    int resets = 0, s = 0;
    float fx, fy, fz;
    double t0 = now_s();
    LBM_ComputeDragForce(g, &fx, &fy, &fz);
    while (s < steps) {
        for (int i = 0; i < window; i++, s++) {
            double r = o->ramp > 0 && s < o->ramp ? (double)s / o->ramp : 1.0;
            LBM_Step(g, (float)(U * r), 0, 0);
        }
        LBM_ComputeDragForce(g, &fx, &fy, &fz);
        resets += check_diag(g, "obj");
        // Density at the inlet and mid-domain centre line
        read_buf(
            g->velocityBuffer, (size_t)4 * g->totalCells * sizeof(float), vel);
        size_t row = (size_t)nx * (ny / 2 + (size_t)ny * (nz / 2));
        printf("WIN step=%d cd=%.6f cside=%.6f clift=%.6f rho_in=%.6f "
               "rho_mid=%.6f\n",
               s,
               fx / dyn,
               fy / dyn,
               fz / dyn,
               vel[4 * (row + 1) + 3],
               vel[4 * (row + nx / 2) + 3]);
        if (resets > 0)
            break;
    }
    glFinish();
    printf("RESULT case=obj model=%s grid=%dx%dx%d re=%.1f tau=%.5f steps=%d "
           "area_mesh=%.2f area_voxel=%.1f resets=%d seconds=%.1f\n",
           o->model,
           nx,
           ny,
           nz,
           re,
           tau,
           s,
           areaMesh,
           areaVox,
           resets,
           now_s() - t0);
    free(vel);
    free(tris);
    freeModel(&model);
    LBM_Free(g);
    return resets > 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s uniform|tgv|poiseuille|sphere [--key=value]\n",
                argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    Opts o;
    memset(&o, 0, sizeof(o));
    o.name = argv[1];
    o.sponge = -1;
    o.periodicYZ = 1;
    snprintf(o.op, sizeof(o.op), "mrt");
    parse(argc, argv, &o);

    GLContext *ctx = GLContext_CreateHeadless(64, 64);
    if (!ctx) {
        fprintf(stderr, "no GL context\n");
        return 1;
    }
    print_gpu();

    int rc;
    if (strcmp(o.name, "uniform") == 0)
        rc = case_uniform(&o);
    else if (strcmp(o.name, "tgv") == 0)
        rc = case_tgv(&o);
    else if (strcmp(o.name, "poiseuille") == 0)
        rc = case_poiseuille(&o);
    else if (strcmp(o.name, "sphere") == 0)
        rc = case_sphere(&o);
    else if (strcmp(o.name, "qcheck") == 0)
        rc = case_qcheck(&o);
    else if (strcmp(o.name, "obj") == 0)
        rc = case_obj(&o);
    else {
        fprintf(stderr, "unknown case %s\n", o.name);
        rc = 2;
    }
    GLContext_Destroy(ctx);
    return rc;
}
