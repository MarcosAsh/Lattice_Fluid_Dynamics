#include "../lib/manifest.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    FILE *f;
    int depth;
    int first;
} Json;

static void put_string(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p;
         p++) {
        if (*p == '"' || *p == '\\')
            fprintf(f, "\\%c", *p);
        else if (*p == '\n')
            fputs("\\n", f);
        else if (*p == '\r')
            fputs("\\r", f);
        else if (*p == '\t')
            fputs("\\t", f);
        else if (*p < 0x20)
            fprintf(f, "\\u%04x", *p);
        else
            fputc(*p, f);
    }
    fputc('"', f);
}

static void key(Json *j, const char *name) {
    fputs(j->first ? "\n" : ",\n", j->f);
    for (int i = 0; i < j->depth; i++)
        fputs("  ", j->f);
    put_string(j->f, name);
    fputs(": ", j->f);
    j->first = 0;
}

static void open_obj(Json *j, const char *name) {
    key(j, name);
    fputc('{', j->f);
    j->depth++;
    j->first = 1;
}

static void close_obj(Json *j) {
    j->depth--;
    fputc('\n', j->f);
    for (int i = 0; i < j->depth; i++)
        fputs("  ", j->f);
    fputc('}', j->f);
    j->first = 0;
}

static void str(Json *j, const char *name, const char *v) {
    key(j, name);
    put_string(j->f, v);
}

// Empty strings mean "not set"
static void opt_str(Json *j, const char *name, const char *v) {
    key(j, name);
    if (v && v[0])
        put_string(j->f, v);
    else
        fputs("null", j->f);
}

// Shortest form that reads back to the same float, else full double
static void put_number(FILE *f, double v) {
    if ((double)(float)v == v) {
        char buf[32];
        for (int prec = 6; prec <= 9; prec++) {
            snprintf(buf, sizeof(buf), "%.*g", prec, v);
            if (strtof(buf, NULL) == (float)v)
                break;
        }
        fputs(buf, f);
    } else {
        fprintf(f, "%.17g", v);
    }
}

static void num(Json *j, const char *name, double v) {
    key(j, name);
    if (isfinite(v))
        put_number(j->f, v);
    else
        fputs("null", j->f);
}

static void integer(Json *j, const char *name, long v) {
    key(j, name);
    fprintf(j->f, "%ld", v);
}

static void boolean(Json *j, const char *name, int v) {
    key(j, name);
    fputs(v ? "true" : "false", j->f);
}

static void grid(Json *j, const char *name, int x, int y, int z) {
    key(j, name);
    fprintf(j->f, "[%d, %d, %d]", x, y, z);
}

static void vec3(Json *j, const char *name, const double *v) {
    key(j, name);
    fputc('[', j->f);
    for (int i = 0; i < 3; i++) {
        if (i)
            fputs(", ", j->f);
        if (isfinite(v[i]))
            fprintf(j->f, "%.17g", v[i]);
        else
            fputs("null", j->f);
    }
    fputc(']', j->f);
}

void manifest_init(RunManifest *m, const CliOptions *opts) {
    *m = (RunManifest){0};
    m->opts = opts;
    m->wallSeconds = NAN;
    m->modelScale = m->viscosity = m->tau = m->latticeVelocity = NAN;
    m->reynolds = m->reynoldsEffective = m->charLength = NAN;
    m->refArea = m->blockageFactor = m->blockageRatio = m->emaAlpha = NAN;
    m->solidCells = -1;
    m->cd = m->cdCorr = m->cdEma = m->cdMean = m->cdCorrMean = NAN;
    m->cdPressure = m->cdFriction = m->cl = m->clMean = NAN;
    m->tStar = m->flowThroughs = m->strouhal = m->sheddingFreq = NAN;
    m->stopFrame = -1;
    m->restartStep = -1;
    m->checkpointStep = -1;
    conv_series_stats(NULL, 0, &m->conv);
}

static void write_parameters(Json *j, const RunManifest *m) {
    const CliOptions *o = m->opts;
    open_obj(j, "parameters");
    num(j, "wind", o->windSpeed);
    integer(j, "viz", o->visualizationMode);
    integer(j, "collision", o->collisionMode);
    integer(j, "duration", o->renderDuration);
    opt_str(j, "output", o->outputPath);
    str(j, "model", o->modelPath);
    integer(j, "angle", o->slantAngle);
    num(j, "scale", m->modelScale);
    boolean(j, "scale_from_cli", o->scaleFromCLI);
    num(j, "reynolds", o->reynoldsNumber);
    grid(j, "grid", o->gridX, o->gridY, o->gridZ);
    num(j, "smagorinsky", o->smagorinskyCs);
    str(j, "collision_operator", cli_operator_name(o->collisionOperator));
    num(j, "turbulence", o->turbulenceIntensity);
    integer(j, "sponge", m->spongeCells);
    opt_str(j, "vtk_output", o->vtkOutputPath);
    integer(j, "vtk_interval", o->vtkInterval);
    boolean(j, "vtk_stats", o->vtkStats);
    boolean(j, "superres", o->useSuperRes);
    str(j, "sr_weights", o->srWeightsPath);
    str(j, "sr_norm", o->srNormPath);
    opt_str(j, "manifest", o->manifestPath);
    num(j, "converge", o->convergeTol > 0.0f ? o->convergeTol : NAN);
    integer(j, "converge_window", o->convergeWindow);
    integer(j, "max_duration", o->maxDuration);
    opt_str(j, "checkpoint", o->checkpointPath);
    integer(j, "checkpoint_interval", o->checkpointInterval);
    opt_str(j, "restart", o->restartPath);
    opt_str(j, "surface_output", o->surfacePath);
    close_obj(j);
}

static void write_body(Json *j, const RunManifest *m) {
    const BuildInfo *b = cli_build_info();
    const CliOptions *o = m->opts;

    integer(j, "schema_version", MANIFEST_SCHEMA_VERSION);
    str(j, "program", "3d_fluid_simulation_car");
    str(j, "version", b->version);
    open_obj(j, "git");
    str(j, "describe", b->gitDescribe);
    str(j, "commit", b->gitCommit);
    boolean(j, "dirty", b->gitDirty);
    close_obj(j);
    str(j, "start_time_utc", m->startTimeUtc);
    num(j, "wall_seconds", m->wallSeconds);

    open_obj(j, "gl");
    opt_str(j, "vendor", m->glVendor);
    opt_str(j, "renderer", m->glRenderer);
    opt_str(j, "version", m->glVersion);
    close_obj(j);

    write_parameters(j, m);

    open_obj(j, "lattice");
    grid(j, "grid", o->gridX, o->gridY, o->gridZ);
    opt_str(j, "collision_operator", m->collisionOperator);
    boolean(j, "smagorinsky", m->smagorinsky);
    boolean(j, "periodic_yz", m->periodicYZ);
    num(j, "viscosity", m->viscosity);
    num(j, "tau", m->tau);
    num(j, "velocity", m->latticeVelocity);
    // Lattice speed of sound is 1/sqrt(3)
    num(j, "mach", m->latticeVelocity * sqrt(3.0));
    num(j, "reynolds", m->reynolds);
    num(j, "reynolds_effective", m->reynoldsEffective);
    num(j, "char_length", m->charLength);
    close_obj(j);

    open_obj(j, "model");
    opt_str(j, "path", m->modelPath);
    integer(j, "triangles", m->triangles);
    key(j, "solid_cells");
    if (m->solidCells >= 0)
        fprintf(j->f, "%ld", m->solidCells);
    else
        fputs("null", j->f);
    num(j, "ref_area", m->refArea);
    num(j, "blockage_factor", m->blockageFactor);
    num(j, "blockage_ratio", m->blockageRatio);
    close_obj(j);

    open_obj(j, "run");
    integer(j, "frames", m->frames);
    integer(j, "output_frames", m->outputFrames);
    integer(j, "substeps_per_frame", m->substeps);
    integer(j, "lbm_steps", m->lbmSteps);
    close_obj(j);

    if (m->restartStep >= 0) {
        open_obj(j, "restarted_from");
        str(j, "path", o->restartPath);
        str(j, "commit", m->restartCommit);
        boolean(j, "dirty", m->restartDirty);
        integer(j, "step", m->restartStep);
        integer(j, "frame", m->restartFrame);
        // Cd samples and their statistics carry over, --vtk-stats does not
        str(j, "drag_statistics", "continued");
        str(j, "vtk_statistics", "restarted");
        close_obj(j);
    } else {
        key(j, "restarted_from");
        fputs("null", j->f);
    }
    if (o->checkpointPath[0]) {
        open_obj(j, "checkpoint");
        str(j, "path", o->checkpointPath);
        integer(j, "interval_frames", o->checkpointInterval);
        integer(j, "written", m->checkpointsWritten);
        key(j, "last_step");
        if (m->checkpointStep >= 0)
            fprintf(j->f, "%ld", m->checkpointStep);
        else
            fputs("null", j->f);
        close_obj(j);
    } else {
        key(j, "checkpoint");
        fputs("null", j->f);
    }

    if (m->surfaceSteps > 0) {
        open_obj(j, "surface");
        str(j, "path", o->surfacePath);
        integer(j, "steps", m->surfaceSteps);
        integer(j, "windows", m->surfaceWindows);
        integer(j, "boundary_cells", m->surfaceCells);
        integer(j, "triangles", m->surfaceTriangles);
        integer(j, "unmapped_triangles", m->surfaceUnmapped);
        integer(j, "reference_plane_x", 1);
        num(j, "rho_ref", m->surfaceRhoRef);
        // Sum over cells, integral over the mesh and the drag readback
        vec3(j, "force_cells", m->surfaceForceCells);
        vec3(j, "force_mesh", m->surfaceForceMesh);
        vec3(j, "force_drag", m->surfaceForceDrag);
        close_obj(j);
    } else {
        key(j, "surface");
        fputs("null", j->f);
    }

    open_obj(j, "results");
    integer(j, "cd_samples", m->cdSamples);
    integer(j, "sample_interval_frames", m->sampleIntervalFrames);
    integer(
        j, "force_window_steps", (long)m->sampleIntervalFrames * m->substeps);
    num(j, "ema_alpha", m->emaAlpha);
    num(j, "cd", m->cd);
    num(j, "cd_corr", m->cdCorr);
    num(j, "cd_ema", m->cdEma);
    num(j, "cd_mean", m->cdMean);
    num(j, "cd_corr_mean", m->cdCorrMean);
    num(j, "cd_pressure", m->cdPressure);
    num(j, "cd_friction", m->cdFriction);
    num(j, "cl", m->cl);
    num(j, "cl_mean", m->clMean);
    num(j, "t_star", m->tStar);
    num(j, "flow_throughs", m->flowThroughs);
    num(j, "strouhal", m->strouhal);
    num(j, "shedding_frequency", m->sheddingFreq);
    close_obj(j);

    // Counts are Cd samples, Cd values are dimensionless
    const ConvStats *c = &m->conv;
    open_obj(j, "convergence");
    str(j, "method", CONV_METHOD);
    integer(j, "method_version", CONV_METHOD_VERSION);
    boolean(j, "enabled", o->convergeTol > 0.0f);
    num(j, "tolerance", o->convergeTol > 0.0f ? o->convergeTol : NAN);
    integer(j, "window", o->convergeWindow);
    boolean(j, "converged", m->converged);
    key(j, "stop_frame");
    if (m->stopFrame >= 0)
        fprintf(j->f, "%d", m->stopFrame);
    else
        fputs("null", j->f);
    integer(j, "samples", c->samples);
    integer(j, "rejected_samples", c->rejected);
    integer(j, "truncated_samples", c->truncated);
    integer(j, "kept_samples", c->kept);
    key(j, "truncation_ok");
    fputs(!c->valid ? "null" : c->truncationOk ? "true" : "false", j->f);
    key(j, "stationary");
    fputs(!c->valid ? "null" : c->stationary ? "true" : "false", j->f);
    opt_str(j, "reason", c->reason);
    num(j, "cd_mean", c->mean);
    num(j, "cd_sd", c->sd);
    num(j, "cd_sem", c->sem);
    str(j, "sem_method", "lugsail_obm_sqrt_n");
    num(j, "cd_ci95_half_width", c->halfWidth);
    num(j, "relative_half_width", c->relHalfWidth);
    num(j, "effective_sample_size", c->ess);
    num(j, "batch_size", c->valid ? c->batchSize : NAN);
    num(j, "dof", c->dof);
    num(j, "lag1_autocorrelation", c->lag1);
    num(j, "cd_drift", c->drift);
    num(j, "relative_drift", c->relDrift);
    num(j, "stationarity_statistic", c->cvm);
    num(j, "stationarity_critical", CONV_CVM_95);
    integer(j, "min_kept_samples", CONV_MIN_KEPT);
    integer(j, "min_effective_sample_size", CONV_MIN_ESS);
    close_obj(j);
}

int manifest_write(const char *path, const RunManifest *m) {
    if (!path || !path[0] || !m || !m->opts)
        return -1;

    size_t len = strlen(path) + 5;
    char *tmp = (char *)malloc(len);
    if (!tmp)
        return -1;
    snprintf(tmp, len, "%s.tmp", path);

    FILE *f = fopen(tmp, "w");
    if (!f) {
        free(tmp);
        return -1;
    }

    Json j = {f, 1, 1};
    fputc('{', f);
    write_body(&j, m);
    fputs("\n}\n", f);

    int ok = !ferror(f);
    ok = (fclose(f) == 0) && ok;
    if (ok && rename(tmp, path) != 0)
        ok = 0;
    if (!ok)
        remove(tmp);
    free(tmp);
    return ok ? 0 : -1;
}
