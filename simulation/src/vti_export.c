#include "../lib/vti_export.h"

#include <glad/gl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

VTKStats *VTKStats_Create(int total) {
    VTKStats *s = (VTKStats *)calloc(1, sizeof(VTKStats));
    if (!s)
        return NULL;
    s->total = total;
    s->count = 0;
    s->sumVel = (float *)calloc((size_t)total * 3, sizeof(float));
    s->sumVelSq = (float *)calloc((size_t)total * 3, sizeof(float));
    if (!s->sumVel || !s->sumVelSq) {
        VTKStats_Free(s);
        return NULL;
    }
    return s;
}

void VTKStats_Free(VTKStats *s) {
    if (!s)
        return;
    free(s->sumVel);
    free(s->sumVelSq);
    free(s);
}

void VTKStats_Accumulate(VTKStats *s, LBMGrid *grid) {
    if (!s || !grid)
        return;
    int total = grid->sizeX * grid->sizeY * grid->sizeZ;
    if (total != s->total)
        return; // grid resized under us; skip rather than corrupt sums

    size_t velBytes = (size_t)total * 4 * sizeof(float);
    float *vel = (float *)malloc(velBytes);
    if (!vel)
        return;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->velocityBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, velBytes, vel);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    for (int i = 0; i < total; i++) {
        float u = vel[i * 4];
        float v = vel[i * 4 + 1];
        float w = vel[i * 4 + 2];
        s->sumVel[i * 3] += u;
        s->sumVel[i * 3 + 1] += v;
        s->sumVel[i * 3 + 2] += w;
        s->sumVelSq[i * 3] += u * u;
        s->sumVelSq[i * 3 + 1] += v * v;
        s->sumVelSq[i * 3 + 2] += w * w;
    }
    s->count++;
    free(vel);
}

// Write LBM velocity field as VTK ImageData (.vti) for ParaView.
// Binary appended format for compact output.
void writeVTI(LBMGrid *grid, VTKStats *stats, const char *path, int step) {
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/field_%06d.vti", path, step);

    int nx = grid->sizeX;
    int ny = grid->sizeY;
    int nz = grid->sizeZ;
    int total = nx * ny * nz;

    int hasStats = stats && stats->count > 0 && stats->total == total;

    // Read velocity (vec4) and solid (int) from GPU
    size_t velBytes = (size_t)total * 4 * sizeof(float);
    size_t solidBytes = (size_t)total * sizeof(int);
    float *vel = (float *)malloc(velBytes);
    int *solid = (int *)malloc(solidBytes);
    if (!vel || !solid) {
        free(vel);
        free(solid);
        return;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->velocityBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, velBytes, vel);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid->solidBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, solidBytes, solid);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    FILE *f = fopen(filename, "wb");
    if (!f) {
        free(vel);
        free(solid);
        return;
    }

    // Appended-data offsets. Each array is preceded by a UInt64 byte count.
    size_t f3 = (size_t)total * 3 * sizeof(float);
    size_t f1 = (size_t)total * sizeof(float);
    size_t i1 = (size_t)total * sizeof(int);
    size_t rhoOffset = sizeof(uint64_t) + f3;
    size_t solidOffset = rhoOffset + sizeof(uint64_t) + f1;
    // Stats arrays follow solid when enabled.
    size_t meanOffset = solidOffset + sizeof(uint64_t) + i1;
    size_t rmsOffset = meanOffset + sizeof(uint64_t) + f3;
    size_t tkeOffset = rmsOffset + sizeof(uint64_t) + f3;

    // VTI header
    fprintf(
        f,
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"ImageData\" version=\"1.0\""
        " byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
        "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\""
        " Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
        "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n"
        "      <PointData Vectors=\"velocity\" Scalars=\"rho\">\n"
        "        <DataArray type=\"Float32\" Name=\"velocity\""
        " NumberOfComponents=\"3\" format=\"appended\""
        " offset=\"0\"/>\n"
        "        <DataArray type=\"Float32\" Name=\"rho\""
        " format=\"appended\" offset=\"%lu\"/>\n"
        "        <DataArray type=\"Int32\" Name=\"solid\""
        " format=\"appended\" offset=\"%lu\"/>\n",
        nx, ny, nz, nx, ny, nz,
        (unsigned long)rhoOffset,
        (unsigned long)solidOffset);

    if (hasStats) {
        fprintf(
            f,
            "        <DataArray type=\"Float32\" Name=\"mean_velocity\""
            " NumberOfComponents=\"3\" format=\"appended\" offset=\"%lu\"/>\n"
            "        <DataArray type=\"Float32\" Name=\"rms_velocity\""
            " NumberOfComponents=\"3\" format=\"appended\" offset=\"%lu\"/>\n"
            "        <DataArray type=\"Float32\" Name=\"tke\""
            " format=\"appended\" offset=\"%lu\"/>\n",
            (unsigned long)meanOffset,
            (unsigned long)rmsOffset,
            (unsigned long)tkeOffset);
    }

    fprintf(f,
            "      </PointData>\n"
            "    </Piece>\n"
            "  </ImageData>\n"
            "  <AppendedData encoding=\"raw\">\n_");

    // Velocity array (strip w component from vec4)
    uint64_t velDataSize = (uint64_t)f3;
    int writeOk = 1;
    writeOk = writeOk && fwrite(&velDataSize, sizeof(uint64_t), 1, f) == 1;
    for (int i = 0; i < total && writeOk; i++) {
        writeOk = fwrite(&vel[i * 4], sizeof(float), 3, f) == 3;
    }

    // Density (rho) array -- 4th component of the velocity vec4
    uint64_t rhoDataSize = (uint64_t)f1;
    writeOk = writeOk && fwrite(&rhoDataSize, sizeof(uint64_t), 1, f) == 1;
    for (int i = 0; i < total && writeOk; i++) {
        writeOk = fwrite(&vel[i * 4 + 3], sizeof(float), 1, f) == 1;
    }

    // Solid array
    uint64_t solidDataSize = (uint64_t)i1;
    writeOk = writeOk && fwrite(&solidDataSize, sizeof(uint64_t), 1, f) == 1;
    writeOk = writeOk &&
              fwrite(solid, sizeof(int), total, f) == (size_t)total;

    if (hasStats) {
        float inv = 1.0f / (float)stats->count;

        // Mean velocity (time-averaged), 3 components per cell.
        uint64_t meanSize = (uint64_t)f3;
        writeOk = writeOk &&
                  fwrite(&meanSize, sizeof(uint64_t), 1, f) == 1;
        for (int i = 0; i < total && writeOk; i++) {
            float mean[3] = {
                stats->sumVel[i * 3] * inv,
                stats->sumVel[i * 3 + 1] * inv,
                stats->sumVel[i * 3 + 2] * inv,
            };
            writeOk = fwrite(mean, sizeof(float), 3, f) == 3;
        }

        // RMS velocity fluctuation per component: sqrt(<u'^2>).
        uint64_t rmsSize = (uint64_t)f3;
        writeOk = writeOk &&
                  fwrite(&rmsSize, sizeof(uint64_t), 1, f) == 1;
        for (int i = 0; i < total && writeOk; i++) {
            float rms[3];
            for (int c = 0; c < 3; c++) {
                float mean = stats->sumVel[i * 3 + c] * inv;
                float var = stats->sumVelSq[i * 3 + c] * inv - mean * mean;
                rms[c] = var > 0.0f ? sqrtf(var) : 0.0f;
            }
            writeOk = fwrite(rms, sizeof(float), 3, f) == 3;
        }

        // Turbulent kinetic energy: k = 0.5 * (u'u' + v'v' + w'w').
        uint64_t tkeSize = (uint64_t)f1;
        writeOk = writeOk &&
                  fwrite(&tkeSize, sizeof(uint64_t), 1, f) == 1;
        for (int i = 0; i < total && writeOk; i++) {
            float k = 0.0f;
            for (int c = 0; c < 3; c++) {
                float mean = stats->sumVel[i * 3 + c] * inv;
                float var = stats->sumVelSq[i * 3 + c] * inv - mean * mean;
                if (var > 0.0f)
                    k += var;
            }
            k *= 0.5f;
            writeOk = fwrite(&k, sizeof(float), 1, f) == 1;
        }
    }

    if (!writeOk) {
        fprintf(stderr, "VTK: write error for %s\n", filename);
        fclose(f);
        free(vel);
        free(solid);
        return;
    }

    fprintf(f, "\n  </AppendedData>\n</VTKFile>\n");
    fclose(f);
    free(vel);
    free(solid);
    printf("VTK: wrote %s%s\n", filename,
           hasStats ? " (with turbulence stats)" : "");
}
