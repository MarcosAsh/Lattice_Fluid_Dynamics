#include "../lib/vti_export.h"

#include <glad/gl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Write LBM velocity field as VTK ImageData (.vti) for ParaView.
// Binary appended format for compact output.
void writeVTI(LBMGrid *grid, const char *path, int step) {
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/field_%06d.vti", path, step);

    int nx = grid->sizeX;
    int ny = grid->sizeY;
    int nz = grid->sizeZ;
    int total = nx * ny * nz;

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

    // Appended-data offsets:
    //   velocity: 0
    //   rho:      sizeof(uint64_t) + 3*total*float32
    //   solid:    2*sizeof(uint64_t) + 4*total*float32
    size_t rhoOffset =
        sizeof(uint64_t) + (size_t)total * 3 * sizeof(float);
    size_t solidOffset =
        rhoOffset + sizeof(uint64_t) + (size_t)total * sizeof(float);

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
        " format=\"appended\" offset=\"%lu\"/>\n"
        "      </PointData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "  <AppendedData encoding=\"raw\">\n_",
        nx,
        ny,
        nz,
        nx,
        ny,
        nz,
        (unsigned long)rhoOffset,
        (unsigned long)solidOffset);

    // Velocity array (strip w component from vec4)
    uint64_t velDataSize = (uint64_t)total * 3 * sizeof(float);
    fwrite(&velDataSize, sizeof(uint64_t), 1, f);
    for (int i = 0; i < total; i++) {
        fwrite(&vel[i * 4], sizeof(float), 3, f);
    }

    // Density (rho) array -- 4th component of the velocity vec4
    uint64_t rhoDataSize = (uint64_t)total * sizeof(float);
    fwrite(&rhoDataSize, sizeof(uint64_t), 1, f);
    for (int i = 0; i < total; i++) {
        fwrite(&vel[i * 4 + 3], sizeof(float), 1, f);
    }

    // Solid array
    uint64_t solidDataSize = (uint64_t)total * sizeof(int);
    fwrite(&solidDataSize, sizeof(uint64_t), 1, f);
    fwrite(solid, sizeof(int), total, f);

    fprintf(f, "\n  </AppendedData>\n</VTKFile>\n");
    fclose(f);
    free(vel);
    free(solid);
    printf("VTK: wrote %s\n", filename);
}
