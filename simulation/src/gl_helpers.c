#include "../lib/gl_helpers.h"

#include <glad/gl.h>
#include <stdio.h>
#include <stdlib.h>

void checkGLError(const char *label) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        printf("OpenGL error at %s: 0x%x\n", label, err);
    }
}

void saveFrameToPPM(const char *filename, int width, int height) {
    unsigned char *pixels = (unsigned char *)malloc(width * height * 3);
    if (!pixels) {
        printf("Error: Could not allocate pixel buffer for %s\n", filename);
        return;
    }
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Could not open %s for writing\n", filename);
        free(pixels);
        return;
    }

    fprintf(f, "P6\n%d %d\n255\n", width, height);

    size_t row = (size_t)width * 3;
    int ok = 1;
    for (int y = height - 1; y >= 0 && ok; y--)
        ok = fwrite(pixels + (size_t)y * row, 1, row, f) == row;

    if (fclose(f) != 0 || !ok)
        printf("Error: Could not write %s\n", filename);
    free(pixels);
}
