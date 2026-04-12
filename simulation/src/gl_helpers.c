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

    for (int y = height - 1; y >= 0; y--) {
        fwrite(pixels + y * width * 3, 1, width * 3, f);
    }

    fclose(f);
    free(pixels);
}
