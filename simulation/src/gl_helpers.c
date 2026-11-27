#include "../lib/gl_helpers.h"

#include <glad/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void checkGLError(const char *label) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        printf("OpenGL error at %s: 0x%x\n", label, err);
    }
}

int glDebugRequested(void) {
    const char *env = getenv("LATTICE_GL_DEBUG");
    if (env && env[0])
        return strcmp(env, "0") != 0;
#ifdef NDEBUG
    return 0;
#else
    return 1;
#endif
}

static const char *debug_type_name(GLenum type) {
    switch (type) {
    case GL_DEBUG_TYPE_ERROR:
        return "error";
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        return "deprecated";
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        return "undefined";
    case GL_DEBUG_TYPE_PORTABILITY:
        return "portability";
    case GL_DEBUG_TYPE_PERFORMANCE:
        return "performance";
    default:
        return "other";
    }
}

static const char *debug_severity_name(GLenum severity) {
    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        return "high";
    case GL_DEBUG_SEVERITY_MEDIUM:
        return "medium";
    case GL_DEBUG_SEVERITY_LOW:
        return "low";
    default:
        return "info";
    }
}

static void GLAD_API_PTR gl_debug_callback(GLenum source,
                                           GLenum type,
                                           GLuint id,
                                           GLenum severity,
                                           GLsizei length,
                                           const GLchar *message,
                                           const void *user) {
    (void)source;
    (void)length;
    (void)user;
    fprintf(stderr,
            "GL debug [%s, %s, id %u]: %s\n",
            debug_type_name(type),
            debug_severity_name(severity),
            id,
            message);
}

void enableGLDebugOutput(void) {
    if (!glDebugRequested())
        return;
    if (!GLAD_GL_VERSION_4_3 && !GLAD_GL_KHR_debug) {
        fprintf(stderr, "GL debug output not supported\n");
        return;
    }
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(gl_debug_callback, NULL);
    glDebugMessageControl(GL_DONT_CARE,
                          GL_DONT_CARE,
                          GL_DEBUG_SEVERITY_NOTIFICATION,
                          0,
                          NULL,
                          GL_FALSE);
    printf("GL debug output enabled\n");
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
