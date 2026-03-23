#include "../lib/opengl_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *shader_type_name(GLenum type) {
    switch (type) {
    case GL_VERTEX_SHADER:
        return "vertex";
    case GL_FRAGMENT_SHADER:
        return "fragment";
    case GL_COMPUTE_SHADER:
        return "compute";
    default:
        return "unknown";
    }
}

// Print the first N lines of source for debugging
static void print_source_preview(const char *source, int lines) {
    const char *p = source;
    for (int i = 0; i < lines && *p; i++) {
        const char *end = strchr(p, '\n');
        if (end) {
            printf("  %3d | %.*s\n", i + 1, (int)(end - p), p);
            p = end + 1;
        } else {
            printf("  %3d | %s\n", i + 1, p);
            break;
        }
    }
}

GLuint loadShader(const char *path, GLenum type) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        printf("Error: shader file not found: %s\n", path);
        return 0;
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (length <= 0) {
        printf("Error: shader file is empty: %s\n", path);
        fclose(file);
        return 0;
    }

    char *source = (char *)malloc(length + 1);
    if (!source) {
        printf("Error: out of memory loading shader: %s\n", path);
        fclose(file);
        return 0;
    }
    fread(source, 1, length, file);
    source[length] = '\0';
    fclose(file);

    GLuint shader = glCreateShader(type);
    glShaderSource(
        shader, 1, (const GLchar *const *)&source, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        char log[2048];
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        printf("\n=== Shader compile error ===\n");
        printf("  File: %s\n", path);
        printf("  Type: %s\n", shader_type_name(type));
        printf("  Error: %s\n", log);
        printf("  Source preview:\n");
        print_source_preview(source, 15);
        printf("============================\n\n");
        free(source);
        glDeleteShader(shader);
        return 0;
    }
    free(source);
    return shader;
}

GLuint createShaderProgram(
    const char *vertexPath, const char *fragmentPath) {
    GLuint vertexShader =
        loadShader(vertexPath, GL_VERTEX_SHADER);
    if (!vertexShader)
        return 0;

    GLuint fragmentShader =
        loadShader(fragmentPath, GL_FRAGMENT_SHADER);
    if (!fragmentShader) {
        glDeleteShader(vertexShader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        char log[2048];
        glGetProgramInfoLog(
            program, sizeof(log), NULL, log);
        printf("\n=== Shader link error ===\n");
        printf("  Vertex:   %s\n", vertexPath);
        printf("  Fragment: %s\n", fragmentPath);
        printf("  Error: %s\n", log);
        printf("=========================\n\n");
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        glDeleteProgram(program);
        return 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

GLuint createComputeShader(const char *computePath) {
    GLuint computeShader =
        loadShader(computePath, GL_COMPUTE_SHADER);
    if (!computeShader)
        return 0;

    GLuint program = glCreateProgram();
    glAttachShader(program, computeShader);
    glLinkProgram(program);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        char log[2048];
        glGetProgramInfoLog(
            program, sizeof(log), NULL, log);
        printf("\n=== Compute shader link error ===\n");
        printf("  File:  %s\n", computePath);
        printf("  Error: %s\n", log);
        printf("=================================\n\n");
        glDeleteShader(computeShader);
        glDeleteProgram(program);
        return 0;
    }

    glDeleteShader(computeShader);
    return program;
}

GLuint createBuffer(
    GLenum type,
    GLsizeiptr size,
    const void *data,
    GLenum usage) {
    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(type, buffer);
    glBufferData(type, size, data, usage);
    return buffer;
}
