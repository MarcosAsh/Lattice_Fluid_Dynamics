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

// Full info log of a shader or program, NULL if empty
static char *info_log(GLuint obj, int isProgram) {
    GLint len = 0;
    if (isProgram)
        glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &len);
    else
        glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &len);
    if (len <= 1)
        return NULL;
    char *log = (char *)malloc((size_t)len);
    if (!log)
        return NULL;
    if (isProgram)
        glGetProgramInfoLog(obj, len, NULL, log);
    else
        glGetShaderInfoLog(obj, len, NULL, log);
    return log;
}

static void print_warnings(const char *what, const char *path, char *log) {
    if (log)
        printf("Shader %s log for %s:\n%s\n", what, path, log);
    free(log);
}

static GLuint loadShader(const char *path, GLenum type) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        printf("Error: shader file not found: %s\n", path);
        return 0;
    }
    long length = -1;
    if (fseek(file, 0, SEEK_END) == 0)
        length = ftell(file);
    if (fseek(file, 0, SEEK_SET) != 0)
        length = -1;

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
    size_t got = fread(source, 1, (size_t)length, file);
    fclose(file);
    if (got != (size_t)length) {
        printf("Error: short read on shader file: %s\n", path);
        free(source);
        return 0;
    }
    source[length] = '\0';

    GLuint shader = glCreateShader(type);
    glShaderSource(
        shader, 1, (const GLchar *const *)&source, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        char *log = info_log(shader, 0);
        printf("\n=== Shader compile error ===\n");
        printf("  File: %s\n", path);
        printf("  Type: %s\n", shader_type_name(type));
        printf("  Error: %s\n", log ? log : "(no log)");
        free(log);
        printf("  Source preview:\n");
        print_source_preview(source, 15);
        printf("============================\n\n");
        free(source);
        glDeleteShader(shader);
        return 0;
    }
    free(source);
    print_warnings("compile", path, info_log(shader, 0));
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
        char *log = info_log(program, 1);
        printf("\n=== Shader link error ===\n");
        printf("  Vertex:   %s\n", vertexPath);
        printf("  Fragment: %s\n", fragmentPath);
        printf("  Error: %s\n", log ? log : "(no log)");
        free(log);
        printf("=========================\n\n");
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        glDeleteProgram(program);
        return 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    print_warnings("link", vertexPath, info_log(program, 1));
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
        char *log = info_log(program, 1);
        printf("\n=== Compute shader link error ===\n");
        printf("  File:  %s\n", computePath);
        printf("  Error: %s\n", log ? log : "(no log)");
        free(log);
        printf("=================================\n\n");
        glDeleteShader(computeShader);
        glDeleteProgram(program);
        return 0;
    }

    glDeleteShader(computeShader);
    print_warnings("link", computePath, info_log(program, 1));
    return program;
}
