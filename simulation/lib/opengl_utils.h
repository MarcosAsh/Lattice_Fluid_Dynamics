#ifndef OPENGL_UTILS_H
#define OPENGL_UTILS_H

#include <glad/gl.h>

// Function declarations
GLuint createShaderProgram(const char *vertexPath, const char *fragmentPath);
GLuint createComputeShader(const char *computePath);
GLuint
createBuffer(GLenum type, GLsizeiptr size, const void *data, GLenum usage);

#endif // OPENGL_UTILS_H