#include "../lib/opengl_utils.h"
#include <stdio.h>
#include <stdlib.h>

// Load and compile a shader from a file
GLuint loadShader(const char* path, GLenum type) {
  FILE* file = fopen(path, "rb");
  if (!file) {
    printf("Error: Could not open the shader file %s \n", path);
    return 0;
  }
  fseek(file, 0, SEEK_END);
  long length = ftell(file);
  fseek(file, 0, SEEK_SET);
  char* source = (char*)malloc(length + 1);
  fread(source, 1, length, file);
  source[length] = '\0';
  fclose(file);

  // Load and compile shader
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, (const GLchar * const *)&source, NULL);
  glCompileShader(shader);

  // Check for compilation errors
  GLint status;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (status == GL_FALSE) {
    char log[1024];
    glGetShaderInfoLog(shader, sizeof(log), NULL, log);
    printf("Shader compilation error in %s: %s\n", path, log); // Add file path to error message
    free(source); // Free allocated memory
    glDeleteShader(shader); // Delete the shader object
    return 0;
  }
  free(source);
  return shader;
}

// Create a shader program for vertex and fragment shaders
GLuint createShaderProgram(const char* vertexPath, const char* fragmentPath) {
  GLuint vertexShader = loadShader(vertexPath, GL_VERTEX_SHADER);
  if (!vertexShader) return 0;

  GLuint fragmentShader = loadShader(fragmentPath, GL_FRAGMENT_SHADER);
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
    char log[1024];
    glGetProgramInfoLog(program, sizeof(log), NULL, log);
    printf("Shader linking error: %s\n", log);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteProgram(program);
    return 0;
  }

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);
  return program;
}

// Create a compute shader program
GLuint createComputeShader(const char* computePath) {
  GLuint computeShader = loadShader(computePath, GL_COMPUTE_SHADER);
  if (!computeShader) return 0;

  GLuint program = glCreateProgram();
  glAttachShader(program, computeShader);
  glLinkProgram(program);

  GLint status;
  glGetProgramiv(program, GL_LINK_STATUS, &status);
  if (status == GL_FALSE) {
    char log[1024];
    glGetProgramInfoLog(program, sizeof(log), NULL, log);
    printf("Shader linking error: %s\n", log);
    glDeleteShader(computeShader);
    glDeleteProgram(program);
    return 0;
  }

  glDeleteShader(computeShader);
  return program;
}

// Create a buffer (like SSBO, VBO)
GLuint createBuffer(GLenum type, GLsizeiptr size, const void* data, GLenum usage) {
  GLuint buffer;
  glGenBuffers(1, &buffer);
  glBindBuffer(type, buffer);
  glBufferData(type, size, data, usage);
  return buffer;
}