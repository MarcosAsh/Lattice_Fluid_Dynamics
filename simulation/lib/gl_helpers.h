#ifndef GL_HELPERS_H
#define GL_HELPERS_H

// Small GL-side helpers used throughout the simulation.
// Kept separate from opengl_utils.h (shader + buffer builders).
void checkGLError(const char *label);
void saveFrameToPPM(const char *filename, int width, int height);

#endif // GL_HELPERS_H
