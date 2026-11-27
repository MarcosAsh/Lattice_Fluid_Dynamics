#ifndef GL_HELPERS_H
#define GL_HELPERS_H

// Small GL-side helpers used throughout the simulation.
// Kept separate from opengl_utils.h (shader + buffer builders).
void checkGLError(const char *label);

// On in debug builds, LATTICE_GL_DEBUG=0/1 overrides
int glDebugRequested(void);
// Route KHR_debug messages to stderr if requested and supported
void enableGLDebugOutput(void);
void saveFrameToPPM(const char *filename, int width, int height);

#endif // GL_HELPERS_H
