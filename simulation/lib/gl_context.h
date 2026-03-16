#ifndef GL_CONTEXT_H
#define GL_CONTEXT_H

#include <glad/gl.h>

typedef struct {
    int useEGL;
    int width;
    int height;

    /* SDL fields (interactive mode) */
    void *sdlWindow;
    void *sdlContext;

    /* EGL fields (headless mode) */
    void *eglDisplay;
    void *eglContext;
    void *eglSurface;
} GLContext;

/* Create an interactive window with SDL + GLX. */
GLContext *GLContext_CreateInteractive(int w, int h);

/* Create a headless pbuffer with EGL (for GPU compute). */
GLContext *GLContext_CreateHeadless(int w, int h);

/* Swap buffers (no-op for headless). */
void GLContext_SwapBuffers(GLContext *ctx);

/* Clean up. */
void GLContext_Destroy(GLContext *ctx);

#endif
