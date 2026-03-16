#include "../lib/gl_context.h"
#include <glad/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- SDL path (interactive) ---- */

#include <SDL2/SDL.h>

GLContext *GLContext_CreateInteractive(int w, int h) {
    GLContext *ctx = (GLContext *)calloc(1, sizeof(*ctx));
    ctx->useEGL = 0;
    ctx->width = w;
    ctx->height = h;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL init failed: %s\n", SDL_GetError());
        free(ctx);
        return NULL;
    }

    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_PROFILE_MASK,
        SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window *win = SDL_CreateWindow(
        "Lattice - LBM Wind Tunnel",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        w, h,
        SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (!win) {
        printf("Window failed: %s\n", SDL_GetError());
        SDL_Quit();
        free(ctx);
        return NULL;
    }

    SDL_GLContext gl = SDL_GL_CreateContext(win);
    if (!gl) {
        printf("GL context failed: %s\n",
               SDL_GetError());
        SDL_DestroyWindow(win);
        SDL_Quit();
        free(ctx);
        return NULL;
    }
    SDL_GL_MakeCurrent(win, gl);
    SDL_GL_SetSwapInterval(1);

    int version = gladLoadGL(
        (GLADloadfunc)SDL_GL_GetProcAddress);
    if (!version) {
        printf("glad failed to load GL\n");
        SDL_GL_DeleteContext(gl);
        SDL_DestroyWindow(win);
        SDL_Quit();
        free(ctx);
        return NULL;
    }

    ctx->sdlWindow = win;
    ctx->sdlContext = gl;

    printf("GL context: SDL (interactive)\n");
    printf("  OpenGL: %s\n", glGetString(GL_VERSION));
    printf("  Renderer: %s\n",
           glGetString(GL_RENDERER));
    return ctx;
}

/* SDL hidden window fallback for headless */
static GLContext *headless_sdl_fallback(int w, int h) {
    printf("Using SDL hidden window for headless\n");

    /* Force software GL for Xvfb environments. */
    setenv("LIBGL_ALWAYS_SOFTWARE", "1", 1);
    setenv("MESA_GL_VERSION_OVERRIDE", "4.3", 0);

    GLContext *ctx = (GLContext *)calloc(1, sizeof(*ctx));
    ctx->useEGL = 0;
    ctx->width = w;
    ctx->height = h;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL init failed: %s\n", SDL_GetError());
        free(ctx);
        return NULL;
    }

    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_PROFILE_MASK,
        SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);

    SDL_Window *win = SDL_CreateWindow(
        "headless", 0, 0, w, h,
        SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    if (!win) {
        printf("Window failed: %s\n", SDL_GetError());
        SDL_Quit();
        free(ctx);
        return NULL;
    }

    SDL_GLContext gl = SDL_GL_CreateContext(win);
    if (!gl) {
        printf("GL context failed: %s\n",
               SDL_GetError());
        SDL_DestroyWindow(win);
        SDL_Quit();
        free(ctx);
        return NULL;
    }
    SDL_GL_MakeCurrent(win, gl);

    int version = gladLoadGL(
        (GLADloadfunc)SDL_GL_GetProcAddress);
    if (!version) {
        printf("glad failed to load GL\n");
        SDL_GL_DeleteContext(gl);
        SDL_DestroyWindow(win);
        SDL_Quit();
        free(ctx);
        return NULL;
    }

    ctx->sdlWindow = win;
    ctx->sdlContext = gl;

    printf("GL context: SDL hidden (headless)\n");
    printf("  OpenGL: %s\n", glGetString(GL_VERSION));
    printf("  Renderer: %s\n",
           glGetString(GL_RENDERER));
    return ctx;
}

/* ---- EGL path (headless GPU) ---- */

#ifdef HAVE_EGL
#include <EGL/egl.h>
#include <glad/egl.h>

/* EXT function types for device enumeration */
typedef EGLBoolean (*PFNEGLQUERYDEVICESEXTPROC)(
    EGLint, EGLDeviceEXT *, EGLint *);
typedef EGLDisplay (*PFNEGLGETPLATFORMDISPLAYEXTPROC)(
    EGLenum, void *, const EGLint *);

#ifndef EGL_PLATFORM_DEVICE_EXT
#define EGL_PLATFORM_DEVICE_EXT 0x313F
#endif

static EGLDisplay get_egl_device_display(void) {
    /* Try the EXT device enumeration API first.
     * This is the only way to get NVIDIA GPU on
     * headless systems without X. */
    PFNEGLQUERYDEVICESEXTPROC queryDevices =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress(
            "eglQueryDevicesEXT");
    PFNEGLGETPLATFORMDISPLAYEXTPROC getPlatformDisplay =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress(
            "eglGetPlatformDisplayEXT");

    if (queryDevices && getPlatformDisplay) {
        EGLDeviceEXT devices[8];
        EGLint numDevices = 0;
        if (queryDevices(8, devices, &numDevices) &&
            numDevices > 0) {
            printf("EGL: found %d device(s)\n",
                   numDevices);
            /* Use first device */
            EGLDisplay d = getPlatformDisplay(
                EGL_PLATFORM_DEVICE_EXT,
                devices[0], NULL);
            if (d != EGL_NO_DISPLAY)
                return d;
        }
    }

    /* Fall back to default display */
    return eglGetDisplay(EGL_DEFAULT_DISPLAY);
}

GLContext *GLContext_CreateHeadless(int w, int h) {
    /* Skip EGL if explicitly disabled. */
    const char *skip = getenv("USE_EGL");
    if (skip && skip[0] == '0') {
        return headless_sdl_fallback(w, h);
    }

    printf("Trying EGL headless...\n");

    EGLDisplay display = get_egl_device_display();
    if (display == EGL_NO_DISPLAY) {
        printf("EGL: no display, falling back\n");
        return headless_sdl_fallback(w, h);
    }

    EGLint major, minor;
    if (!eglInitialize(display, &major, &minor)) {
        printf("EGL: init failed, falling back\n");
        return headless_sdl_fallback(w, h);
    }
    printf("EGL %d.%d initialized\n", major, minor);

    /* Load full EGL API via glad */
    gladLoadEGL(display,
                (GLADloadfunc)eglGetProcAddress);

    eglBindAPI(EGL_OPENGL_API);

    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_NONE,
    };

    EGLConfig config;
    EGLint numConfigs;
    if (!eglChooseConfig(
            display, configAttribs, &config, 1,
            &numConfigs) ||
        numConfigs == 0) {
        printf("EGL: no config, falling back\n");
        eglTerminate(display);
        return headless_sdl_fallback(w, h);
    }

    EGLint ctxAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK,
        EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
        EGL_NONE,
    };

    EGLContext eglCtx = eglCreateContext(
        display, config, EGL_NO_CONTEXT, ctxAttribs);
    if (eglCtx == EGL_NO_CONTEXT) {
        printf("EGL: context failed (0x%x), "
               "falling back\n",
               eglGetError());
        eglTerminate(display);
        return headless_sdl_fallback(w, h);
    }

    EGLint surfAttribs[] = {
        EGL_WIDTH, w,
        EGL_HEIGHT, h,
        EGL_NONE,
    };
    EGLSurface surface = eglCreatePbufferSurface(
        display, config, surfAttribs);
    if (surface == EGL_NO_SURFACE) {
        printf("EGL: surface failed, falling back\n");
        eglDestroyContext(display, eglCtx);
        eglTerminate(display);
        return headless_sdl_fallback(w, h);
    }

    if (!eglMakeCurrent(
            display, surface, surface, eglCtx)) {
        printf("EGL: make current failed\n");
        eglDestroySurface(display, surface);
        eglDestroyContext(display, eglCtx);
        eglTerminate(display);
        return headless_sdl_fallback(w, h);
    }

    /* Load GL functions via EGL */
    int glver = gladLoadGL(
        (GLADloadfunc)eglGetProcAddress);
    if (!glver) {
        printf("EGL: glad GL load failed, "
               "falling back\n");
        eglDestroySurface(display, surface);
        eglDestroyContext(display, eglCtx);
        eglTerminate(display);
        return headless_sdl_fallback(w, h);
    }

    GLContext *ctx = (GLContext *)calloc(1, sizeof(*ctx));
    ctx->useEGL = 1;
    ctx->width = w;
    ctx->height = h;
    ctx->eglDisplay = display;
    ctx->eglContext = eglCtx;
    ctx->eglSurface = surface;

    printf("GL context: EGL (headless GPU)\n");
    printf("  OpenGL: %s\n", glGetString(GL_VERSION));
    printf("  Renderer: %s\n",
           glGetString(GL_RENDERER));
    return ctx;
}

#else /* no EGL at compile time */

GLContext *GLContext_CreateHeadless(int w, int h) {
    return headless_sdl_fallback(w, h);
}
#endif

/* ---- Common ---- */

void GLContext_SwapBuffers(GLContext *ctx) {
    if (!ctx || ctx->useEGL)
        return;
    if (ctx->sdlWindow)
        SDL_GL_SwapWindow((SDL_Window *)ctx->sdlWindow);
}

void GLContext_Destroy(GLContext *ctx) {
    if (!ctx)
        return;

#ifdef HAVE_EGL
    if (ctx->useEGL) {
        EGLDisplay d = (EGLDisplay)ctx->eglDisplay;
        eglDestroySurface(d, (EGLSurface)ctx->eglSurface);
        eglDestroyContext(d, (EGLContext)ctx->eglContext);
        eglTerminate(d);
        free(ctx);
        return;
    }
#endif

    if (ctx->sdlContext)
        SDL_GL_DeleteContext(
            (SDL_GLContext)ctx->sdlContext);
    if (ctx->sdlWindow)
        SDL_DestroyWindow(
            (SDL_Window *)ctx->sdlWindow);
    SDL_Quit();
    free(ctx);
}
