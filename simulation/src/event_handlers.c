#include "../lib/event_handlers.h"

#include <math.h>
#include <stdio.h>

// Camera / input globals are owned by main.c.
extern float cameraAngleY;
extern float cameraAngleX;
extern float cameraDistance;
extern float cameraTargetX;
extern float cameraTargetY;
extern float cameraTargetZ;

extern int mouseDown;
extern int middleMouseDown;
extern int lastMouseX;
extern int lastMouseY;

extern float maxSpeed;
extern int useLBM;
extern int paused;
extern int stepOnce;

extern const char *vizModeNames[];
extern const int numVizModes;

void handle_sdl_event(const SDL_Event *event,
                      int *running,
                      int *collisionMode,
                      int *visualizationMode,
                      float *windSpeed,
                      int numTriangles,
                      int frameCount) {
    if (event->type == SDL_QUIT) {
        *running = 0;
    } else if (event->type == SDL_MOUSEBUTTONDOWN) {
        if (event->button.button == SDL_BUTTON_LEFT) {
            mouseDown = 1;
            lastMouseX = event->button.x;
            lastMouseY = event->button.y;
        } else if (event->button.button == SDL_BUTTON_MIDDLE) {
            middleMouseDown = 1;
            lastMouseX = event->button.x;
            lastMouseY = event->button.y;
        }
    } else if (event->type == SDL_MOUSEBUTTONUP) {
        if (event->button.button == SDL_BUTTON_LEFT) {
            mouseDown = 0;
        } else if (event->button.button == SDL_BUTTON_MIDDLE) {
            middleMouseDown = 0;
        }
    } else if (event->type == SDL_MOUSEMOTION) {
        int dx = event->motion.x - lastMouseX;
        int dy = event->motion.y - lastMouseY;
        if (mouseDown && (SDL_GetModState() & KMOD_SHIFT)) {
            // Shift+left-drag: pan
            float panScale = cameraDistance * 0.002f;
            cameraTargetX -= dx * panScale * cosf(cameraAngleY);
            cameraTargetZ += dx * panScale * sinf(cameraAngleY);
            cameraTargetY += dy * panScale;
            lastMouseX = event->motion.x;
            lastMouseY = event->motion.y;
        } else if (mouseDown) {
            cameraAngleY += dx * 0.005f;
            cameraAngleX += dy * 0.005f;

            if (cameraAngleX > 1.5f)
                cameraAngleX = 1.5f;
            if (cameraAngleX < -1.5f)
                cameraAngleX = -1.5f;

            lastMouseX = event->motion.x;
            lastMouseY = event->motion.y;
        } else if (middleMouseDown) {
            // Middle-click drag: pan
            float panScale = cameraDistance * 0.002f;
            cameraTargetX -= dx * panScale * cosf(cameraAngleY);
            cameraTargetZ += dx * panScale * sinf(cameraAngleY);
            cameraTargetY += dy * panScale;
            lastMouseX = event->motion.x;
            lastMouseY = event->motion.y;
        }
    } else if (event->type == SDL_MOUSEWHEEL) {
        cameraDistance -= event->wheel.y * 0.5f;
        if (cameraDistance < 1.0f)
            cameraDistance = 1.0f;
        if (cameraDistance > 20.0f)
            cameraDistance = 20.0f;
    } else if (event->type == SDL_KEYDOWN) {
        switch (event->key.keysym.sym) {
        case SDLK_ESCAPE:
            *running = 0;
            break;
        case SDLK_0:
            *collisionMode = 0;
            printf("Collision: OFF\n");
            break;
        case SDLK_1:
            *collisionMode = 1;
            printf("Collision: AABB (fast)\n");
            break;
        case SDLK_2:
            *collisionMode = 2;
            printf(
                "Collision: Per-Triangle (accurate) - %d triangles\n",
                numTriangles);
            break;
        case SDLK_3:
            *visualizationMode = 0;
            printf("Visualization: %s\n",
                   vizModeNames[*visualizationMode]);
            break;
        case SDLK_4:
            *visualizationMode = 1;
            printf("Visualization: %s\n",
                   vizModeNames[*visualizationMode]);
            break;
        case SDLK_5:
            *visualizationMode = 2;
            printf("Visualization: %s\n",
                   vizModeNames[*visualizationMode]);
            break;
        case SDLK_6:
            *visualizationMode = 3;
            printf("Visualization: %s\n",
                   vizModeNames[*visualizationMode]);
            break;
        case SDLK_7:
            *visualizationMode = 4;
            printf("Visualization: %s\n",
                   vizModeNames[*visualizationMode]);
            break;
        case SDLK_8:
            *visualizationMode = 5;
            printf("Visualization: %s\n",
                   vizModeNames[*visualizationMode]);
            break;
        case SDLK_9:
            *visualizationMode = 6;
            printf("Visualization: %s\n",
                   vizModeNames[*visualizationMode]);
            break;
        case SDLK_v:
            *visualizationMode = (*visualizationMode + 1) % numVizModes;
            printf("Visualization: %s\n",
                   vizModeNames[*visualizationMode]);
            break;
        case SDLK_UP:
            *windSpeed += 0.5f;
            if (*windSpeed > 5.0f)
                *windSpeed = 5.0f;
            printf("Wind speed: %.1f\n", *windSpeed);
            break;
        case SDLK_DOWN:
            *windSpeed -= 0.5f;
            if (*windSpeed < 0.0f)
                *windSpeed = 0.0f;
            printf("Wind speed: %.1f\n", *windSpeed);
            break;
        case SDLK_LEFT:
            maxSpeed -= 0.2f;
            if (maxSpeed < 0.2f)
                maxSpeed = 0.2f;
            printf("Max speed scale: %.1f\n", maxSpeed);
            break;
        case SDLK_RIGHT:
            maxSpeed += 0.2f;
            if (maxSpeed > 10.0f)
                maxSpeed = 10.0f;
            printf("Max speed scale: %.1f\n", maxSpeed);
            break;
        case SDLK_a:
            cameraAngleY -= 0.1f;
            break;
        case SDLK_d:
            cameraAngleY += 0.1f;
            break;
        case SDLK_w:
            cameraAngleX -= 0.1f;
            if (cameraAngleX < -1.5f)
                cameraAngleX = -1.5f;
            break;
        case SDLK_s:
            cameraAngleX += 0.1f;
            if (cameraAngleX > 1.5f)
                cameraAngleX = 1.5f;
            break;
        case SDLK_q:
            cameraDistance -= 0.5f;
            if (cameraDistance < 1.0f)
                cameraDistance = 1.0f;
            break;
        case SDLK_e:
            cameraDistance += 0.5f;
            if (cameraDistance > 20.0f)
                cameraDistance = 20.0f;
            break;
        case SDLK_r:
            cameraAngleY = 0.0f;
            cameraAngleX = 0.3f;
            cameraDistance = 6.0f;
            cameraTargetX = -1.5f;
            cameraTargetY = 0.0f;
            cameraTargetZ = 0.0f;
            printf("Camera reset\n");
            break;
        case SDLK_F1:
            // Front view
            cameraAngleY = 0.0f;
            cameraAngleX = 0.0f;
            cameraDistance = 6.0f;
            printf("Camera: front\n");
            break;
        case SDLK_F2:
            // Side view
            cameraAngleY = 1.5708f;
            cameraAngleX = 0.0f;
            cameraDistance = 6.0f;
            printf("Camera: side\n");
            break;
        case SDLK_F3:
            // Top view
            cameraAngleY = 0.0f;
            cameraAngleX = 1.5f;
            cameraDistance = 8.0f;
            printf("Camera: top\n");
            break;
        case SDLK_F4:
            // Isometric view
            cameraAngleY = 0.6f;
            cameraAngleX = 0.5f;
            cameraDistance = 7.0f;
            printf("Camera: isometric\n");
            break;
        case SDLK_l:
            useLBM = !useLBM;
            printf("LBM: %s\n", useLBM ? "ON" : "OFF");
            break;
        case SDLK_SPACE:
            paused = !paused;
            printf("Simulation %s at frame %d\n",
                   paused ? "paused" : "resumed",
                   frameCount);
            break;
        case SDLK_PERIOD:
            if (paused)
                stepOnce = 1;
            break;
        }
    }
}
