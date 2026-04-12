#ifndef EVENT_HANDLERS_H
#define EVENT_HANDLERS_H

#include <SDL2/SDL.h>

// Handle one SDL event, updating the camera globals (extern'd from
// main.c) and the small set of loop-local variables the user can
// toggle from the keyboard. frameCount is only used for the pause
// printout, so it is passed by value rather than via pointer.
void handle_sdl_event(const SDL_Event *event,
                      int *running,
                      int *collisionMode,
                      int *visualizationMode,
                      float *windSpeed,
                      int numTriangles,
                      int frameCount);

#endif // EVENT_HANDLERS_H
