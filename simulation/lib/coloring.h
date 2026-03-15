#ifndef COLORING_H
#define COLORING_H

#include <stdio.h>
#include <SDL2/SDL.h>
#include <math.h>
#include <stdlib.h>

#define PI                          3.14159265
#define IX(x, y, N)                 ((x) + (y) * N)
#define IX3D(x, y, z, sizeX, sizeY) ((x) + (y) * sizeX + (z) * sizeX * sizeY)

// Function declarations
void DensityToColor(float density, Uint8 *r, Uint8 *g, Uint8 *b);
void VelocityToColor(float velX,
                     float velY,
                     float velZ,
                     Uint8 *r,
                     Uint8 *g,
                     Uint8 *b,
                     int velThreshold);
void DensityAndVelocityToColor(float density,
                               float velX,
                               float velY,
                               float velZ,
                               Uint8 *r,
                               Uint8 *g,
                               Uint8 *b,
                               int velThreshold);

#endif // COLORING_H