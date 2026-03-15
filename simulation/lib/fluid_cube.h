#ifndef FLUID_CUBE_H
#define FLUID_CUBE_H

#include "../obj-file-loader/lib/model_loader.h"

struct FluidCube {
    int sizeX;
    int sizeY;
    int sizeZ;
    float dt;
    float diff;
    float visc;

    float *s;
    float *density;

    float *Vx;
    float *Vy;
    float *Vz;

    float *Vx0;
    float *Vy0;
    float *Vz0;

    Model *model; // Add a pointer to the Model
};
typedef struct FluidCube FluidCube;

FluidCube *FluidCubeCreate(int sizeX,
                           int sizeY,
                           int sizeZ,
                           float diffusion,
                           float viscosity,
                           float dt,
                           Model *model);
void FluidCubeFree(FluidCube *cube);
void FluidCubeAddDensity(FluidCube *cube, int x, int y, int z, float amount);
void FluidCubeAddVelocity(
    FluidCube *cube, int x, int y, int z, float amtX, float amtY, float amtZ);
void FluidCubeStep(FluidCube *cube);

void SDL_ExitWithError(const char *message);

static void
set_bnd(int b, float *x, int sizeX, int sizeY, int sizeZ, FluidCube *cube);
static void lin_solve(int b,
                      float *x,
                      float *x0,
                      float a,
                      float c,
                      int iter,
                      int sizeX,
                      int sizeY,
                      int sizeZ,
                      FluidCube *cube);
static void diffuse(int b,
                    float *x,
                    float *x0,
                    float diff,
                    float dt,
                    int iter,
                    int sizeX,
                    int sizeY,
                    int sizeZ,
                    FluidCube *cube);
static void project(float *velocX,
                    float *velocY,
                    float *velocZ,
                    float *p,
                    float *div,
                    int iter,
                    int sizeX,
                    int sizeY,
                    int sizeZ,
                    FluidCube *cube);
static void advect(int b,
                   float *d,
                   float *d0,
                   float *velocX,
                   float *velocY,
                   float *velocZ,
                   float dt,
                   int sizeX,
                   int sizeY,
                   int sizeZ,
                   FluidCube *cube);

#endif // FLUID_CUBE_H