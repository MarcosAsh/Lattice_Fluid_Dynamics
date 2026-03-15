#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include <SDL2/SDL.h>
#include "fluid_cube.h"
#include "config.h"

#define GRID_CELL_SIZE 10 // Size of each grid cell

// Particle structure
// The struct has padding to align with GPU memory layout
typedef struct {
    float x, y, z;    // Position (vec3)
    float padding1;   // Padding for 16-byte alignment
    float vx, vy, vz; // Velocity (vec3)
    float life;       // Lifetime (0.0 to 1.0)
} Particle;

// Grid cell structure for spatial partitioning
typedef struct {
    Particle particles[MAX_PARTICLES / GRID_CELL_SIZE];
    int count;
} GridCell;

// Particle system structure
typedef struct {
    Particle particles[MAX_PARTICLES];
    int numParticles;
    GridCell grid[GRID_CELL_SIZE][GRID_CELL_SIZE];
} ParticleSystem;

// Car collision bounds
typedef struct {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    float centerX, centerY, centerZ;
} CollisionBounds;

// Function Declarations
void ParticleSystem_Init(ParticleSystem *system);
void ParticleSystem_AddParticle(ParticleSystem *system,
                                float x,
                                float y,
                                float z,
                                float vx,
                                float vy,
                                float vz);
void ParticleSystem_Update(ParticleSystem *system, FluidCube *fluid, float dt);
void ParticleSystem_UpdateWithCollision(ParticleSystem *system,
                                        FluidCube *fluid,
                                        float dt,
                                        CollisionBounds *bounds);
void ParticleSystem_Render(ParticleSystem *system,
                           SDL_Renderer *renderer,
                           int scale);

// Collision detection helpers
int ParticleSystem_CheckCollision(float x,
                                  float y,
                                  float z,
                                  CollisionBounds *bounds);
void ParticleSystem_ResolveCollision(Particle *p, CollisionBounds *bounds);

#endif // PARTICLE_SYSTEM_H