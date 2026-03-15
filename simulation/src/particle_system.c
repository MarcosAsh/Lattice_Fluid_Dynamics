#include "../lib/particle_system.h"
#include "../lib/fluid_cube.h"
#include "../lib/config.h"

#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

// Initialize the particle system
void ParticleSystem_Init(ParticleSystem *system) {
    system->numParticles = 0;
    for (int i = 0; i < GRID_CELL_SIZE; i++) {
        for (int j = 0; j < GRID_CELL_SIZE; j++) {
            system->grid[i][j].count = 0;
        }
    }
}

// Add a particle to the system
void ParticleSystem_AddParticle(ParticleSystem *system,
                                float x,
                                float y,
                                float z,
                                float vx,
                                float vy,
                                float vz) {
    if (system->numParticles < MAX_PARTICLES) {
        Particle *p = &system->particles[system->numParticles++];
        p->x = x;
        p->y = y;
        p->z = z;
        p->padding1 = 0.0f;
        p->vx = vx;
        p->vy = vy;
        p->vz = vz;
        p->life = 1.0f;

        // Add particle to the grid
        int gridX = (int)(x / GRID_CELL_SIZE);
        int gridY = (int)(y / GRID_CELL_SIZE);
        if (gridX >= 0 && gridX < GRID_CELL_SIZE && gridY >= 0 &&
            gridY < GRID_CELL_SIZE) {
            GridCell *cell = &system->grid[gridX][gridY];
            if (cell->count < MAX_PARTICLES / GRID_CELL_SIZE) {
                cell->particles[cell->count++] = *p;
            }
        }
    }
}

// Check if a point is inside collision bounds (AABB)
int ParticleSystem_CheckCollision(float x,
                                  float y,
                                  float z,
                                  CollisionBounds *bounds) {
    return (x >= bounds->minX && x <= bounds->maxX && y >= bounds->minY &&
            y <= bounds->maxY && z >= bounds->minZ && z <= bounds->maxZ);
}

// Resolve collision - push particle out and reflect velocity
void ParticleSystem_ResolveCollision(Particle *p, CollisionBounds *bounds) {
    float halfSizeX = (bounds->maxX - bounds->minX) * 0.5f;
    float halfSizeY = (bounds->maxY - bounds->minY) * 0.5f;
    float halfSizeZ = (bounds->maxZ - bounds->minZ) * 0.5f;

    float toParticleX = p->x - bounds->centerX;
    float toParticleY = p->y - bounds->centerY;
    float toParticleZ = p->z - bounds->centerZ;

    // Find penetration depth for each axis
    float penX = halfSizeX - fabsf(toParticleX);
    float penY = halfSizeY - fabsf(toParticleY);
    float penZ = halfSizeZ - fabsf(toParticleZ);

    // Find the collision normal (axis with minimum penetration)
    float normalX = 0.0f, normalY = 0.0f, normalZ = 0.0f;

    if (penX < penY && penX < penZ) {
        normalX = (toParticleX > 0) ? 1.0f : -1.0f;
        p->x = bounds->centerX + normalX * (halfSizeX + 0.01f);
    } else if (penY < penZ) {
        normalY = (toParticleY > 0) ? 1.0f : -1.0f;
        p->y = bounds->centerY + normalY * (halfSizeY + 0.01f);
    } else {
        normalZ = (toParticleZ > 0) ? 1.0f : -1.0f;
        p->z = bounds->centerZ + normalZ * (halfSizeZ + 0.01f);
    }

    // Reflect velocity with energy loss
    float restitution = 0.3f;
    float velDotNormal = p->vx * normalX + p->vy * normalY + p->vz * normalZ;

    if (velDotNormal < 0.0f) {
        p->vx -= (1.0f + restitution) * velDotNormal * normalX;
        p->vy -= (1.0f + restitution) * velDotNormal * normalY;
        p->vz -= (1.0f + restitution) * velDotNormal * normalZ;

        // Add turbulence
        p->vy += ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        p->vz += ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
}

// Update with collision detection (CPU fallback)
void ParticleSystem_UpdateWithCollision(ParticleSystem *system,
                                        FluidCube *fluid,
                                        float dt,
                                        CollisionBounds *bounds) {
    int sizeX = fluid->sizeX;
    int sizeY = fluid->sizeY;
    int sizeZ = fluid->sizeZ;

    // Clear the grid
    for (int i = 0; i < GRID_CELL_SIZE; i++) {
        for (int j = 0; j < GRID_CELL_SIZE; j++) {
            system->grid[i][j].count = 0;
        }
    }

// Update particles
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < system->numParticles; i++) {
        Particle *p = &system->particles[i];

        // Update position
        p->x += p->vx * dt;
        p->y += p->vy * dt;
        p->z += p->vz * dt;

        // Check collision with car bounds
        if (bounds && ParticleSystem_CheckCollision(p->x, p->y, p->z, bounds)) {
            ParticleSystem_ResolveCollision(p, bounds);
        }

        // Decrease lifetime
        p->life -= 0.01f * dt;

        // Reset expired particles (wind tunnel style)
        if (p->life <= 0.0f || p->x > 4.0f) {
            p->x = -4.0f - ((float)rand() / RAND_MAX *
                            0.5f); // Randomize start X slightly
            p->y = ((float)rand() / RAND_MAX - 0.5f) * 3.0f;
            p->z = ((float)rand() / RAND_MAX - 0.5f) * 3.0f;
            p->vx = 0.5f + ((float)rand() / RAND_MAX) * 0.2f;
            p->vy = 0.0f;
            p->vz = 0.0f;
            p->life = 1.0f;
        }

        // Boundary constraints with epsilon pushback
        if (p->y < -2.0f || p->y > 2.0f) {
            p->vy *= -0.5f;
            p->y = (p->y < -2.0f) ? -1.99f : 1.99f;
        }
        if (p->z < -2.0f || p->z > 2.0f) {
            p->vz *= -0.5f;
            p->z = (p->z < -2.0f) ? -1.99f : 1.99f;
        }
    }

    // Reinsert particles into grid
    for (int i = 0; i < system->numParticles; i++) {
        Particle *p = &system->particles[i];
        int gridX = (int)((p->x + 4.0f) / 8.0f * GRID_CELL_SIZE);
        int gridY = (int)((p->y + 2.0f) / 4.0f * GRID_CELL_SIZE);

        if (gridX >= 0 && gridX < GRID_CELL_SIZE && gridY >= 0 &&
            gridY < GRID_CELL_SIZE) {
            GridCell *cell = &system->grid[gridX][gridY];
            if (cell->count < MAX_PARTICLES / GRID_CELL_SIZE) {
                cell->particles[cell->count++] = *p;
            }
        }
    }
}

void ParticleSystem_Update(ParticleSystem *system, FluidCube *fluid, float dt) {
    // Call the collision version with NULL bounds (no collision)
    ParticleSystem_UpdateWithCollision(system, fluid, dt, NULL);
}

void ParticleSystem_Render(ParticleSystem *system,
                           SDL_Renderer *renderer,
                           int scale) {
    // Set particle color with alpha blending
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    for (int i = 0; i < GRID_CELL_SIZE; i++) {
        for (int j = 0; j < GRID_CELL_SIZE; j++) {
            GridCell *cell = &system->grid[i][j];
            for (int k = 0; k < cell->count; k++) {
                Particle *p = &cell->particles[k];

                // Calculate distance from the camera (simplified to 2D for now)
                float distance = sqrt(p->x * p->x + p->y * p->y);

                // Skip particles that are far away (LOD)
                if (distance > 200)
                    continue;

                // Map particle position to screen coordinates
                int screenX = (int)(p->x * scale);
                int screenY = (int)(p->y * scale);

                // Draw particle as a small rectangle (2x2 pixels)
                SDL_Rect rect = {screenX, screenY, 2, 2};
                Uint8 alpha = (Uint8)(p->life * 255);
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, alpha);
                SDL_RenderFillRect(renderer, &rect);
            }
        }
    }
}