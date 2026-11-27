#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

// Particle structure
// The struct has padding to align with GPU memory layout
typedef struct {
    float x, y, z;    // Position (vec3)
    float padding1;   // Padding for 16-byte alignment
    float vx, vy, vz; // Velocity (vec3)
    float life;       // Lifetime (0.0 to 1.0)
} Particle;

#endif // PARTICLE_SYSTEM_H
