#ifndef CONFIG_H
#define CONFIG_H

#ifndef IX3D
#define IX3D(x, y, z, sizeX, sizeY)                                            \
    ((x) + (y) * (sizeX) + (z) * (sizeX) * (sizeY))
#endif

#ifndef WIDTH
#define WIDTH 1920
#endif

#ifndef HEIGHT
#define HEIGHT 1080
#endif

// Global constants
#define DEPTH 500
#define SCALE 5

// Global model transform (extern declarations)
extern float g_modelScale;
extern float g_offsetX;
extern float g_offsetY;
extern float g_offsetZ;
extern float g_carRotationY;

// Particle count used for both CPU and GPU particle systems
#ifndef MAX_PARTICLES
#define MAX_PARTICLES 30000
#endif

#endif // CONFIG_H