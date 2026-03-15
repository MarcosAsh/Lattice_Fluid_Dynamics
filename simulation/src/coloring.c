#include "../lib/coloring.h"

// Function to map density to a grayscale color
void DensityToColor(float density, Uint8 *r, Uint8 *g, Uint8 *b) {
    int color = (int)(density * 255);
    color = color > 255 ? 255 : color; // Clamp to 255
    *r = color;
    *g = color;
    *b = color;
    printf("DensityToColor: %f, r=%d, g=%d b=%d\n",
           density,
           *r,
           *g,
           *b); // Debug print
}

// Function to map velocity magnitude to a red color
void VelocityToColor(float velX,
                     float velY,
                     float velZ,
                     Uint8 *r,
                     Uint8 *g,
                     Uint8 *b,
                     int velThreshold) {
    float velMagnitude = sqrt(velX * velX + velY * velY + velZ * velZ);
    float normalizedVel = velMagnitude / velThreshold;
    normalizedVel = normalizedVel > 1.0f ? 1.0f : normalizedVel; // Clamp to 1.0

    *r = (Uint8)(normalizedVel * 255);
    *g = 0;
    *b = 0;
    printf("VelocityToColor: velMagnitude=%f, r=%d, g=%d b%d\n",
           velMagnitude,
           *r,
           *g,
           *b); // Debug print
}

// Function to map both density and velocity to a combined color
void DensityAndVelocityToColor(float density,
                               float velX,
                               float velY,
                               float velZ,
                               Uint8 *r,
                               Uint8 *g,
                               Uint8 *b,
                               int velThreshold) {
    int densityColor = (int)(density * 255);
    densityColor = densityColor > 255 ? 255 : densityColor; // Clamp to 255

    float velMagnitude = sqrt(velX * velX + velY * velY + velZ * velZ);
    float normalizedVel = velMagnitude / velThreshold;
    normalizedVel = normalizedVel > 1.0f ? 1.0f : normalizedVel; // Clamp to 1

    // Combine density and velocity into a single color
    *r = 0;
    *g = (Uint8)(50 + (densityColor + normalizedVel * 255) / 2);
    *g = *g > 255 ? 255 : *g; // Clamp to 255
    *b = (Uint8)(100 + (densityColor + normalizedVel * 255) / 2);
    *b = *b > 255 ? 255 : *b; // Clamp to 255
    printf("DensityAndVelocityToColor: density=%f, velMagnitude=%f, r=%d, "
           "g=%d, b=%d\n",
           density,
           velMagnitude,
           *r,
           *g,
           *b); // Debug print
}