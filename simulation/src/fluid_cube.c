#include "../lib/fluid_cube.h"
#include "../lib/coloring.h"
#include "../obj-file-loader/lib/model_loader.h"
#include "../lib/render_model.h"

void SDL_ExitWithError(const char *message) {
    printf("Error: %s > %s\n", message, SDL_GetError());
    SDL_Quit();
    exit(EXIT_FAILURE);
}

FluidCube *FluidCubeCreate(int sizeX,
                           int sizeY,
                           int sizeZ,
                           float diffusion,
                           float viscosity,
                           float dt,
                           Model *model) {
    printf("Creating fluid cube with size: %dx%dx%d\n", sizeX, sizeY, sizeZ);
    FluidCube *cube = malloc(sizeof(*cube));
    if (cube == NULL) {
        printf("Out of memory!\n");
        return NULL;
    }

    int N = sizeX * sizeY * sizeZ;

    cube->sizeX = sizeX;
    cube->sizeY = sizeY;
    cube->sizeZ = sizeZ;
    cube->dt = dt;
    cube->diff = diffusion;
    cube->visc = viscosity;
    cube->model = model; // Store the model pointer

    // Allocate memory for arrays...
    cube->s = calloc(N, sizeof(float));
    cube->density = calloc(N, sizeof(float));
    cube->Vx = calloc(N, sizeof(float));
    cube->Vy = calloc(N, sizeof(float));
    cube->Vz = calloc(N, sizeof(float));
    cube->Vx0 = calloc(N, sizeof(float));
    cube->Vy0 = calloc(N, sizeof(float));
    cube->Vz0 = calloc(N, sizeof(float));

    if (cube->s == NULL || cube->density == NULL || cube->Vx == NULL ||
        cube->Vy == NULL || cube->Vz == NULL || cube->Vx0 == NULL ||
        cube->Vy0 == NULL || cube->Vz0 == NULL) {
        printf("Out of memory!\n");
        FluidCubeFree(cube);
        return NULL;
    }

    printf("Fluid cube memory allocated successfully.\n");
    return cube;
}

void FluidCubeFree(FluidCube *cube) {
    free(cube->s);
    free(cube->density);

    free(cube->Vx);
    free(cube->Vy);
    free(cube->Vz);

    free(cube->Vx0);
    free(cube->Vy0);
    free(cube->Vz0);

    free(cube);
}

void FluidCubeAddDensity(FluidCube *cube, int x, int y, int z, float amount) {
    if (x < 0 || x >= cube->sizeX || y < 0 || y >= cube->sizeY || z < 0 ||
        z >= cube->sizeZ) {
        printf("Error: Invalid indices in FluidAddDensity (%d, %d, %d)\n",
               x,
               y,
               z); // Debug print
        return;
    }
    cube->density[IX3D(x, y, z, cube->sizeX, cube->sizeY)] += amount;
}

void FluidCubeAddVelocity(
    FluidCube *cube, int x, int y, int z, float amtX, float amtY, float amtZ) {
    if (x < 0 || x >= cube->sizeX || y < 0 || y >= cube->sizeY || z < 0 ||
        z >= cube->sizeZ) {
        printf("Error: Invalid indices in FluidAddVelocity (%d, %d, %d)\n",
               x,
               y,
               z);
        return;
    }
    int index = IX3D(x, y, z, cube->sizeX, cube->sizeY);
    cube->Vx[index] += amtX;
    cube->Vy[index] += amtY;
    cube->Vz[index] += amtZ;
}

static void
set_bnd(int b, float *x, int sizeX, int sizeY, int sizeZ, FluidCube *cube) {
    // Implement boundary conditions for 3D
    for (int i = 1; i < sizeX - 1; i++) {
        for (int j = 1; j < sizeY - 1; j++) {
            x[IX3D(i, j, 0, sizeX, sizeY)] =
                b == 3 ? -x[IX3D(i, j, 1, sizeX, sizeY)]
                       : x[IX3D(i, j, 1, sizeX, sizeY)];
            x[IX3D(i, j, sizeZ - 1, sizeX, sizeY)] =
                b == 3 ? -x[IX3D(i, j, sizeZ - 2, sizeX, sizeY)]
                       : x[IX3D(i, j, sizeZ - 2, sizeX, sizeY)];
        }
    }

    for (int i = 1; i < sizeX - 1; i++) {
        for (int k = 1; k < sizeZ - 1; k++) {
            x[IX3D(i, 0, k, sizeX, sizeY)] =
                b == 2 ? -x[IX3D(i, 1, k, sizeX, sizeY)]
                       : x[IX3D(i, 1, k, sizeX, sizeY)];
            x[IX3D(i, sizeY - 1, k, sizeX, sizeY)] =
                b == 2 ? -x[IX3D(i, sizeY - 2, k, sizeX, sizeY)]
                       : x[IX3D(i, sizeY - 2, k, sizeX, sizeY)];
        }
    }

    for (int j = 1; j < sizeY - 1; j++) {
        for (int k = 1; k < sizeZ - 1; k++) {
            x[IX3D(0, j, k, sizeX, sizeY)] =
                b == 1 ? -x[IX3D(1, j, k, sizeX, sizeY)]
                       : x[IX3D(1, j, k, sizeX, sizeY)];
            x[IX3D(sizeX - 1, j, k, sizeX, sizeY)] =
                b == 1 ? -x[IX3D(sizeX - 2, j, k, sizeX, sizeY)]
                       : x[IX3D(sizeX - 2, j, k, sizeX, sizeY)];
        }
    }

    // Corners
    x[IX3D(0, 0, 0, sizeX, sizeY)] = 0.5f * (x[IX3D(1, 0, 0, sizeX, sizeY)] +
                                             x[IX3D(0, 1, 0, sizeX, sizeY)] +
                                             x[IX3D(0, 0, 1, sizeX, sizeY)]);
    x[IX3D(0, sizeY - 1, 0, sizeX, sizeY)] =
        0.5f * (x[IX3D(1, sizeY - 1, 0, sizeX, sizeY)] +
                x[IX3D(0, sizeY - 2, 0, sizeX, sizeY)] +
                x[IX3D(0, sizeY - 1, 1, sizeX, sizeY)]);
    x[IX3D(sizeX - 1, 0, 0, sizeX, sizeY)] =
        0.5f * (x[IX3D(sizeX - 2, 0, 0, sizeX, sizeY)] +
                x[IX3D(sizeX - 1, 1, 0, sizeX, sizeY)] +
                x[IX3D(sizeX - 1, 0, 1, sizeX, sizeY)]);
    x[IX3D(sizeX - 1, sizeY - 1, 0, sizeX, sizeY)] =
        0.5f * (x[IX3D(sizeX - 2, sizeY - 1, 0, sizeX, sizeY)] +
                x[IX3D(sizeX - 1, sizeY - 2, 0, sizeX, sizeY)] +
                x[IX3D(sizeX - 1, sizeY - 1, 1, sizeX, sizeY)]);
    x[IX3D(0, 0, sizeZ - 1, sizeX, sizeY)] =
        0.5f * (x[IX3D(1, 0, sizeZ - 1, sizeX, sizeY)] +
                x[IX3D(0, 1, sizeZ - 1, sizeX, sizeY)] +
                x[IX3D(0, 0, sizeZ - 2, sizeX, sizeY)]);
    x[IX3D(0, sizeY - 1, sizeZ - 1, sizeX, sizeY)] =
        0.5f * (x[IX3D(1, sizeY - 1, sizeZ - 1, sizeX, sizeY)] +
                x[IX3D(0, sizeY - 2, sizeZ - 1, sizeX, sizeY)] +
                x[IX3D(0, sizeY - 1, sizeZ - 2, sizeX, sizeY)]);
    x[IX3D(sizeX - 1, 0, sizeZ - 1, sizeX, sizeY)] =
        0.5f * (x[IX3D(sizeX - 2, 0, sizeZ - 1, sizeX, sizeY)] +
                x[IX3D(sizeX - 1, 1, sizeZ - 1, sizeX, sizeY)] +
                x[IX3D(sizeX - 1, 0, sizeZ - 2, sizeX, sizeY)]);
    x[IX3D(sizeX - 1, sizeY - 1, sizeZ - 1, sizeX, sizeY)] =
        0.5f * (x[IX3D(sizeX - 2, sizeY - 1, sizeZ - 1, sizeX, sizeY)] +
                x[IX3D(sizeX - 1, sizeY - 2, sizeZ - 1, sizeX, sizeY)] +
                x[IX3D(sizeX - 1, sizeY - 1, sizeZ - 2, sizeX, sizeY)]);

    // Add boundary conditions for the car model
    if (cube->model) { // Check if the model exists
        for (int i = 0; i < sizeX; i++) {
            for (int j = 0; j < sizeY; j++) {
                for (int k = 0; k < sizeZ; k++) {
                    if (isInsideCarModel(
                            i, j, k, cube->model, sizeX, sizeY, sizeZ)) {
                        x[IX3D(i, j, k, sizeX, sizeY)] =
                            0; // Set velocity/density to zero inside the car
                               // model
                    }
                }
            }
        }
    }
}

static void lin_solve(int b,
                      float *x,
                      float *x0,
                      float a,
                      float c,
                      int iter,
                      int sizeX,
                      int sizeY,
                      int sizeZ,
                      FluidCube *cube) {
    float cRecip = 1.0 / c;
    for (int k = 0; k < iter; k++) {
        for (int i = 1; i < sizeX - 1; i++) {
            for (int j = 1; j < sizeY - 1; j++) {
                for (int k = 1; k < sizeZ - 1; k++) {
                    x[IX3D(i, j, k, sizeX, sizeY)] =
                        (x0[IX3D(i, j, k, sizeX, sizeY)] +
                         a * (x[IX3D(i + 1, j, k, sizeX, sizeY)] +
                              x[IX3D(i - 1, j, k, sizeX, sizeY)] +
                              x[IX3D(i, j + 1, k, sizeX, sizeY)] +
                              x[IX3D(i, j - 1, k, sizeX, sizeY)] +
                              x[IX3D(i, j, k + 1, sizeX, sizeY)] +
                              x[IX3D(i, j, k - 1, sizeX, sizeY)])) *
                        cRecip;
                }
            }
        }
        set_bnd(b, x, sizeX, sizeY, sizeZ, cube);
    }
}

static void diffuse(int b,
                    float *x,
                    float *x0,
                    float diff,
                    float dt,
                    int iter,
                    int sizeX,
                    int sizeY,
                    int sizeZ,
                    FluidCube *cube) {
    float a = dt * diff * (sizeX - 2) * (sizeY - 2) * (sizeZ - 2);
    lin_solve(b, x, x0, a, 1 + 6 * a, iter, sizeX, sizeY, sizeZ, cube);
    set_bnd(b, x, sizeX, sizeY, sizeZ, cube);
}

static void project(float *velocX,
                    float *velocY,
                    float *velocZ,
                    float *p,
                    float *div,
                    int iter,
                    int sizeX,
                    int sizeY,
                    int sizeZ,
                    FluidCube *cube) {
    // Calculate divergence
    for (int i = 1; i < sizeX - 1; i++) {
        for (int j = 1; j < sizeY - 1; j++) {
            for (int k = 1; k < sizeZ - 1; k++) {
                div[IX3D(i, j, k, sizeX, sizeY)] =
                    -0.5f *
                    (velocX[IX3D(i + 1, j, k, sizeX, sizeY)] -
                     velocX[IX3D(i - 1, j, k, sizeX, sizeY)] +
                     velocY[IX3D(i, j + 1, k, sizeX, sizeY)] -
                     velocY[IX3D(i, j - 1, k, sizeX, sizeY)] +
                     velocZ[IX3D(i, j, k + 1, sizeX, sizeY)] -
                     velocZ[IX3D(i, j, k - 1, sizeX, sizeY)]) /
                    (sizeX + sizeY + sizeZ);
                p[IX3D(i, j, k, sizeX, sizeY)] = 0;
            }
        }
    }

    // Set boundary conditions for divergence and pressure
    set_bnd(0, div, sizeX, sizeY, sizeZ, cube);
    set_bnd(0, p, sizeX, sizeY, sizeZ, cube);

    // Solve for pressure
    lin_solve(0, p, div, 1, 6, iter, sizeX, sizeY, sizeZ, cube);

    // Update velocity fields based on pressure
    for (int i = 1; i < sizeX - 1; i++) {
        for (int j = 1; j < sizeY - 1; j++) {
            for (int k = 1; k < sizeZ - 1; k++) {
                velocX[IX3D(i, j, k, sizeX, sizeY)] -=
                    0.5f *
                    (p[IX3D(i + 1, j, k, sizeX, sizeY)] -
                     p[IX3D(i - 1, j, k, sizeX, sizeY)]) *
                    sizeX;
                velocY[IX3D(i, j, k, sizeX, sizeY)] -=
                    0.5f *
                    (p[IX3D(i, j + 1, k, sizeX, sizeY)] -
                     p[IX3D(i, j - 1, k, sizeX, sizeY)]) *
                    sizeY;
                velocZ[IX3D(i, j, k, sizeX, sizeY)] -=
                    0.5f *
                    (p[IX3D(i, j, k + 1, sizeX, sizeY)] -
                     p[IX3D(i, j, k - 1, sizeX, sizeY)]) *
                    sizeZ;
            }
        }
    }

    // Set boundary conditions for velocity fields
    set_bnd(1, velocX, sizeX, sizeY, sizeZ, cube);
    set_bnd(2, velocY, sizeX, sizeY, sizeZ, cube);
    set_bnd(3, velocZ, sizeX, sizeY, sizeZ, cube);
}

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
                   FluidCube *cube) {
    float i0, i1, j0, j1, k0, k1;

    float dtx = dt * (sizeX - 2);
    float dty = dt * (sizeY - 2);
    float dtz = dt * (sizeZ - 2);

    float s0, s1, t0, t1, u0, u1;
    float tmp1, tmp2, tmp3, x, y, z;

    float NfloatX = sizeX;
    float NfloatY = sizeY;
    float NfloatZ = sizeZ;
    float ifloat, jfloat, kfloat;
    int i, j, k;

    for (i = 1, ifloat = 1; i < sizeX - 1; i++, ifloat++) {
        for (j = 1, jfloat = 1; j < sizeY - 1; j++, jfloat++) {
            for (k = 1, kfloat = 1; k < sizeZ - 1; k++, kfloat++) {
                tmp1 = dtx * velocX[IX3D(i, j, k, sizeX, sizeY)];
                tmp2 = dty * velocY[IX3D(i, j, k, sizeX, sizeY)];
                tmp3 = dtz * velocZ[IX3D(i, j, k, sizeX, sizeY)];
                x = ifloat - tmp1;
                y = jfloat - tmp2;
                z = kfloat - tmp3;

                if (x < 0.5f)
                    x = 0.5f;
                if (x > NfloatX + 0.5f)
                    x = NfloatX + 0.5f;
                i0 = floorf(x);
                i1 = i0 + 1.0f;
                if (y < 0.5f)
                    y = 0.5f;
                if (y > NfloatY + 0.5f)
                    y = NfloatY + 0.5f;
                j0 = floorf(y);
                j1 = j0 + 1.0f;
                if (z < 0.5f)
                    z = 0.5f;
                if (z > NfloatZ + 0.5f)
                    z = NfloatZ + 0.5f;
                k0 = floorf(z);
                k1 = k0 + 1.0f;

                s1 = x - i0;
                s0 = 1.0f - s1;
                t1 = y - j0;
                t0 = 1.0f - t1;
                u1 = z - k0;
                u0 = 1.0f - u1;

                int i0i = i0;
                int i1i = i1;
                int j0i = j0;
                int j1i = j1;
                int k0i = k0;
                int k1i = k1;

                // Ensure indices are within bounds
                if (i0i < 0 || i0i >= sizeX || i1i < 0 || i1i >= sizeX ||
                    j0i < 0 || j0i >= sizeY || j1i < 0 || j1i >= sizeY ||
                    k0i < 0 || k0i >= sizeZ || k1i < 0 || k1i >= sizeZ) {
                    printf("Error: Invalid indices in advect (%d, %d, %d).\n",
                           i0i,
                           j0i,
                           k0i);
                    continue;
                }

                d[IX3D(i, j, k, sizeX, sizeY)] =
                    s0 * (t0 * (u0 * d0[IX3D(i0i, j0i, k0i, sizeX, sizeY)] +
                                u1 * d0[IX3D(i0i, j0i, k1i, sizeX, sizeY)]) +
                          t1 * (u0 * d0[IX3D(i0i, j1i, k0i, sizeX, sizeY)] +
                                u1 * d0[IX3D(i0i, j1i, k1i, sizeX, sizeY)])) +
                    s1 * (t0 * (u0 * d0[IX3D(i1i, j0i, k0i, sizeX, sizeY)] +
                                u1 * d0[IX3D(i1i, j0i, k1i, sizeX, sizeY)]) +
                          t1 * (u0 * d0[IX3D(i1i, j1i, k0i, sizeX, sizeY)] +
                                u1 * d0[IX3D(i1i, j1i, k1i, sizeX, sizeY)]));
            }
        }
    }

    set_bnd(b, d, sizeX, sizeY, sizeZ, cube);
}

void FluidCubeStep(FluidCube *cube) {
    int sizeX = cube->sizeX;
    int sizeY = cube->sizeY;
    int sizeZ = cube->sizeZ;
    float *Vx = cube->Vx;
    float *Vy = cube->Vy;
    float *Vz = cube->Vz;
    float *Vx0 = cube->Vx0;
    float *Vy0 = cube->Vy0;
    float *Vz0 = cube->Vz0;
    float *p = cube->s;         // Pressure array
    float *div = cube->density; // Divergence array

    // Diffuse velocity fields
    diffuse(1, Vx0, Vx, cube->visc, cube->dt, 4, sizeX, sizeY, sizeZ, cube);
    diffuse(2, Vy0, Vy, cube->visc, cube->dt, 4, sizeX, sizeY, sizeZ, cube);
    diffuse(3, Vz0, Vz, cube->visc, cube->dt, 4, sizeX, sizeY, sizeZ, cube);

    // Project velocity fields to enforce incompressibility
    project(Vx0, Vy0, Vz0, p, div, 4, sizeX, sizeY, sizeZ, cube);

    // Advect velocity fields
    advect(1, Vx, Vx0, Vx0, Vy0, Vz0, cube->dt, sizeX, sizeY, sizeZ, cube);
    advect(2, Vy, Vy0, Vx0, Vy0, Vz0, cube->dt, sizeX, sizeY, sizeZ, cube);
    advect(3, Vz, Vz0, Vx0, Vy0, Vz0, cube->dt, sizeX, sizeY, sizeZ, cube);

    // Project velocity fields again
    project(Vx, Vy, Vz, p, div, 4, sizeX, sizeY, sizeZ, cube);

    // Diffuse and advect density
    diffuse(0,
            cube->s,
            cube->density,
            cube->diff,
            cube->dt,
            4,
            sizeX,
            sizeY,
            sizeZ,
            cube);
    advect(0,
           cube->density,
           cube->s,
           Vx,
           Vy,
           Vz,
           cube->dt,
           sizeX,
           sizeY,
           sizeZ,
           cube);
}