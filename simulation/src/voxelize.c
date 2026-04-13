#include "../lib/voxelize.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOXELIZE_EPS 1e-8f

typedef struct {
    float x, y, z;
} Vec3;

static inline int voxel_index(int x, int y, int z, int R) {
    return x + R * (y + R * z);
}

static Vec3 *copy_and_normalize_vertices(const Model *model,
                                         float padding,
                                         int align_longest_x) {
    if (model->vertexCount <= 0)
        return NULL;

    Vec3 *v = (Vec3 *)malloc((size_t)model->vertexCount * sizeof(Vec3));
    if (!v)
        return NULL;

    float min_x = model->vertices[0].x, max_x = min_x;
    float min_y = model->vertices[0].y, max_y = min_y;
    float min_z = model->vertices[0].z, max_z = min_z;

    for (int i = 0; i < model->vertexCount; i++) {
        float x = model->vertices[i].x;
        float y = model->vertices[i].y;
        float z = model->vertices[i].z;
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
        if (z < min_z) min_z = z;
        if (z > max_z) max_z = z;
    }

    float cx = 0.5f * (min_x + max_x);
    float cy = 0.5f * (min_y + max_y);
    float cz = 0.5f * (min_z + max_z);

    for (int i = 0; i < model->vertexCount; i++) {
        v[i].x = model->vertices[i].x - cx;
        v[i].y = model->vertices[i].y - cy;
        v[i].z = model->vertices[i].z - cz;
    }

    float size_x = max_x - min_x;
    float size_y = max_y - min_y;
    float size_z = max_z - min_z;

    if (align_longest_x) {
        /* Permute axes so the longest dimension becomes X.
         * Y longest: (x,y,z) -> (y,x,z).
         * Z longest: (x,y,z) -> (z,y,x). */
        if (size_y > size_x && size_y >= size_z) {
            for (int i = 0; i < model->vertexCount; i++) {
                float tmp = v[i].x;
                v[i].x = v[i].y;
                v[i].y = tmp;
            }
            float t = size_x; size_x = size_y; size_y = t;
        } else if (size_z > size_x && size_z > size_y) {
            for (int i = 0; i < model->vertexCount; i++) {
                float tmp = v[i].x;
                v[i].x = v[i].z;
                v[i].z = tmp;
            }
            float t = size_x; size_x = size_z; size_z = t;
        }
    }

    float max_dim = size_x;
    if (size_y > max_dim) max_dim = size_y;
    if (size_z > max_dim) max_dim = size_z;
    if (max_dim < VOXELIZE_EPS) {
        free(v);
        return NULL;
    }

    float scale = (2.0f - 2.0f * padding) / max_dim;
    for (int i = 0; i < model->vertexCount; i++) {
        v[i].x *= scale;
        v[i].y *= scale;
        v[i].z *= scale;
    }

    return v;
}

typedef struct {
    float *xs;
    int count;
    int capacity;
} HitList;

static int hitlist_push(HitList *h, float x) {
    if (h->count >= h->capacity) {
        int new_cap = h->capacity ? h->capacity * 2 : 4;
        float *new_xs = (float *)realloc(h->xs, (size_t)new_cap * sizeof(float));
        if (!new_xs)
            return -1;
        h->xs = new_xs;
        h->capacity = new_cap;
    }
    h->xs[h->count++] = x;
    return 0;
}

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

static void voxelize_solid(const Vec3 *verts,
                           const Face *faces,
                           int face_count,
                           int R,
                           uint8_t *grid) {
    HitList *hits = (HitList *)calloc((size_t)R * R, sizeof(HitList));
    if (!hits)
        return;

    for (int f = 0; f < face_count; f++) {
        int i0 = faces[f].v1 - 1;
        int i1 = faces[f].v2 - 1;
        int i2 = faces[f].v3 - 1;
        Vec3 v0 = verts[i0];
        Vec3 v1 = verts[i1];
        Vec3 v2 = verts[i2];

        float y_min = fminf(v0.y, fminf(v1.y, v2.y));
        float y_max = fmaxf(v0.y, fmaxf(v1.y, v2.y));
        float z_min = fminf(v0.z, fminf(v1.z, v2.z));
        float z_max = fmaxf(v0.z, fmaxf(v1.z, v2.z));

        if (y_max < -1.0f || y_min > 1.0f || z_max < -1.0f || z_min > 1.0f)
            continue;

        int gy_lo = (int)floorf((y_min + 1.0f) * 0.5f * R);
        int gy_hi = (int)ceilf((y_max + 1.0f) * 0.5f * R) + 1;
        int gz_lo = (int)floorf((z_min + 1.0f) * 0.5f * R);
        int gz_hi = (int)ceilf((z_max + 1.0f) * 0.5f * R) + 1;
        if (gy_lo < 0) gy_lo = 0;
        if (gz_lo < 0) gz_lo = 0;
        if (gy_hi > R) gy_hi = R;
        if (gz_hi > R) gz_hi = R;
        if (gy_lo >= gy_hi || gz_lo >= gz_hi)
            continue;

        float e1x = v1.x - v0.x, e1y = v1.y - v0.y, e1z = v1.z - v0.z;
        float e2x = v2.x - v0.x, e2y = v2.y - v0.y, e2z = v2.z - v0.z;

        /* Moller-Trumbore with ray direction (1, 0, 0):
         *   h = dir x e2 = (0, -e2z, e2y)
         *   a = e1 . h   = -e1y*e2z + e1z*e2y  (= x-component of e1 x e2) */
        float a = -e1y * e2z + e1z * e2y;
        if (fabsf(a) < VOXELIZE_EPS)
            continue;
        float inv_a = 1.0f / a;

        for (int gy = gy_lo; gy < gy_hi; gy++) {
            float wy = ((float)gy + 0.5f) / (float)R * 2.0f - 1.0f;
            float sy = wy - v0.y;
            for (int gz = gz_lo; gz < gz_hi; gz++) {
                float wz = ((float)gz + 0.5f) / (float)R * 2.0f - 1.0f;
                float sz = wz - v0.z;

                /* u = inv_a * (s . h)  where s_x is irrelevant (h_x = 0) */
                float u = inv_a * (sy * -e2z + sz * e2y);
                if (u < 0.0f || u > 1.0f)
                    continue;

                /* q = s x e1 */
                float sx = -v0.x; /* ray origin x = 0 */
                float qx = sy * e1z - sz * e1y;
                float qy = sz * e1x - sx * e1z;
                float qz = sx * e1y - sy * e1x;

                /* v = inv_a * (dir . q) = inv_a * qx */
                float v_coord = inv_a * qx;
                if (v_coord < 0.0f || u + v_coord > 1.0f)
                    continue;

                /* t = inv_a * (e2 . q); since origin is (0, wy, wz) and
                 * dir is (1, 0, 0), the intersection x-coordinate is t. */
                float t = inv_a * (e2x * qx + e2y * qy + e2z * qz);
                if (hitlist_push(&hits[gy * R + gz], t) != 0)
                    goto cleanup;
            }
        }
    }

    for (int gy = 0; gy < R; gy++) {
        for (int gz = 0; gz < R; gz++) {
            HitList *h = &hits[gy * R + gz];
            if (h->count < 2)
                continue;
            qsort(h->xs, (size_t)h->count, sizeof(float), cmp_float);

            /* Drop near-duplicate intersections (edge/vertex hits). */
            int n = 1;
            for (int i = 1; i < h->count; i++) {
                if (fabsf(h->xs[i] - h->xs[n - 1]) > 1e-6f) {
                    h->xs[n++] = h->xs[i];
                }
            }

            for (int i = 0; i + 1 < n; i += 2) {
                float x0 = h->xs[i];
                float x1 = h->xs[i + 1];
                int gx0 = (int)ceilf((x0 + 1.0f) * 0.5f * R - 0.5f);
                int gx1 = (int)floorf((x1 + 1.0f) * 0.5f * R - 0.5f) + 1;
                if (gx0 < 0) gx0 = 0;
                if (gx1 > R) gx1 = R;
                for (int gx = gx0; gx < gx1; gx++) {
                    grid[voxel_index(gx, gy, gz, R)] = 1;
                }
            }
        }
    }

cleanup:
    for (int i = 0; i < R * R; i++)
        free(hits[i].xs);
    free(hits);
}

static void voxelize_surface(const Vec3 *verts,
                             const Face *faces,
                             int face_count,
                             int R,
                             uint8_t *grid) {
    for (int f = 0; f < face_count; f++) {
        int i0 = faces[f].v1 - 1;
        int i1 = faces[f].v2 - 1;
        int i2 = faces[f].v3 - 1;
        Vec3 v0 = verts[i0];
        Vec3 v1 = verts[i1];
        Vec3 v2 = verts[i2];

        float x_min = fminf(v0.x, fminf(v1.x, v2.x));
        float x_max = fmaxf(v0.x, fmaxf(v1.x, v2.x));
        float y_min = fminf(v0.y, fminf(v1.y, v2.y));
        float y_max = fmaxf(v0.y, fmaxf(v1.y, v2.y));
        float z_min = fminf(v0.z, fminf(v1.z, v2.z));
        float z_max = fmaxf(v0.z, fmaxf(v1.z, v2.z));

        int gx0 = (int)floorf((x_min + 1.0f) * 0.5f * R);
        int gx1 = (int)ceilf((x_max + 1.0f) * 0.5f * R);
        int gy0 = (int)floorf((y_min + 1.0f) * 0.5f * R);
        int gy1 = (int)ceilf((y_max + 1.0f) * 0.5f * R);
        int gz0 = (int)floorf((z_min + 1.0f) * 0.5f * R);
        int gz1 = (int)ceilf((z_max + 1.0f) * 0.5f * R);
        if (gx0 < 0) gx0 = 0;
        if (gy0 < 0) gy0 = 0;
        if (gz0 < 0) gz0 = 0;
        if (gx1 > R) gx1 = R;
        if (gy1 > R) gy1 = R;
        if (gz1 > R) gz1 = R;

        for (int z = gz0; z < gz1; z++) {
            for (int y = gy0; y < gy1; y++) {
                for (int x = gx0; x < gx1; x++) {
                    grid[voxel_index(x, y, z, R)] = 1;
                }
            }
        }
    }
}

static int faces_valid(const Model *model) {
    for (int i = 0; i < model->faceCount; i++) {
        int i0 = model->faces[i].v1 - 1;
        int i1 = model->faces[i].v2 - 1;
        int i2 = model->faces[i].v3 - 1;
        if (i0 < 0 || i0 >= model->vertexCount) return 0;
        if (i1 < 0 || i1 >= model->vertexCount) return 0;
        if (i2 < 0 || i2 >= model->vertexCount) return 0;
    }
    return 1;
}

VoxelGrid *Voxelize_FromModel(const Model *model,
                              int resolution,
                              float padding,
                              int align_longest_x,
                              VoxelizeMode mode) {
    if (!model || resolution <= 0 || model->vertexCount <= 0 ||
        model->faceCount <= 0) {
        return NULL;
    }
    if (padding < 0.0f) padding = 0.0f;
    if (padding >= 0.5f) padding = 0.499f;
    if (!faces_valid(model))
        return NULL;

    Vec3 *verts = copy_and_normalize_vertices(model, padding, align_longest_x);
    if (!verts)
        return NULL;

    VoxelGrid *grid = (VoxelGrid *)malloc(sizeof(VoxelGrid));
    if (!grid) {
        free(verts);
        return NULL;
    }
    grid->resolution = resolution;
    size_t n = (size_t)resolution * resolution * resolution;
    grid->data = (uint8_t *)calloc(n, 1);
    if (!grid->data) {
        free(verts);
        free(grid);
        return NULL;
    }

    if (mode == VOXELIZE_MODE_SURFACE) {
        voxelize_surface(verts, model->faces, model->faceCount, resolution,
                         grid->data);
    } else {
        voxelize_solid(verts, model->faces, model->faceCount, resolution,
                       grid->data);
        if (mode == VOXELIZE_MODE_AUTO) {
            float frac = Voxelize_SolidFraction(grid);
            if (frac < 0.001f || frac > 0.99f) {
                memset(grid->data, 0, n);
                voxelize_surface(verts, model->faces, model->faceCount,
                                 resolution, grid->data);
            }
        }
    }

    free(verts);
    return grid;
}

float Voxelize_SolidFraction(const VoxelGrid *grid) {
    if (!grid || !grid->data || grid->resolution <= 0)
        return 0.0f;
    size_t n = (size_t)grid->resolution * grid->resolution * grid->resolution;
    size_t solid = 0;
    for (size_t i = 0; i < n; i++) {
        if (grid->data[i]) solid++;
    }
    return (float)solid / (float)n;
}

int Voxelize_WriteBinary(const VoxelGrid *grid, const char *path) {
    if (!grid || !grid->data || !path)
        return -1;

    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;

    const char magic[4] = {'V', 'O', 'X', 'B'};
    uint32_t version = 1;
    uint32_t resolution = (uint32_t)grid->resolution;

    if (fwrite(magic, 1, 4, f) != 4) goto fail;
    if (fwrite(&version, sizeof(uint32_t), 1, f) != 1) goto fail;
    if (fwrite(&resolution, sizeof(uint32_t), 1, f) != 1) goto fail;

    size_t n = (size_t)grid->resolution * grid->resolution * grid->resolution;
    if (fwrite(grid->data, 1, n, f) != n) goto fail;

    fclose(f);
    return 0;

fail:
    fclose(f);
    return -1;
}

void Voxelize_Free(VoxelGrid *grid) {
    if (!grid) return;
    free(grid->data);
    free(grid);
}
