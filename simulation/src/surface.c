#include "../lib/surface.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int surface_init(Surface *s,
                 int nx,
                 int ny,
                 int nz,
                 int n,
                 int *cell,
                 unsigned char *role,
                 int *w36,
                 int *links) {
    memset(s, 0, sizeof(*s));
    s->nx = nx;
    s->ny = ny;
    s->nz = nz;
    s->n = n;
    s->cell = cell;
    s->role = role;
    s->w36 = w36;
    s->links = links;
    s->area = (double *)calloc((size_t)n, sizeof(double));
    s->normal = (double *)calloc((size_t)n * 3, sizeof(double));
    s->sumF = (double *)calloc((size_t)n * 3, sizeof(double));
    s->sumRho = (double *)calloc((size_t)n, sizeof(double));
    if (!s->area || !s->normal || !s->sumF || !s->sumRho)
        return -1;
    // Without a mesh the normal points against the links into the solid
    for (int k = 0; k < n; k++) {
        double v[3] = {-w36[3 * k], -w36[3 * k + 1], -w36[3 * k + 2]};
        double len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        for (int c = 0; c < 3 && len > 0; c++)
            s->normal[3 * k + c] = v[c] / len;
    }
    return 0;
}

void surface_free(Surface *s) {
    free(s->cell);
    free(s->role);
    free(s->w36);
    free(s->links);
    free(s->area);
    free(s->normal);
    free(s->tri);
    free(s->triArea);
    free(s->triNormal);
    free(s->wStart);
    free(s->wEntry);
    free(s->wGive);
    free(s->wShare);
    free(s->sumF);
    free(s->sumRho);
    memset(s, 0, sizeof(*s));
}

int surface_find(const Surface *s, int cell) {
    int lo = 0, hi = s->n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (s->cell[mid] == cell)
            return mid;
        if (s->cell[mid] < cell)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;
}

static void centre_of(const Surface *s, int k, double c[3]) {
    int g = s->cell[k];
    int plane = s->nx * s->ny;
    c[0] = g % s->nx + 0.5;
    c[1] = (g % plane) / s->nx + 0.5;
    c[2] = g / plane + 0.5;
}

typedef struct {
    float v[9];
} Tri;

typedef struct {
    Tri *t;
    int n, cap;
} TriList;

static int push_tri(TriList *l, const float *v) {
    if (l->n == l->cap) {
        int cap = l->cap ? 2 * l->cap : 1024;
        Tri *p = (Tri *)realloc(l->t, (size_t)cap * sizeof(Tri));
        if (!p)
            return -1;
        l->t = p;
        l->cap = cap;
    }
    memcpy(l->t[l->n++].v, v, sizeof(float) * 9);
    return 0;
}

// Splits into four until every edge is at most SURF_MAX_EDGE cells
static int split_tri(TriList *l, const float *v, int depth) {
    double e = 0;
    for (int a = 0; a < 3; a++) {
        const float *p = v + 3 * a, *q = v + 3 * ((a + 1) % 3);
        double d2 = 0;
        for (int c = 0; c < 3; c++)
            d2 += (double)(p[c] - q[c]) * (p[c] - q[c]);
        e = fmax(e, sqrt(d2));
    }
    if (e <= SURF_MAX_EDGE || depth >= 12)
        return push_tri(l, v);
    float m[3][3];
    for (int a = 0; a < 3; a++)
        for (int c = 0; c < 3; c++)
            m[a][c] = 0.5f * (v[3 * a + c] + v[3 * ((a + 1) % 3) + c]);
    float t[4][9];
    for (int c = 0; c < 3; c++) {
        t[0][c] = v[c], t[0][3 + c] = m[0][c], t[0][6 + c] = m[2][c];
        t[1][c] = m[0][c], t[1][3 + c] = v[3 + c], t[1][6 + c] = m[1][c];
        t[2][c] = m[2][c], t[2][3 + c] = m[1][c], t[2][6 + c] = v[6 + c];
        t[3][c] = m[0][c], t[3][3 + c] = m[1][c], t[3][6 + c] = m[2][c];
    }
    for (int k = 0; k < 4; k++)
        if (split_tri(l, t[k], depth + 1))
            return -1;
    return 0;
}

typedef struct {
    int key; // grid cell of the centroid
    int tri;
} Bucket;

static int bucket_cmp(const void *a, const void *b) {
    const Bucket *p = (const Bucket *)a, *q = (const Bucket *)b;
    if (p->key != q->key)
        return p->key < q->key ? -1 : 1;
    return (p->tri > q->tri) - (p->tri < q->tri);
}

typedef struct {
    int tri;
    int entry;
    double a; // area times kernel
} Pair;

static int pair_cmp(const void *a, const void *b) {
    const Pair *p = (const Pair *)a, *q = (const Pair *)b;
    if (p->tri != q->tri)
        return p->tri < q->tri ? -1 : 1;
    return (p->entry > q->entry) - (p->entry < q->entry);
}

int surface_map_mesh(Surface *s, const float *tris, int ntri) {
    double scale[3] = {s->nx / 8.0, s->ny / 4.0, s->nz / 4.0};
    double shift[3] = {4.0, 2.0, 2.0};
    TriList l = {0};
    for (int t = 0; t < ntri; t++) {
        float v[9];
        for (int a = 0; a < 3; a++)
            for (int c = 0; c < 3; c++)
                v[3 * a + c] =
                    (float)((tris[12 * t + 4 * a + c] + shift[c]) * scale[c]);
        if (split_tri(&l, v, 0)) {
            free(l.t);
            return -1;
        }
    }
    int nt = s->ntri = l.n;
    s->tri = (float *)malloc((nt > 0 ? (size_t)nt : 1) * 9 * sizeof(float));
    s->triArea = (double *)calloc(nt > 0 ? (size_t)nt : 1, sizeof(double));
    s->triNormal =
        (double *)calloc((nt > 0 ? (size_t)nt : 1) * 3, sizeof(double));
    s->wStart = (int *)calloc((size_t)nt + 1, sizeof(int));
    Bucket *bk = (Bucket *)malloc((nt > 0 ? (size_t)nt : 1) * sizeof(Bucket));
    if (!s->tri || !s->triArea || !s->triNormal || !s->wStart || !bk) {
        free(l.t);
        free(bk);
        return -1;
    }
    double (*cen)[3] =
        (double (*)[3])malloc((nt > 0 ? (size_t)nt : 1) * sizeof(*cen));
    if (!cen) {
        free(l.t);
        free(bk);
        return -1;
    }
    for (int t = 0; t < nt; t++) {
        const float *v = l.t[t].v;
        memcpy(&s->tri[9 * t], v, sizeof(float) * 9);
        double a[3], b[3];
        int ic[3];
        for (int d = 0; d < 3; d++) {
            a[d] = v[3 + d] - v[d];
            b[d] = v[6 + d] - v[d];
            cen[t][d] = (v[d] + v[3 + d] + v[6 + d]) / 3.0;
            ic[d] = (int)floor(cen[t][d]);
        }
        double n[3] = {a[1] * b[2] - a[2] * b[1],
                       a[2] * b[0] - a[0] * b[2],
                       a[0] * b[1] - a[1] * b[0]};
        double len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        s->triArea[t] = 0.5 * len;
        for (int d = 0; d < 3 && len > 0; d++)
            s->triNormal[3 * t + d] = n[d] / len;
        int inside = ic[0] >= 0 && ic[1] >= 0 && ic[2] >= 0 && ic[0] < s->nx &&
                     ic[1] < s->ny && ic[2] < s->nz;
        bk[t].key = inside ? ic[0] + s->nx * (ic[1] + s->ny * ic[2]) : -1;
        bk[t].tri = t;
    }
    free(l.t);
    qsort(bk, nt, sizeof(Bucket), bucket_cmp);

    // Pairs of boundary cell and triangle within the kernel radius
    const double R = SURF_KERNEL_RADIUS;
    int r = (int)ceil(R) + 1;
    Pair *pairs = NULL;
    size_t np = 0, cap = 0;
    double *rowSum =
        (double *)calloc(s->n > 0 ? (size_t)s->n : 1, sizeof(double));
    if (!rowSum) {
        free(bk);
        free(cen);
        return -1;
    }
    for (int k = 0; k < s->n; k++) {
        if (!(s->role[k] & SURF_BOUNDARY))
            continue;
        double c[3];
        centre_of(s, k, c);
        int ix = (int)c[0], iy = (int)c[1], iz = (int)c[2];
        for (int z = iz - r; z <= iz + r; z++)
            for (int y = iy - r; y <= iy + r; y++)
                for (int x = ix - r; x <= ix + r; x++) {
                    if (x < 0 || y < 0 || z < 0 || x >= s->nx || y >= s->ny ||
                        z >= s->nz)
                        continue;
                    int key = x + s->nx * (y + s->ny * z);
                    int lo = 0, hi = nt;
                    while (lo < hi) {
                        int mid = lo + (hi - lo) / 2;
                        if (bk[mid].key < key)
                            lo = mid + 1;
                        else
                            hi = mid;
                    }
                    for (int b = lo; b < nt && bk[b].key == key; b++) {
                        int t = bk[b].tri;
                        double d =
                            sqrt((cen[t][0] - c[0]) * (cen[t][0] - c[0]) +
                                 (cen[t][1] - c[1]) * (cen[t][1] - c[1]) +
                                 (cen[t][2] - c[2]) * (cen[t][2] - c[2]));
                        if (d >= R || s->triArea[t] <= 0)
                            continue;
                        double w = s->triArea[t] * (1 - d / R) * (1 - d / R);
                        if (np == cap) {
                            cap = cap ? 2 * cap : 4096;
                            Pair *p =
                                (Pair *)realloc(pairs, cap * sizeof(Pair));
                            if (!p) {
                                free(pairs);
                                free(bk);
                                free(cen);
                                free(rowSum);
                                return -1;
                            }
                            pairs = p;
                        }
                        pairs[np++] = (Pair){t, k, w};
                        rowSum[k] += w;
                    }
                }
    }
    free(bk);
    qsort(pairs, np, sizeof(Pair), pair_cmp);

    s->wEntry = (int *)malloc((np ? np : 1) * sizeof(int));
    s->wGive = (double *)malloc((np ? np : 1) * sizeof(double));
    s->wShare = (double *)malloc((np ? np : 1) * sizeof(double));
    if (!s->wEntry || !s->wGive || !s->wShare) {
        free(pairs);
        free(cen);
        free(rowSum);
        return -1;
    }
    for (size_t i = 0; i < np; i++)
        s->wStart[pairs[i].tri + 1]++;
    for (int t = 0; t < nt; t++)
        s->wStart[t + 1] += s->wStart[t];
    for (int t = 0; t < nt; t++) {
        int i0 = s->wStart[t], i1 = s->wStart[t + 1];
        if (i0 == i1) {
            s->unowned++;
            continue;
        }
        double colSum = 0, side = 0;
        for (int i = i0; i < i1; i++) {
            double c[3];
            centre_of(s, pairs[i].entry, c);
            colSum += pairs[i].a;
            side += pairs[i].a * ((c[0] - cen[t][0]) * s->triNormal[3 * t] +
                                  (c[1] - cen[t][1]) * s->triNormal[3 * t + 1] +
                                  (c[2] - cen[t][2]) * s->triNormal[3 * t + 2]);
        }
        // Out of the body is toward the fluid cells
        if (side < 0)
            for (int d = 0; d < 3; d++)
                s->triNormal[3 * t + d] = -s->triNormal[3 * t + d];
        for (int i = i0; i < i1; i++) {
            int k = pairs[i].entry;
            s->wEntry[i] = k;
            s->wGive[i] = pairs[i].a / rowSum[k];
            s->wShare[i] = pairs[i].a / colSum;
            s->area[k] += s->triArea[t] * s->wShare[i];
            for (int d = 0; d < 3; d++)
                s->normal[3 * k + d] += pairs[i].a * s->triNormal[3 * t + d];
        }
    }
    free(pairs);
    free(cen);
    free(rowSum);
    for (int k = 0; k < s->n; k++) {
        double *nk = &s->normal[3 * k];
        double len = sqrt(nk[0] * nk[0] + nk[1] * nk[1] + nk[2] * nk[2]);
        if (s->area[k] > 0 && len > 0)
            for (int d = 0; d < 3; d++)
                nk[d] /= len;
    }
    return 0;
}

void surface_add_window(Surface *s,
                        const float *sums,
                        int steps,
                        const float drag[3]) {
    for (int k = 0; k < s->n; k++) {
        for (int d = 0; d < 3; d++)
            s->sumF[3 * k + d] += sums[4 * k + d];
        s->sumRho[k] += sums[4 * k + 3];
    }
    for (int d = 0; d < 3; d++)
        s->dragSum[d] += (double)drag[d] * steps;
    s->steps += steps;
    s->windows++;
}

double surface_reference_rho(const Surface *s) {
    double sum = 0;
    long n = 0;
    for (int k = 0; k < s->n; k++)
        if (s->role[k] & SURF_REFERENCE) {
            sum += s->sumRho[k];
            n++;
        }
    return (n > 0 && s->steps > 0) ? sum / ((double)n * s->steps) : 1.0;
}

void surface_point(
    const Surface *s, int k, double u, double rhoRef, SurfacePoint *p) {
    memset(p, 0, sizeof(*p));
    if (s->steps == 0)
        return;
    double q = 0.5 * u * u;
    double inv = 1.0 / (double)s->steps;
    p->rho = s->sumRho[k] * inv;
    for (int d = 0; d < 3; d++)
        p->force[d] = s->sumF[3 * k + d] * inv + s->w36[3 * k + d] / 36.0;
    // p = rho cs^2
    p->cp = (p->rho - rhoRef) / 3.0 / q;

    // At rest the force is rho times the link sum W, so W.n / (1/3) is
    // the area the links see, staircase included
    const double *n = &s->normal[3 * k];
    double fn = 0, wn = 0;
    for (int d = 0; d < 3; d++) {
        fn += p->force[d] * n[d];
        wn += s->w36[3 * k + d] / 36.0 * n[d];
    }
    double latArea = -3.0 * wn;
    if (latArea >= SURF_MIN_LINK_AREA) {
        p->cpWall = (fn / (3.0 * wn) - rhoRef / 3.0) / q;
        for (int d = 0; d < 3; d++)
            p->shear[d] = (p->force[d] - fn * n[d]) / latArea;
        p->cf = sqrt(p->shear[0] * p->shear[0] + p->shear[1] * p->shear[1] +
                     p->shear[2] * p->shear[2]) /
                q;
        p->wall = 1;
    }
    if (s->area[k] > 0) {
        for (int d = 0; d < 3; d++)
            p->traction[d] = p->force[d] / s->area[k];
        p->mapped = 1;
    }
}

// Force the cells give triangle t, over its area. 0 if none in reach
static int tri_traction(const Surface *s, int t, double u, double out[3]) {
    out[0] = out[1] = out[2] = 0;
    int i0 = s->wStart[t], i1 = s->wStart[t + 1];
    if (i0 == i1 || s->triArea[t] <= 0)
        return 0;
    for (int i = i0; i < i1; i++) {
        SurfacePoint p;
        surface_point(s, s->wEntry[i], u, 1.0, &p);
        for (int d = 0; d < 3; d++)
            out[d] += s->wGive[i] * p.force[d] / s->triArea[t];
    }
    return 1;
}

int surface_triangle(const Surface *s,
                     int t,
                     double u,
                     double rhoRef,
                     double *cp,
                     double *cpWall,
                     double traction[3]) {
    *cp = *cpWall = 0;
    if (!tri_traction(s, t, u, traction))
        return 0;
    // Ratio of sums over the nearby cells along the triangle normal, a
    // per-cell ratio scatters where cells have few links
    const double *n = &s->triNormal[3 * t];
    double fn = 0, wn = 0;
    for (int i = s->wStart[t]; i < s->wStart[t + 1]; i++) {
        int k = s->wEntry[i];
        SurfacePoint p;
        surface_point(s, k, u, rhoRef, &p);
        *cp += s->wShare[i] * p.cp;
        for (int d = 0; d < 3; d++) {
            fn += s->wShare[i] * p.force[d] * n[d];
            wn += s->wShare[i] * s->w36[3 * k + d] / 36.0 * n[d];
        }
    }
    if (wn < 0)
        *cpWall = (fn / (3.0 * wn) - rhoRef / 3.0) / (0.5 * u * u);
    return 1;
}

void surface_totals(const Surface *s,
                    double cells[3],
                    double mesh[3],
                    double drag[3],
                    double unmapped[3]) {
    for (int d = 0; d < 3; d++)
        cells[d] = mesh[d] = drag[d] = unmapped[d] = 0;
    if (s->steps == 0)
        return;
    for (int k = 0; k < s->n; k++) {
        if (!(s->role[k] & SURF_BOUNDARY))
            continue;
        SurfacePoint p;
        surface_point(s, k, 1.0, 1.0, &p);
        for (int d = 0; d < 3; d++) {
            cells[d] += p.force[d];
            if (s->ntri > 0 && s->area[k] <= 0)
                unmapped[d] += p.force[d];
        }
    }
    for (int t = 0; t < s->ntri; t++) {
        double tr[3];
        if (tri_traction(s, t, 1.0, tr))
            for (int d = 0; d < 3; d++)
                mesh[d] += tr[d] * s->triArea[t];
    }
    for (int d = 0; d < 3; d++)
        drag[d] = s->dragSum[d] / (double)s->steps;
}

static void vtk_scalar(FILE *f, const char *name, const char *type) {
    fprintf(f, "SCALARS %s %s 1\nLOOKUP_TABLE default\n", name, type);
}

static int
write_cells_vtk(const Surface *s, const char *path, double u, double rhoRef) {
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    int nb = 0;
    for (int k = 0; k < s->n; k++)
        nb += (s->role[k] & SURF_BOUNDARY) != 0;
    SurfacePoint *pt = (SurfacePoint *)malloc((nb > 0 ? (size_t)nb : 1) *
                                              sizeof(SurfacePoint));
    int *idx = (int *)malloc((nb > 0 ? (size_t)nb : 1) * sizeof(int));
    if (!pt || !idx) {
        free(pt);
        free(idx);
        fclose(f);
        return -1;
    }
    for (int k = 0, j = 0; k < s->n; k++)
        if (s->role[k] & SURF_BOUNDARY) {
            idx[j] = k;
            surface_point(s, k, u, rhoRef, &pt[j++]);
        }

    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f,
            "Lattice boundary cells, steps %ld, windows %d, U %.9g, "
            "rho_ref %.9g, lattice units\n",
            s->steps,
            s->windows,
            u,
            rhoRef);
    fprintf(f, "ASCII\nDATASET POLYDATA\nPOINTS %d double\n", nb);
    for (int j = 0; j < nb; j++) {
        double c[3];
        centre_of(s, idx[j], c);
        fprintf(f, "%.9g %.9g %.9g\n", c[0], c[1], c[2]);
    }
    fprintf(f, "VERTICES %d %d\n", nb, 2 * nb);
    for (int j = 0; j < nb; j++)
        fprintf(f, "1 %d\n", j);
    fprintf(f, "POINT_DATA %d\n", nb);
    vtk_scalar(f, "cp", "double");
    for (int j = 0; j < nb; j++)
        fprintf(f, "%.9g\n", pt[j].cp);
    vtk_scalar(f, "cp_wall", "double");
    for (int j = 0; j < nb; j++)
        fprintf(f, "%.9g\n", pt[j].cpWall);
    vtk_scalar(f, "cf", "double");
    for (int j = 0; j < nb; j++)
        fprintf(f, "%.9g\n", pt[j].cf);
    vtk_scalar(f, "rho", "double");
    for (int j = 0; j < nb; j++)
        fprintf(f, "%.9g\n", pt[j].rho);
    vtk_scalar(f, "area", "double");
    for (int j = 0; j < nb; j++)
        fprintf(f, "%.9g\n", s->area[idx[j]]);
    vtk_scalar(f, "links", "int");
    for (int j = 0; j < nb; j++)
        fprintf(f, "%d\n", s->links[idx[j]]);
    vtk_scalar(f, "samples", "int");
    for (int j = 0; j < nb; j++)
        fprintf(f, "%ld\n", s->steps);
    const char *vec[] = {"force", "traction", "shear", "normal"};
    for (int v = 0; v < 4; v++) {
        fprintf(f, "VECTORS %s double\n", vec[v]);
        for (int j = 0; j < nb; j++) {
            const double *x = v == 0   ? pt[j].force
                              : v == 1 ? pt[j].traction
                              : v == 2 ? pt[j].shear
                                       : &s->normal[3 * idx[j]];
            fprintf(f, "%.9g %.9g %.9g\n", x[0], x[1], x[2]);
        }
    }
    free(pt);
    free(idx);
    int ok = !ferror(f);
    return (fclose(f) == 0 && ok) ? 0 : -1;
}

static int
write_mesh_vtk(const Surface *s, const char *path, double u, double rhoRef) {
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    int nt = s->ntri;
    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f,
            "Lattice surface, steps %ld, windows %d, U %.9g, rho_ref %.9g, "
            "lattice units\n",
            s->steps,
            s->windows,
            u,
            rhoRef);
    fprintf(f, "ASCII\nDATASET POLYDATA\nPOINTS %d float\n", 3 * nt);
    for (int t = 0; t < nt; t++)
        for (int v = 0; v < 3; v++)
            fprintf(f,
                    "%.9g %.9g %.9g\n",
                    s->tri[9 * t + 3 * v],
                    s->tri[9 * t + 3 * v + 1],
                    s->tri[9 * t + 3 * v + 2]);
    fprintf(f, "POLYGONS %d %d\n", nt, 4 * nt);
    for (int t = 0; t < nt; t++)
        fprintf(f, "3 %d %d %d\n", 3 * t, 3 * t + 1, 3 * t + 2);
    fprintf(f, "CELL_DATA %d\n", nt);

    // Triangles out of reach get 0 and cells 0
    double *val =
        (double *)malloc((nt > 0 ? (size_t)nt : 1) * 5 * sizeof(double));
    if (!val) {
        fclose(f);
        return -1;
    }
    for (int t = 0; t < nt; t++)
        surface_triangle(
            s, t, u, rhoRef, &val[5 * t], &val[5 * t + 1], &val[5 * t + 2]);
    vtk_scalar(f, "cp", "double");
    for (int t = 0; t < nt; t++)
        fprintf(f, "%.9g\n", val[5 * t]);
    vtk_scalar(f, "cp_wall", "double");
    for (int t = 0; t < nt; t++)
        fprintf(f, "%.9g\n", val[5 * t + 1]);
    vtk_scalar(f, "cells", "int");
    for (int t = 0; t < nt; t++)
        fprintf(f, "%d\n", s->wStart[t + 1] - s->wStart[t]);
    vtk_scalar(f, "samples", "int");
    for (int t = 0; t < nt; t++)
        fprintf(f, "%ld\n", s->steps);
    fprintf(f, "VECTORS traction double\n");
    for (int t = 0; t < nt; t++)
        fprintf(f,
                "%.9g %.9g %.9g\n",
                val[5 * t + 2],
                val[5 * t + 3],
                val[5 * t + 4]);
    // Mean shear of the nearby cells
    fprintf(f, "VECTORS shear double\n");
    for (int t = 0; t < nt; t++) {
        double sh[3] = {0, 0, 0}, wsum = 0;
        for (int i = s->wStart[t]; i < s->wStart[t + 1]; i++) {
            SurfacePoint p;
            surface_point(s, s->wEntry[i], u, rhoRef, &p);
            if (!p.wall)
                continue;
            for (int d = 0; d < 3; d++)
                sh[d] += s->wShare[i] * p.shear[d];
            wsum += s->wShare[i];
        }
        for (int d = 0; d < 3 && wsum > 0; d++)
            sh[d] /= wsum;
        fprintf(f, "%.9g %.9g %.9g\n", sh[0], sh[1], sh[2]);
    }
    fprintf(f, "NORMALS normal double\n");
    for (int t = 0; t < nt; t++)
        fprintf(f,
                "%.9g %.9g %.9g\n",
                s->triNormal[3 * t],
                s->triNormal[3 * t + 1],
                s->triNormal[3 * t + 2]);
    free(val);
    int ok = !ferror(f);
    return (fclose(f) == 0 && ok) ? 0 : -1;
}

static int write_plane_csv(const Surface *s,
                           const char *path,
                           double u,
                           double rhoRef,
                           int side,
                           const double centre[3]) {
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    // Third axis of the plane, besides x
    int lift = (side == 1) ? 2 : 1;
    fprintf(f,
            "x,y,z,theta_deg,cp,cp_wall,cf,traction_x,traction_y,traction_z,"
            "normal_x,normal_y,normal_z,area,links,samples\n");
    for (int k = 0; k < s->n; k++) {
        if (!(s->role[k] & SURF_BOUNDARY))
            continue;
        double c[3];
        centre_of(s, k, c);
        if (fabs(c[side] - centre[side]) > 0.5 + 1e-9)
            continue;
        SurfacePoint p;
        surface_point(s, k, u, rhoRef, &p);
        // From upstream, positive toward +lift
        double theta =
            atan2(c[lift] - centre[lift], centre[0] - c[0]) * 180.0 / M_PI;
        const double *n = &s->normal[3 * k];
        fprintf(f,
                "%.6f,%.6f,%.6f,%.4f,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.6f,%.6f,"
                "%.6f,%.9g,%d,%ld\n",
                c[0],
                c[1],
                c[2],
                theta,
                p.cp,
                p.cpWall,
                p.cf,
                p.traction[0],
                p.traction[1],
                p.traction[2],
                n[0],
                n[1],
                n[2],
                s->area[k],
                s->links[k],
                s->steps);
    }
    int ok = !ferror(f);
    return (fclose(f) == 0 && ok) ? 0 : -1;
}

int surface_write(const Surface *s,
                  const char *path,
                  double u,
                  int sideAxis,
                  const double centre[3]) {
    size_t len = strlen(path);
    char *stem = (char *)malloc(len + 16);
    if (!stem)
        return -1;
    memcpy(stem, path, len + 1);
    if (len > 4 && strcmp(stem + len - 4, ".vtk") == 0)
        stem[len - 4] = '\0';
    size_t sl = strlen(stem);
    char *cells = (char *)malloc(sl + 16), *csv = (char *)malloc(sl + 16);
    if (!cells || !csv) {
        free(stem);
        free(cells);
        free(csv);
        return -1;
    }
    snprintf(cells, sl + 16, "%s_cells.vtk", stem);
    snprintf(csv, sl + 16, "%s_plane.csv", stem);
    double rhoRef = surface_reference_rho(s);
    int rc = 0;
    if (s->ntri > 0)
        rc |= write_mesh_vtk(s, path, u, rhoRef);
    else
        rc |= write_cells_vtk(s, path, u, rhoRef);
    rc |= write_cells_vtk(s, cells, u, rhoRef);
    rc |= write_plane_csv(s, csv, u, rhoRef, sideAxis, centre);
    free(stem);
    free(cells);
    free(csv);
    return rc ? -1 : 0;
}
