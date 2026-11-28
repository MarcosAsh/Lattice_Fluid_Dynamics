#ifndef SURFACE_H
#define SURFACE_H

// Time-averaged momentum exchange and density on the body surface, in
// lattice units, cell (x, y, z) centred at (x + 0.5, y + 0.5, z + 0.5)

#define SURF_BOUNDARY  1 // fluid cell with at least one Bouzidi link
#define SURF_REFERENCE 2 // cell of the upstream plane that sets p_inf

// Cell forces are spread over the triangles within this many cells
#define SURF_KERNEL_RADIUS 2.0
// Triangles with a longer edge, in cells, are split before mapping
#define SURF_MAX_EDGE 0.75
// Cells whose links see less area, in cells^2, get no wall values
#define SURF_MIN_LINK_AREA 0.25

typedef struct {
    int nx, ny, nz;
    int n;               // listed cells
    int *cell;           // global indices, ascending
    unsigned char *role; // SURF_* bits
    int *w36;            // 3 per entry, 36 times the sum of 2 w_i e_i
    int *links;          // Bouzidi links per entry

    // Mesh mapping, area 0 without a mesh
    double *area;   // surface area each boundary cell stands for
    double *normal; // 3 per entry, out of the body
    int ntri;
    float *tri; // 9 per triangle, after splitting
    double *triArea;
    double *triNormal; // 3 per triangle, out of the body
    int *wStart;       // ntri + 1 offsets into the pairs below
    int *wEntry;       // cells near each triangle
    double *wGive;     // share of the cell force the triangle gets
    double *wShare;    // share of the triangle area the cell gets
    int unowned;       // triangles with no boundary cell in reach

    // Sums over the sampled steps
    double *sumF; // 3 per entry, without the 2 w_i e_i part
    double *sumRho;
    long steps;
    int windows;
    double dragSum[3]; // total force readback times window steps
} Surface;

typedef struct {
    double rho;
    double force[3];    // time-averaged momentum exchange of the cell
    double traction[3]; // force / area
    double shear[3];    // tangential force over the link area
    double cp;          // from the cell density
    double cpWall;      // normal force over the link area
    double cf;          // |shear| / (0.5 U^2)
    int mapped;         // area > 0
    int wall;           // link area >= SURF_MIN_LINK_AREA
} SurfacePoint;

// Takes ownership of the arrays. 0 on success
int surface_init(Surface *s,
                 int nx,
                 int ny,
                 int nz,
                 int n,
                 int *cell,
                 unsigned char *role,
                 int *w36,
                 int *links);
void surface_free(Surface *s);

// Index of a global cell in the list, -1 if absent
int surface_find(const Surface *s, int cell);

// Spreads each boundary cell force over the triangles within
// SURF_KERNEL_RADIUS with weights area * (1 - d / R)^2, so the forces on
// the triangles add up to the cell forces. tris are world coordinates, 12
// floats per triangle as LBM_SetSolidMesh takes them
int surface_map_mesh(Surface *s, const float *tris, int ntri);

// Adds one sampling window, 4 floats per entry (force, density) summed
// over steps, and the drag readback of the same window
void surface_add_window(Surface *s,
                        const float *sums,
                        int steps,
                        const float drag[3]);

// Mean density of the reference plane
double surface_reference_rho(const Surface *s);

// Point values of entry k, u the inlet velocity
void surface_point(
    const Surface *s, int k, double u, double rhoRef, SurfacePoint *p);

// Values on triangle t from the nearby cells: cp from their densities,
// cp_wall from their normal force over their link sums, traction from
// the force they give it. 0 if no cell is in reach
int surface_triangle(const Surface *s,
                     int t,
                     double u,
                     double rhoRef,
                     double *cp,
                     double *cpWall,
                     double traction[3]);

// Totals over boundary cells, over the mesh (traction times area) and the
// mean drag readback. unmapped is the force of cells without area
void surface_totals(const Surface *s,
                    double cells[3],
                    double mesh[3],
                    double drag[3],
                    double unmapped[3]);

// Writes PATH (the mesh if mapped, else the cells), STEM_cells.vtk and
// STEM_plane.csv, STEM being PATH without .vtk. The plane passes through
// centre (lattice cells) normal to sideAxis. 0 on success
int surface_write(const Surface *s,
                  const char *path,
                  double u,
                  int sideAxis,
                  const double centre[3]);

#endif // SURFACE_H
