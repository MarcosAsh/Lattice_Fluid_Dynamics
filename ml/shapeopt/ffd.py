"""
Free Form Deformation (FFD) lattice for 3D mesh shape parameterization.

Implements trivariate Bernstein polynomial interpolation over a regular
lattice of control points. Displacing control points smoothly deforms
the enclosed mesh, giving a low-dimensional parameterization suitable
for shape optimization and surrogate model training.

Usage:
    from ml.shapeopt.ffd import FFDLattice, load_obj, save_obj, deform_obj

    verts, faces = load_obj("body.obj")
    bbox_min, bbox_max = compute_bbox(verts)
    lattice = FFDLattice(bbox_min, bbox_max, n_control=(4, 3, 3))

    displacements = np.zeros(lattice.get_num_params())
    deformed = lattice.deform(verts, displacements)
    save_obj("body_deformed.obj", deformed, faces)
"""

import numpy as np
from pathlib import Path


# Bernstein basis

def _binomial_coeffs(n):
    """Row n of Pascal's triangle, computed iteratively to avoid scipy."""
    c = np.ones(n + 1, dtype=np.float64)
    for k in range(1, n + 1):
        c[k] = c[k - 1] * (n - k + 1) / k
    return c


def _bernstein_basis(n, t):
    """
    Evaluate all Bernstein basis polynomials of degree n at parameter values t.

    Returns array of shape (len(t), n+1) where column i is B(i, n, t).
    """
    t = np.asarray(t, dtype=np.float64)
    coeffs = _binomial_coeffs(n)
    # Shape: (len(t), n+1)
    i = np.arange(n + 1)
    basis = coeffs[None, :] * (t[:, None] ** i[None, :]) * ((1.0 - t[:, None]) ** (n - i)[None, :])
    return basis


# FFDLattice

class FFDLattice:
    """Trivariate Bernstein FFD lattice for deforming 3D meshes."""

    def __init__(self, bbox_min, bbox_max, n_control=(4, 3, 3)):
        """
        Create an FFD lattice around a bounding box.

        Args:
            bbox_min: (3,) array, min corner of bounding box.
            bbox_max: (3,) array, max corner of bounding box.
            n_control: (nx, ny, nz) number of control points per axis.
        """
        self.bbox_min = np.asarray(bbox_min, dtype=np.float64)
        self.bbox_max = np.asarray(bbox_max, dtype=np.float64)
        self.extent = self.bbox_max - self.bbox_min
        if np.any(self.extent <= 0):
            raise ValueError("bbox_max must be strictly greater than bbox_min on every axis")

        self.nx, self.ny, self.nz = int(n_control[0]), int(n_control[1]), int(n_control[2])
        self.n_control = (self.nx, self.ny, self.nz)

        # Precompute undeformed control point positions on a regular grid.
        # Layout: (nx, ny, nz, 3), stored in C order (x varies slowest).
        sx = np.linspace(0.0, 1.0, self.nx)
        sy = np.linspace(0.0, 1.0, self.ny)
        sz = np.linspace(0.0, 1.0, self.nz)
        gx, gy, gz = np.meshgrid(sx, sy, sz, indexing="ij")
        self._param_grid = np.stack([gx, gy, gz], axis=-1)  # (nx, ny, nz, 3)

        # Physical positions of undeformed control points
        self.control_points = self.bbox_min + self._param_grid * self.extent

    # Core deformation

    def deform(self, vertices, displacements):
        """
        Deform mesh vertices through the FFD lattice.

        Args:
            vertices: (N, 3) array of vertex positions.
            displacements: (nx*ny*nz*3,) flat array of control point
                displacements in physical space.

        Returns:
            (N, 3) array of deformed vertex positions.
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        displacements = np.asarray(displacements, dtype=np.float64).reshape(
            self.nx, self.ny, self.nz, 3
        )

        N = vertices.shape[0]
        l, m, n = self.nx - 1, self.ny - 1, self.nz - 1  # polynomial degrees

        # Map vertices to [0, 1]^3 parametric coordinates, clamped to stay
        # inside the lattice (vertices slightly outside the bbox due to
        # floating point are pulled to the boundary).
        s = np.clip((vertices[:, 0] - self.bbox_min[0]) / self.extent[0], 0.0, 1.0)
        t = np.clip((vertices[:, 1] - self.bbox_min[1]) / self.extent[1], 0.0, 1.0)
        u = np.clip((vertices[:, 2] - self.bbox_min[2]) / self.extent[2], 0.0, 1.0)

        # Bernstein bases: each is (N, degree+1)
        Bs = _bernstein_basis(l, s)  # (N, nx)
        Bt = _bernstein_basis(m, t)  # (N, ny)
        Bu = _bernstein_basis(n, u)  # (N, nz)

        # Displaced control points in physical space
        P = self.control_points + displacements  # (nx, ny, nz, 3)

        # Evaluate X(s,t,u) = sum_i sum_j sum_k B_i(s)*B_j(t)*B_k(u) * P_ijk
        # Using einsum for clarity and reasonable speed.
        # Bs: (N, nx)  Bt: (N, ny)  Bu: (N, nz)  P: (nx, ny, nz, 3)
        # Result: (N, 3)
        deformed = np.einsum("ni,nj,nk,ijkd->nd", Bs, Bt, Bu, P)

        return deformed

    # Symmetry

    def enforce_symmetry(self, displacements, axis=1):
        """
        Mirror displacements across the specified axis for bilateral symmetry.

        For the default axis=1 (Y), control points at y-index j and
        (ny-1-j) are paired. Their x and z displacements are averaged,
        and the y displacement of the mirror is negated. Control points
        on the symmetry plane (when ny is odd and j == ny//2) have their
        y displacement zeroed out.

        Args:
            displacements: (nx*ny*nz*3,) flat array.
            axis: symmetry axis (0=X, 1=Y, 2=Z). Default 1.

        Returns:
            (nx*ny*nz*3,) symmetrized displacement array.
        """
        d = np.asarray(displacements, dtype=np.float64).reshape(
            self.nx, self.ny, self.nz, 3
        ).copy()

        n_ax = self.n_control[axis]
        half = n_ax // 2

        # Build axis slicing helpers so the code works for any axis.
        def _slice(ax_idx):
            """Return a tuple of slices selecting ax_idx along `axis`."""
            s = [slice(None)] * 3
            s[axis] = ax_idx
            return tuple(s)

        for j in range(half):
            j_mirror = n_ax - 1 - j
            sj = _slice(j)
            sm = _slice(j_mirror)

            # Tangential axes (not the symmetry axis): average magnitudes
            for comp in range(3):
                if comp == axis:
                    # Symmetry axis component: negate for mirror
                    avg = (d[sj][..., comp] - d[sm][..., comp]) / 2.0
                    d[sj][..., comp] = avg
                    d[sm][..., comp] = -avg
                else:
                    # Tangential components: average keeping same sign
                    avg = (d[sj][..., comp] + d[sm][..., comp]) / 2.0
                    d[sj][..., comp] = avg
                    d[sm][..., comp] = avg

        # Midplane: zero out the symmetry-axis component
        if n_ax % 2 == 1:
            mid = _slice(half)
            d[mid][..., axis] = 0.0

        return d.ravel()

    # Parameter counts

    def get_num_params(self):
        """Total displacement DOFs: nx * ny * nz * 3."""
        return self.nx * self.ny * self.nz * 3

    def get_num_free_params(self):
        """
        Number of independent parameters after Y-symmetry reduction.

        Only the "lower half" of control points along Y (plus the midplane
        if ny is odd) are free. Each free control point has 3 components,
        but midplane points lose their Y DOF.
        """
        half = self.ny // 2
        n_half = self.nx * half * self.nz  # paired points
        free = n_half * 3

        if self.ny % 2 == 1:
            n_mid = self.nx * 1 * self.nz
            free += n_mid * 2  # midplane points lose the symmetry-axis DOF

        return free

    # Clamping

    def clamp_displacements(self, displacements, max_fraction=0.15):
        """
        Clamp each displacement component to +/- max_fraction * bbox extent
        along the corresponding axis.

        Args:
            displacements: (nx*ny*nz*3,) flat array.
            max_fraction: fraction of bbox extent used as limit.

        Returns:
            (nx*ny*nz*3,) clamped displacement array.
        """
        d = np.asarray(displacements, dtype=np.float64).reshape(
            self.nx, self.ny, self.nz, 3
        ).copy()
        limits = max_fraction * self.extent  # (3,)
        for comp in range(3):
            np.clip(d[..., comp], -limits[comp], limits[comp], out=d[..., comp])
        return d.ravel()

    # Random sampling (Latin Hypercube)

    def sample_random(self, n_samples, max_fraction=0.15, symmetric=True):
        """
        Latin Hypercube Sampling of the FFD displacement space.

        Args:
            n_samples: number of design samples to generate.
            max_fraction: fraction of bbox extent for displacement bounds.
            symmetric: if True, sample only free (half-lattice) parameters
                and expand them via enforce_symmetry.

        Returns:
            (n_samples, nx*ny*nz*3) array of displacement vectors.
        """
        if symmetric:
            n_dim = self.get_num_free_params()
        else:
            n_dim = self.get_num_params()

        limits = max_fraction * self.extent  # (3,)

        # LHS: for each dimension, create a stratified random permutation
        rng = np.random.default_rng()
        samples = np.empty((n_samples, n_dim), dtype=np.float64)

        for d in range(n_dim):
            perm = rng.permutation(n_samples)
            samples[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples

        # Scale from [0, 1] to [-limit, +limit] per axis.
        # We need a mapping from flat free-parameter index to axis.
        axis_map = self._build_free_param_axis_map() if symmetric else self._build_full_axis_map()
        for d in range(n_dim):
            ax = axis_map[d]
            samples[:, d] = samples[:, d] * 2.0 * limits[ax] - limits[ax]

        if not symmetric:
            return samples

        # Expand free params to full displacement vectors via symmetry
        result = np.empty((n_samples, self.get_num_params()), dtype=np.float64)
        for i in range(n_samples):
            full = self._expand_free_to_full(samples[i])
            result[i] = self.enforce_symmetry(full)
        return result

    def _build_full_axis_map(self):
        """Map from flat full-param index to axis (0, 1, or 2)."""
        # Layout: (nx, ny, nz, 3) in C order, so last axis is xyz component.
        n_pts = self.nx * self.ny * self.nz
        return np.tile(np.arange(3), n_pts)

    def _build_free_param_axis_map(self):
        """Map from flat free-param index to axis (0, 1, or 2)."""
        axes = []
        half = self.ny // 2
        # Lower-half points: 3 DOFs each
        for _ in range(self.nx * half * self.nz):
            axes.extend([0, 1, 2])
        # Midplane points (if ny is odd): 2 DOFs each (skip Y)
        if self.ny % 2 == 1:
            for _ in range(self.nx * 1 * self.nz):
                axes.extend([0, 2])
        return np.array(axes, dtype=int)

    def _expand_free_to_full(self, free_params):
        """
        Expand free (symmetry-reduced) parameters into a full displacement
        vector. The upper-half Y control points get zeros; enforce_symmetry
        will fill them in.
        """
        d = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)
        half = self.ny // 2
        idx = 0

        # Lower half
        for j in range(half):
            n_slice = self.nx * self.nz * 3
            d[:, j, :, :] = free_params[idx:idx + n_slice].reshape(self.nx, self.nz, 3)
            idx += n_slice

        # Midplane
        if self.ny % 2 == 1:
            mid_j = half
            n_slice = self.nx * self.nz * 2
            mid_vals = free_params[idx:idx + n_slice].reshape(self.nx, self.nz, 2)
            d[:, mid_j, :, 0] = mid_vals[:, :, 0]
            d[:, mid_j, :, 2] = mid_vals[:, :, 1]
            # Y component stays zero
            idx += n_slice

        return d.ravel()


# OBJ I/O

def load_obj(path):
    """
    Parse a Wavefront OBJ file.

    Returns:
        vertices: (N, 3) float64 array.
        faces: (M, 3) int array, 0-indexed.
    """
    vertices = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                # Handle formats: "f 1 2 3", "f 1/2 3/4 5/6", "f 1/2/3 ..."
                idx = []
                for p in parts:
                    idx.append(int(p.split("/")[0]) - 1)  # OBJ is 1-indexed
                # Triangulate if more than 3 verts (fan triangulation)
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def save_obj(path, vertices, faces):
    """
    Write a Wavefront OBJ file with 1-indexed faces.

    Args:
        path: output file path.
        vertices: (N, 3) array.
        faces: (M, 3) array of 0-indexed vertex indices.
    """
    with open(path, "w") as f:
        f.write("# Generated by ml/shapeopt/ffd.py\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def compute_bbox(vertices, padding=0.05):
    """
    Compute axis-aligned bounding box with padding.

    Args:
        vertices: (N, 3) array.
        padding: fractional padding added on each side.

    Returns:
        (bbox_min, bbox_max) as numpy arrays of shape (3,).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = vmax - vmin
    pad = padding * extent
    return vmin - pad, vmax + pad


def deform_obj(input_path, output_path, displacements, n_control=(4, 3, 3)):
    """
    Load an OBJ, apply FFD deformation, and write the result.

    Args:
        input_path: path to input OBJ file.
        output_path: path to write deformed OBJ.
        displacements: (nx*ny*nz*3,) flat displacement array.
        n_control: (nx, ny, nz) control point counts.
    """
    verts, faces = load_obj(input_path)
    bbox_min, bbox_max = compute_bbox(verts)
    lattice = FFDLattice(bbox_min, bbox_max, n_control=n_control)
    deformed = lattice.deform(verts, displacements)
    save_obj(output_path, deformed, faces)


# Self-tests

if __name__ == "__main__":
    import sys
    import tempfile

    passed = 0
    failed = 0

    def check(name, condition):
        global passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}")
            failed += 1

    # Unit cube vertices
    cube_verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 3, 7], [0, 7, 4],
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)

    # Test 1: Identity deformation
    print("Test 1: Zero displacements preserve vertex positions")
    bbox_min, bbox_max = compute_bbox(cube_verts, padding=0.0)
    lattice = FFDLattice(bbox_min, bbox_max, n_control=(4, 3, 3))
    zero_disp = np.zeros(lattice.get_num_params())
    deformed = lattice.deform(cube_verts, zero_disp)
    max_err = np.max(np.abs(deformed - cube_verts))
    check("max error < 1e-12", max_err < 1e-12)

    # Test 2: Uniform translation
    print("Test 2: Uniform displacement translates all vertices equally")
    lattice2 = FFDLattice(bbox_min, bbox_max, n_control=(2, 2, 2))
    shift = np.zeros((2, 2, 2, 3))
    shift[..., 0] = 0.5  # shift all control points +0.5 in X
    deformed2 = lattice2.deform(cube_verts, shift.ravel())
    expected = cube_verts.copy()
    expected[:, 0] += 0.5
    max_err2 = np.max(np.abs(deformed2 - expected))
    check("uniform X shift matches", max_err2 < 1e-12)

    # Test 3: Parameter counts
    print("Test 3: Parameter counts")
    lattice3 = FFDLattice([0, 0, 0], [1, 1, 1], n_control=(4, 3, 3))
    check("total params = 4*3*3*3 = 108", lattice3.get_num_params() == 108)
    # ny=3: half=1 paired, 1 midplane. Free = 4*1*3*3 + 4*1*3*2 = 36 + 24 = 60
    check("free params after Y-symmetry = 60", lattice3.get_num_free_params() == 60)

    # Test 4: Symmetry enforcement
    print("Test 4: Symmetry enforcement")
    rng = np.random.default_rng(42)
    raw = rng.standard_normal(lattice3.get_num_params()) * 0.01
    sym = lattice3.enforce_symmetry(raw, axis=1)
    sym_4d = sym.reshape(4, 3, 3, 3)

    # Paired points: j=0 and j=2 should have equal x,z and negated y
    x_match = np.allclose(sym_4d[:, 0, :, 0], sym_4d[:, 2, :, 0])
    z_match = np.allclose(sym_4d[:, 0, :, 2], sym_4d[:, 2, :, 2])
    y_neg = np.allclose(sym_4d[:, 0, :, 1], -sym_4d[:, 2, :, 1])
    check("x displacements mirrored (same sign)", x_match)
    check("z displacements mirrored (same sign)", z_match)
    check("y displacements mirrored (negated)", y_neg)

    # Midplane Y should be zero
    mid_y_zero = np.allclose(sym_4d[:, 1, :, 1], 0.0)
    check("midplane y displacement = 0", mid_y_zero)

    # Test 5: Clamping
    print("Test 5: Displacement clamping")
    big_disp = np.ones(lattice3.get_num_params()) * 10.0
    clamped = lattice3.clamp_displacements(big_disp, max_fraction=0.15)
    clamped_4d = clamped.reshape(4, 3, 3, 3)
    limit = 0.15 * lattice3.extent
    within = True
    for comp in range(3):
        if np.any(np.abs(clamped_4d[..., comp]) > limit[comp] + 1e-14):
            within = False
    check("all components within limits", within)

    # Test 6: OBJ round-trip
    print("Test 6: OBJ I/O round-trip")
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    save_obj(tmp_path, cube_verts, cube_faces)
    loaded_v, loaded_f = load_obj(tmp_path)
    check("vertices survive round-trip", np.allclose(loaded_v, cube_verts, atol=1e-7))
    check("faces survive round-trip", np.array_equal(loaded_f, cube_faces))
    Path(tmp_path).unlink()

    # Test 7: deform_obj convenience
    print("Test 7: deform_obj end-to-end")
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False, mode="w") as src:
        src_path = src.name
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False, mode="w") as dst:
        dst_path = dst.name
    save_obj(src_path, cube_verts, cube_faces)
    n_ctrl = (3, 3, 3)
    n_params = n_ctrl[0] * n_ctrl[1] * n_ctrl[2] * 3
    deform_obj(src_path, dst_path, np.zeros(n_params), n_control=n_ctrl)
    v_out, f_out = load_obj(dst_path)
    # With padding in compute_bbox the vertices won't be exactly identical
    # because parametric coords map through the padded box, but zero
    # displacement should still reconstruct the original positions.
    check("deform_obj identity preserves shape", np.allclose(v_out, cube_verts, atol=1e-10))
    Path(src_path).unlink()
    Path(dst_path).unlink()

    # Test 8: LHS sampling shape
    print("Test 8: Latin Hypercube Sampling")
    samples = lattice3.sample_random(10, max_fraction=0.15, symmetric=True)
    check("LHS output shape (10, 108)", samples.shape == (10, 108))
    # Each sample should be symmetric
    sym_ok = True
    for i in range(samples.shape[0]):
        resym = lattice3.enforce_symmetry(samples[i])
        if not np.allclose(samples[i], resym, atol=1e-12):
            sym_ok = False
            break
    check("all LHS samples are symmetric", sym_ok)

    # Test 9: Bernstein partition of unity
    print("Test 9: Bernstein basis partition of unity")
    t_vals = np.linspace(0, 1, 50)
    for deg in [2, 3, 5]:
        basis = _bernstein_basis(deg, t_vals)
        sums = basis.sum(axis=1)
        check(f"degree {deg} sums to 1", np.allclose(sums, 1.0, atol=1e-14))

    # Summary
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
