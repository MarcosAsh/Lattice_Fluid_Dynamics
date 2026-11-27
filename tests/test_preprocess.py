import json

import numpy as np
import pytest

import preprocess as pp
from meshes import cube
from meshes import cube_obj_text


@pytest.fixture
def cube_tris():
    verts, faces = cube()
    tris, _, _ = pp.triangulate(verts, faces)
    return verts, faces, tris


def test_cube_is_watertight(cube_tris):
    verts, _, tris = cube_tris
    val = pp.validate(verts, tris)
    assert val["watertight"]
    assert val["boundary_edges"] == 0
    assert val["non_manifold_edges"] == 0
    assert val["flipped_edges"] == 0
    assert val["components"] == 1
    assert val["degenerate_triangles"] == 0


def test_normals_point_outward(cube_tris):
    verts, _, tris = cube_tris
    _, fn, _ = pp.compute_normals(verts, tris)
    centroids = verts[tris].mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", fn, centroids) > 0)


def test_inward_winding_is_flipped(cube_tris):
    verts, _, tris = cube_tris
    out_tris, fn, _ = pp.compute_normals(verts, tris[:, ::-1].copy())
    assert np.all(np.einsum("ij,ij->i", fn, verts[out_tris].mean(axis=1)) > 0)


def test_quad_split_on_shorter_diagonal():
    verts = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=np.float64)
    tris, n_quads, n_ngons = pp.triangulate(verts, [[0, 1, 2, 3]])
    assert tris.shape[0] == 2
    assert (n_quads, n_ngons) == (1, 0)


def test_ngon_is_fanned():
    verts = np.zeros((5, 3))
    tris, n_quads, n_ngons = pp.triangulate(verts, [[0, 1, 2, 3, 4]])
    assert tris.shape[0] == 3
    assert (n_quads, n_ngons) == (0, 1)


def test_open_mesh_has_boundary_edges(cube_tris):
    verts, faces, _ = cube_tris
    open_faces = [f for f in faces if f not in ([4, 6, 5], [4, 7, 6])]
    tris, _, _ = pp.triangulate(verts, open_faces)
    val = pp.validate(verts, tris)
    assert val["boundary_edges"] > 0
    assert not val["watertight"]


def test_reversed_face_is_flagged(cube_tris):
    verts, faces, _ = cube_tris
    flipped = [list(f) for f in faces]
    flipped[0] = flipped[0][::-1]
    tris, _, _ = pp.triangulate(verts, flipped)
    assert pp.validate(verts, tris)["flipped_edges"] > 0


def test_degenerate_triangle_is_flagged(cube_tris):
    verts, _, tris = cube_tris
    bad = np.vstack([tris, [[0, 0, 1]]])
    assert pp.validate(verts, bad)["degenerate_triangles"] >= 1


def test_two_cubes_are_two_components(cube_tris):
    verts, faces, _ = cube_tris
    v2, f2 = cube(center=(5.0, 0.0, 0.0))
    both_v = np.vstack([verts, v2])
    both_f = faces + [[i + 8 for i in f] for f in f2]
    tris, _, _ = pp.triangulate(both_v, both_f)
    assert pp.validate(both_v, tris)["components"] == 2


def test_center_and_scale():
    raw = np.array([[1, 2, 3], [5, 2, 3], [1, 4, 3], [1, 2, 9]], dtype=np.float64)
    scaled, tf = pp.center_and_scale(raw, domain=2.0)
    lo, hi = scaled.min(axis=0), scaled.max(axis=0)
    assert np.allclose(0.5 * (lo + hi), 0.0)
    assert np.isclose((hi - lo).max(), 2.0)
    assert tf["scale"] == pytest.approx(2.0 / 6.0)


def test_center_and_scale_point_cloud_keeps_scale():
    _, tf = pp.center_and_scale(np.ones((3, 3)))
    assert tf["scale"] == 1.0


def test_pipeline_on_quad_cube():
    res = pp.preprocess(cube_obj_text())
    assert res["ok"]
    assert res["stats"]["quads_split"] == 6
    assert res["stats"]["triangles"] == 12
    assert res["validation"]["watertight"]


def test_pipeline_volume_matches_domain():
    res = pp.preprocess(cube_obj_text(), domain=2.0)
    assert res["stats"]["volume"] == pytest.approx(8.0)
    assert res["stats"]["surface_area"] == pytest.approx(24.0)


def test_pipeline_empty_mesh():
    res = pp.preprocess("# nothing here\nv 0 0 0\n")
    assert not res["ok"]
    assert res["warnings"] == ["mesh has no faces"]


def test_pipeline_warns_on_open_mesh():
    res = pp.preprocess("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    assert any("boundary" in w for w in res["warnings"])


def test_write_obj_round_trip(tmp_path):
    res = pp.preprocess(cube_obj_text())
    out = tmp_path / "clean.obj"
    pp.write_obj(out, res["vertices"], res["triangles"], res["vertex_normals"])
    again = pp.preprocess(out)
    assert again["ok"]
    assert again["stats"]["triangles"] == res["stats"]["triangles"]


def test_cli_json(tmp_path, capsys):
    src = tmp_path / "cube.obj"
    src.write_text(cube_obj_text())
    assert pp.main([str(src), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"]
    assert data["validation"]["watertight"]
