import json

import numpy as np
import pytest

import mesh_convert as mc
import preprocess as pp
from meshes import cube
from meshes import cube_obj_text

TETRA = (
    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
)


def write_binary_stl(path, corners, header=b"binary stl"):
    rec = np.zeros(len(corners), mc.STL_RECORD)
    rec["v"] = corners
    with path.open("wb") as f:
        f.write(header.ljust(mc.STL_HEADER, b" ")[: mc.STL_HEADER])
        f.write(np.uint32(len(corners)).tobytes())
        f.write(rec.tobytes())


def write_ascii_stl(path, corners, name="test"):
    lines = [f"solid {name}"]
    for tri in corners:
        lines += ["  facet normal 0 0 0", "    outer loop"]
        lines += [f"      vertex {x:.9g} {y:.9g} {z:.9g}" for x, y, z in tri]
        lines += ["    endloop", "  endfacet"]
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n")


def soup(verts, faces):
    return np.asarray(verts)[np.asarray(faces)]


def tri_set(verts, tris):
    """Triangles as sorted corner tuples, independent of vertex numbering"""
    return sorted(tuple(sorted(map(tuple, np.round(verts[t], 6)))) for t in tris)


MESHES = {"cube": cube(), "tetra": TETRA}
WRITERS = {"binary": write_binary_stl, "ascii": write_ascii_stl}


@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("kind", WRITERS)
def test_stl_round_trip(tmp_path, mesh, kind):
    verts, faces = MESHES[mesh]
    src = tmp_path / f"{mesh}.stl"
    WRITERS[kind](src, soup(verts, faces))
    assert mc.stl_kind(src) == kind

    report = mc.convert_file(src, tmp_path / "out.obj", check=True)
    assert report["format"] == f"stl-{kind}"
    assert report["vertices_out"] == len(verts)
    assert report["triangles_out"] == len(faces)
    assert report["validation"]["watertight"]

    out_v, out_f = pp.load_obj(tmp_path / "out.obj")
    out_t = np.asarray(out_f)
    assert out_t.min() >= 0
    assert tri_set(out_v, out_t) == tri_set(verts, np.asarray(faces))
    for line in (tmp_path / "out.obj").read_text().splitlines():
        if line.startswith("f "):
            assert all(int(i) > 0 for i in line.split()[1:])


def test_binary_with_solid_header_is_binary(tmp_path):
    verts, faces = cube()
    src = tmp_path / "trap.stl"
    write_binary_stl(src, soup(verts, faces), header=b"solid cube exported by CAD")
    assert mc.stl_kind(src) == "binary"
    report = mc.convert_file(src, None)
    assert report["format"] == "stl-binary"
    assert report["triangles_out"] == 12


def test_degenerate_triangles_are_dropped():
    verts, faces = cube()
    extra = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],  # collinear
            [[0, 0, 0], [0, 0, 0], [0, 1, 0]],  # repeated corner
        ],
        dtype=np.float64,
    )
    corners = np.concatenate([soup(verts, faces), extra])
    _, tris, report = mc.convert(corners)
    assert report["degenerate_dropped"] == 2
    assert report["triangles_out"] == len(tris) == 12
    assert report["vertices_out"] == 8


def test_merge_count_and_tolerance():
    verts, faces = cube()
    corners = soup(verts, faces)
    assert mc.convert(corners)[2]["vertices_out"] == 8
    assert mc.convert(corners)[2]["corners_in"] == 36

    jittered = corners + np.random.default_rng(0).uniform(-1e-7, 1e-7, corners.shape)
    assert mc.convert(jittered)[2]["vertices_out"] > 8
    assert mc.convert(jittered, merge_tol=1e-4)[2]["vertices_out"] == 8


def test_negative_zero_merges():
    pts = np.array([[0.0, 1.0, 2.0], [-0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    merged, idx = mc.merge_vertices(pts)
    assert len(merged) == 2
    assert idx.tolist() == [0, 0, 1]


def test_truncated_binary_is_a_clear_error(tmp_path):
    verts, faces = cube()
    src = tmp_path / "cut.stl"
    write_binary_stl(src, soup(verts, faces), header=b"solid but binary")
    src.write_bytes(src.read_bytes()[:-30])
    with pytest.raises(mc.MeshError, match="truncated, header says 12 triangles"):
        mc.convert_file(src, None)


def test_truncated_ascii_is_a_clear_error(tmp_path):
    verts, faces = cube()
    src = tmp_path / "cut.stl"
    write_ascii_stl(src, soup(verts, faces))
    text = src.read_text()
    src.write_text(text[: text.rfind("vertex")])
    with pytest.raises(mc.MeshError, match="not a multiple of 3"):
        mc.convert_file(src, None)

    lines = text.splitlines()
    src.write_text("\n".join(lines[: lines.index("  endfacet", 20)]) + "\n")
    with pytest.raises(mc.MeshError, match="no endsolid"):
        mc.convert_file(src, None)


def test_tiny_file_is_rejected(tmp_path):
    src = tmp_path / "tiny.stl"
    src.write_bytes(b"\0" * 10)
    with pytest.raises(mc.MeshError, match="too short"):
        mc.convert_file(src, None)


def test_scale_and_center(tmp_path):
    verts, faces = cube(h=1.0, center=(10.0, 0.0, -5.0))
    _, _, report = mc.convert(soup(verts, faces), scale=0.5, center=True)
    assert report["bbox_min"] == pytest.approx([-0.5, -0.5, -0.5])
    assert report["bbox_max"] == pytest.approx([0.5, 0.5, 0.5])


def test_obj_quads_and_negative_indices(tmp_path):
    src = tmp_path / "quads.obj"
    src.write_text(cube_obj_text())
    report = mc.convert_file(src, tmp_path / "tri.obj", check=True)
    assert report["format"] == "obj"
    assert report["quads_split"] == 6
    assert report["triangles_out"] == 12
    assert report["validation"]["watertight"]

    rel = tmp_path / "rel.obj"
    rel.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\nv 9 9 9\n")
    out = tmp_path / "rel_out.obj"
    report = mc.convert_file(rel, out)
    assert report["vertices_out"] == 3
    assert "f 1 2 3" in out.read_text()


def test_obj_bad_index(tmp_path):
    src = tmp_path / "bad.obj"
    src.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 7\n")
    with pytest.raises(mc.MeshError, match="out of range"):
        mc.convert_file(src, None)


def test_main_writes_obj_and_json(tmp_path, capsys):
    verts, faces = TETRA
    src = tmp_path / "tet.stl"
    write_binary_stl(src, soup(verts, faces))
    assert mc.main([str(src), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["triangles_out"] == 4
    assert (tmp_path / "tet.obj").is_file()


def test_main_refuses_to_overwrite_input(tmp_path):
    src = tmp_path / "a.obj"
    src.write_text(cube_obj_text())
    with pytest.raises(SystemExit):
        mc.main([str(src)])


def test_main_reports_errors(tmp_path, capsys):
    assert mc.main([str(tmp_path / "missing.stl")]) == 1
    assert "no such file" in capsys.readouterr().err
