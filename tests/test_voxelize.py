import numpy as np
import pytest

import voxelize as vx
from meshes import cube
from meshes import cube_obj_text


@pytest.fixture
def cube_vf():
    verts, faces = cube()
    return verts, np.asarray(faces, dtype=np.int32)


def test_solid_cube_is_mostly_solid(cube_vf):
    g = vx.voxelize_mesh(*cube_vf, 16, vx.MODE_SOLID)
    assert g.shape == (16, 16, 16)
    assert 0.80 < vx.solid_fraction(g) <= 1.0


def test_surface_cube_is_hollow(cube_vf):
    g = vx.voxelize_mesh(*cube_vf, 16, vx.MODE_SURFACE)
    assert 0.20 < vx.solid_fraction(g) < 0.95
    assert g[8, 8, 8] == 0


def test_longest_axis_goes_to_x(cube_vf):
    _, faces = cube_vf
    verts = np.array(
        [
            [-0.1, -0.2, -0.8],
            [0.1, -0.2, -0.8],
            [0.1, 0.2, -0.8],
            [-0.1, 0.2, -0.8],
            [-0.1, -0.2, 0.8],
            [0.1, -0.2, 0.8],
            [0.1, 0.2, 0.8],
            [-0.1, 0.2, 0.8],
        ]
    )
    g = vx.voxelize_mesh(verts, faces, 32, vx.MODE_SOLID)
    xs = np.nonzero(g.any(axis=(1, 2)))[0]
    zs = np.nonzero(g.any(axis=(0, 1)))[0]
    assert xs.max() - xs.min() > zs.max() - zs.min()


def test_no_align_keeps_axes(cube_vf):
    _, faces = cube_vf
    verts = np.array(cube_vf[0]) * np.array([0.2, 0.3, 1.0])
    g = vx.voxelize_mesh(verts, faces, 32, vx.MODE_SOLID, align_longest_x=False)
    zs = np.nonzero(g.any(axis=(0, 1)))[0]
    xs = np.nonzero(g.any(axis=(1, 2)))[0]
    assert zs.max() - zs.min() > xs.max() - xs.min()


def test_deterministic(cube_vf):
    a = vx.voxelize_mesh(*cube_vf, 16, vx.MODE_SOLID)
    b = vx.voxelize_mesh(*cube_vf, 16, vx.MODE_SOLID)
    assert np.array_equal(a, b)


def test_quad_is_fan_triangulated():
    _, tris = vx.load_obj_triangles("v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nf 1 2 3 4\n")
    assert tris.shape == (2, 3)


def test_obj_index_forms():
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2//2 -1\n"
    verts, tris = vx.load_obj_triangles(text)
    assert verts.shape == (3, 3)
    assert tris.tolist() == [[0, 1, 2]]


def test_auto_falls_back_to_surface_for_open_sheet():
    tilted = "v -1 -1 -1\nv 1 -1 -1\nv 1 1 1\nv -1 1 1\nf 1 2 3 4\n"
    verts, tris = vx.load_obj_triangles(tilted)
    assert vx.solid_fraction(vx.voxelize_mesh(verts, tris, 16, vx.MODE_SOLID)) < 0.001
    assert vx.solid_fraction(vx.voxelize_mesh(verts, tris, 16, vx.MODE_AUTO)) > 0.0


def test_voxelize_obj_accepts_text_bytes_and_path(tmp_path):
    text = cube_obj_text()
    from_text = vx.voxelize_obj(text, 12)
    from_bytes = vx.voxelize_obj(text.encode(), 12)
    path = tmp_path / "cube.obj"
    path.write_text(text)
    from_path = vx.voxelize_obj(path, 12)
    assert np.array_equal(from_text, from_bytes)
    assert np.array_equal(from_text, from_path)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"resolution": 0}, "resolution"),
        ({"mode": "bogus"}, "mode"),
    ],
)
def test_bad_arguments(cube_vf, kwargs, match):
    with pytest.raises(ValueError, match=match):
        vx.voxelize_mesh(*cube_vf, **kwargs)


def test_empty_and_degenerate_meshes():
    with pytest.raises(ValueError, match="no triangles"):
        vx.voxelize_mesh(np.zeros((3, 3)), np.zeros((0, 3), dtype=np.int32))
    with pytest.raises(ValueError, match="zero extent"):
        vx.voxelize_mesh(np.zeros((3, 3)), np.array([[0, 1, 2]]))


def test_canonicalize_fits_padding():
    v = vx.canonicalize(np.array([[0.0, 0, 0], [4.0, 1, 1]]), padding=0.1)
    assert v[:, 0].max() == pytest.approx(0.9)
    assert v[:, 0].min() == pytest.approx(-0.9)
    assert vx.canonicalize(np.zeros((0, 3))) is None


def test_voxbin_round_trip_keeps_byte_order(tmp_path):
    probe = np.zeros((8, 8, 8), dtype=np.uint8)
    probe[1, 2, 3] = 1
    path = tmp_path / "probe.voxbin"
    vx.write_voxbin(probe, path)
    raw = path.read_bytes()
    assert raw[:4] == b"VOXB"
    # x + R*(y + R*z) ordering
    assert raw[12 + 1 + 8 * (2 + 8 * 3)] == 1
    assert np.array_equal(vx.read_voxbin(path), probe)


def test_voxbin_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError, match="cubic"):
        vx.write_voxbin(np.zeros((2, 3, 4), dtype=np.uint8), tmp_path / "x.voxbin")
    bad = tmp_path / "bad.voxbin"
    bad.write_bytes(b"NOPE" + bytes(20))
    with pytest.raises(ValueError, match="magic"):
        vx.read_voxbin(bad)
    good = tmp_path / "trunc.voxbin"
    vx.write_voxbin(np.zeros((4, 4, 4), dtype=np.uint8), good)
    good.write_bytes(good.read_bytes()[:-5])
    with pytest.raises(ValueError, match="truncated"):
        vx.read_voxbin(good)


def test_cli_writes_output(tmp_path, capsys):
    src = tmp_path / "cube.obj"
    src.write_text(cube_obj_text())
    out = tmp_path / "cube.voxbin"
    assert vx.main([str(src), "--resolution", "8", "--output", str(out)]) == 0
    assert out.exists()
    assert "voxelized" in capsys.readouterr().out
