import struct

import deploy_weights as dw


def test_ltws_header_check(tmp_path):
    good = tmp_path / "g.bin"
    good.write_bytes(struct.pack("<I", dw.LTWS_MAGIC) + b"rest")
    bad = tmp_path / "b.bin"
    bad.write_bytes(b"nope")
    short = tmp_path / "s.bin"
    short.write_bytes(b"ab")
    assert dw._has_ltws_header(good)
    assert not dw._has_ltws_header(bad)
    assert not dw._has_ltws_header(short)


def test_md5_is_short_and_content_based(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    assert len(dw._md5(a)) == 12
    assert dw._md5(a) != dw._md5(b)
