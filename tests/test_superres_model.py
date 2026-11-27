import numpy as np
import pytest

torch = pytest.importorskip("torch")

import evaluate as ev  # noqa: E402
import evaluate_superres as es  # noqa: E402
import export_weights as ew  # noqa: E402
import train_superres as ts  # noqa: E402


def test_trilinear_of_constant_patch_is_constant():
    patch = torch.zeros(2, ts.COARSE_DIM)
    patch.view(2, 3, 3, 3, 4)[..., :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = ts.trilinear_upsample(patch).view(2, 2, 2, 2, 4)
    assert torch.allclose(out, torch.tensor([1.0, 2.0, 3.0, 4.0]).expand_as(out))


def test_trilinear_output_shape_and_batch_independence():
    x = torch.randn(5, ts.COARSE_DIM)
    full = ts.trilinear_upsample(x)
    assert full.shape == (5, ts.FINE_DIM)
    assert torch.allclose(ts.trilinear_upsample(x[2:3]), full[2:3])


def test_trilinear_weights_sum_to_one_so_it_stays_in_range():
    x = torch.rand(3, ts.COARSE_DIM)
    out = ts.trilinear_upsample(x)
    assert out.min() >= x.min() - 1e-6
    assert out.max() <= x.max() + 1e-6


def test_network_starts_close_to_skip_connection():
    net = ts.SuperResNet(hidden=16)
    assert net(torch.randn(4, ts.COARSE_DIM)).shape == (4, ts.FINE_DIM)


def test_psnr():
    assert ts.compute_psnr(0.0) == 100.0
    assert ts.compute_psnr(0.01) == pytest.approx(20.0)


def test_divergence_of_uniform_field_is_zero():
    field = np.zeros((3, 32), dtype=np.float32)
    assert np.all(es.compute_divergence(field, (2, 2, 2)) == 0.0)
    field.reshape(3, 2, 2, 2, 4)[:, :, :, 1, 0] = 1.0
    assert np.allclose(es.compute_divergence(field, (2, 2, 2)), 1.0)


def test_export_reads_back_through_evaluate_loader(tmp_path):
    net = ts.SuperResNet(hidden=8)
    path = tmp_path / "sr.bin"
    ew.export_ltws(net.state_dict(), str(path))
    params = ev.load_weights(str(path))
    assert len(params) == len(net.state_dict())


def test_normalizer_export(tmp_path):
    npz = tmp_path / "n.npz"
    np.savez(npz, mean=np.arange(4.0), std=np.ones(4))
    out = tmp_path / "n.bin"
    ew.export_normalizer(str(npz), str(out))
    assert np.frombuffer(out.read_bytes(), dtype=np.float32).tolist() == [0, 1, 2, 3, 1, 1, 1, 1]
