import random

import numpy as np
import pytest

import seeding


def test_same_seed_same_streams():
    g1 = seeding.seed_everything(5)
    a = (random.random(), np.random.rand(), g1.random())  # noqa: NPY002
    g2 = seeding.seed_everything(5)
    b = (random.random(), np.random.rand(), g2.random())  # noqa: NPY002
    assert a == b


def test_different_seed_differs():
    a = seeding.seed_everything(1).random()
    assert a != seeding.seed_everything(2).random()


def test_torch_is_seeded_when_present():
    torch = pytest.importorskip("torch")
    seeding.seed_everything(9)
    a = torch.rand(3)
    seeding.seed_everything(9)
    assert torch.equal(a, torch.rand(3))


@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_rejects_bad_seed(bad):
    with pytest.raises(ValueError, match="seed"):
        seeding.seed_everything(bad)


def test_trainer_seed_makes_training_repeatable(tmp_path, rng):
    import argparse
    import struct

    torch = pytest.importorskip("torch")
    import train_shapeopt as ts

    n, p = 30, 4
    disp = rng.standard_normal((n, p)).astype(np.float32)
    out = np.column_stack([disp.sum(axis=1), disp[:, 0]]).astype(np.float32)

    def run(folder):
        folder.mkdir()
        data = folder / "d.sobin"
        with data.open("wb") as f:
            f.write(struct.pack("<IIIII", ts.SOBIN_MAGIC, 1, n, p, 2))
            for d, o in zip(disp, out, strict=True):
                f.write(d.tobytes() + o.tobytes())
        args = argparse.Namespace(
            data=str(data),
            epochs=2,
            batch_size=10,
            lr=1e-3,
            patience=5,
            wind_speed=1.0,
            reynolds=1e3,
            seed=3,
        )
        ts.train(args)
        return ts.load_checkpoint(folder / "shapeopt_model.pt")["model_state_dict"]

    a, b = run(tmp_path / "a"), run(tmp_path / "b")
    assert all(torch.equal(a[k], b[k]) for k in a)
