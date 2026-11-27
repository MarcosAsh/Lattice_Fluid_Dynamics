"""Every script imports cleanly, so a broken dependency or typo shows up early"""

import importlib

import pytest

CORE = ["voxelize", "preprocess", "evaluate", "data_gen", "deploy_weights", "ffd", "gen_shapes"]
TORCH = [
    "train_shapeopt",
    "optimize",
    "train_superres",
    "evaluate_superres",
    "export_weights",
]


@pytest.mark.parametrize("name", CORE)
def test_core_scripts_import(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", TORCH)
def test_torch_scripts_import(name):
    pytest.importorskip("torch")
    importlib.import_module(name)


def test_gen_pairs_imports():
    mod = importlib.import_module("gen_pairs")
    assert callable(mod.write_srbin)
    assert callable(mod.read_srbin)
    assert callable(mod.extract_patches)
    assert callable(mod.downsample_2x)


def test_modal_scripts_import():
    pytest.importorskip("modal")
    importlib.import_module("modal_worker")
