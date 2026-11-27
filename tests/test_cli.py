from pathlib import Path

import pytest

import cli

ML = Path(__file__).resolve().parents[1] / "ml"


def test_every_command_points_at_a_script():
    for name, (script, desc) in cli.COMMANDS.items():
        assert (ML / script).is_file(), name
        assert desc


def test_no_arguments_lists_commands(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in cli.COMMANDS)
    assert "test" in out


def test_unknown_command(capsys):
    assert cli.main(["nope"]) == 2
    assert "unknown command 'nope'" in capsys.readouterr().err


def test_dispatch_passes_arguments_and_exit_code(capsys):
    assert cli.main(["check-config", str(ML / "sweep_config.yaml")]) == 0
    assert cli.main(["preprocess", "--help"]) == 0
    assert cli.main(["check-config", str(ML / "missing.yaml")]) == 1


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_flags(flag, capsys):
    assert cli.main([flag]) == 0
    assert "usage: lattice-ml" in capsys.readouterr().out
