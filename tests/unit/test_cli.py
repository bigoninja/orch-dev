from orch import __version__
from orch.cli import main


def test_version_command(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == f"orch {__version__}"


def test_missing_command_exits_nonzero(capsys):
    try:
        main([])
    except SystemExit as e:  # argparse usage error
        assert e.code == 2
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit")
