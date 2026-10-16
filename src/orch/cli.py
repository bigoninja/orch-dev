import argparse
import sys

from orch import __version__


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"orch {__version__}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    from pathlib import Path

    from orch.config import init_config
    from orch.errors import ConfigError

    try:
        cfg = init_config(Path(args.target_repo), Path(args.config))
    except ConfigError as e:
        print(f"error: {e}")
        return 1
    Path(cfg.state_dir).mkdir(exist_ok=True)
    Path(cfg.runs_dir).mkdir(exist_ok=True)
    print(f"wrote {args.config}\nnext: orch plan \"<brief>\"")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orch")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("version", help="print version")
    v.set_defaults(fn=cmd_version)
    i = sub.add_parser("init", help="write orch.config.yaml for a target repo")
    i.add_argument("target_repo", help="path to the target git repository")
    i.add_argument("--config", default="orch.config.yaml", help="path to write the config")
    i.set_defaults(fn=cmd_init)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
