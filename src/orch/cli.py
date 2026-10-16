import argparse
import sys

from orch import __version__


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"orch {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orch")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("version", help="print version")
    v.set_defaults(fn=cmd_version)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
