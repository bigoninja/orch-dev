#!/usr/bin/env python3
"""Fixture check helper: exit 0 when every named file contains the marker ``OK``.

Used by the end-to-end plans as a portable stand-in for a real project's test
command; deliberately tiny so a failing check is obvious in its evidence log.
"""

import sys
from pathlib import Path


def main(names: list[str]) -> int:
    for name in names:
        if "OK" not in Path(name).read_text(encoding="utf-8"):
            print(f"not OK: {name}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
