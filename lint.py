#Credit to Nolan Chai

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> int:
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: uv run python lint.py [folder]")
        raise SystemExit(2)

    target = sys.argv[1] if len(sys.argv) == 2 else "epiaudio"

    exit_code = 0
    for command in (
        ["ruff", "check", target],
        ["pyright", target],
    ):
        exit_code = run(command) or exit_code

    raise SystemExit(exit_code)