#!/usr/bin/env python3
"""Regenerate tests/fixtures/mock.sqlite (the hermetic CI fixture).

Requires a CUDA-capable host with `nsys` on PATH. Produces a minimal profile
(1 tiny kernel, no NCCL, no NVTX) — just enough to exercise nsys-ai CLI
wiring in CI without depending on user-local ~100 MB profiles.

CI (`.github/workflows/plugin-smoke.yml`) loads the committed fixture; this
script is run manually on a CUDA host when the nsys schema changes.

Usage:
    python scripts/build_fixture.py              # actually build
    python scripts/build_fixture.py --dry-run    # print commands only
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures"
FIXTURE = FIXTURE_DIR / "mock.sqlite"
REP = FIXTURE_DIR / "mock.nsys-rep"

SCRIPT = "import torch; x = torch.zeros(16, device='cuda'); x.sum().item()"


def build_commands() -> list[list[str]]:
    # Use the current interpreter so CUDA-enabled venvs (e.g. .venv-gpu) carry through.
    # Minimal-flag capture keeps the fixture small (target: a few hundred KB):
    #   --sample=none       skip CPU IP sampling / symbol downloads
    #   --cpuctxsw=none     skip context-switch traces
    #   --trace=cuda        CUDA only; no OS, NVTX, or OpenGL tracing
    return [
        ["nsys", "profile",
         "-o", str(REP.with_suffix("")),
         "--force-overwrite=true",
         "--sample=none",
         "--cpuctxsw=none",
         "--trace=cuda",
         sys.executable, "-c", SCRIPT],
        ["nsys", "export", "--type=sqlite", "--force-overwrite=true",
         f"--output={FIXTURE}", str(REP)],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands without executing")
    args = parser.parse_args()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    cmds = build_commands()

    if args.dry_run:
        print(f"# Would write fixture to {FIXTURE}")
        for cmd in cmds:
            print(shlex.join(cmd))
        return 0

    for cmd in cmds:
        print(f"$ {shlex.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"command failed with exit {result.returncode}", file=sys.stderr)
            return result.returncode

    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
