"""Verify that committed outputs match a fresh, reproducible evaluation.

    python scripts/verify_outputs.py          # fast mode, about a minute
    python scripts/verify_outputs.py --full   # also re-runs the full nested CV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gradmit.verification import run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="re-run the full nested CV as well")
    args = parser.parse_args()
    for name in run_all(fast=not args.full):
        print(f"ok  {name}")
    print("Committed outputs match the reproducible evaluation.")


if __name__ == "__main__":
    main()
