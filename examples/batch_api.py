"""Run: python examples/batch_api.py input-directory output-directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from rastermoves import Upscaler
from rastermoves.cli import batch_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="4x-realesr-general-x4v3")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    jobs = batch_jobs(args.input, args.output, recursive=True)
    with Upscaler(args.model, device=args.device) as upscaler:
        for source, destination in jobs:
            print(f"{source} -> {destination}")
            upscaler.upscale_file(source, destination, report=True)


if __name__ == "__main__":
    main()
