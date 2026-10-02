"""Run from an installed checkout: python examples/refine_api.py INPUT OUTPUT [--upscale]."""
import argparse

from rastermoves import Upscaler
from rastermoves.refinement import Refiner, RefineOptions, run_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--upscale", action="store_true")
    parser.add_argument("--mask")
    parser.add_argument("--refiner", choices=["sd15-tile", "sdxl-tile"], default="sd15-tile")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    options = RefineOptions(strength=0.22, steps=40, seed=args.seed)
    with Refiner(args.refiner, options=options) as refiner:
        if args.upscale:
            with Upscaler("4x-realesr-general-x4v3") as upscaler:
                result = run_workflow(args.input, args.output, refiner, upscaler=upscaler,
                                      protect_mask=args.mask, image_options={"tile": 256})
        else:
            result = refiner.refine_file(args.input, args.output, protect_mask=args.mask)
        print(result)
        print(refiner.last_report)


if __name__ == "__main__":
    main()
