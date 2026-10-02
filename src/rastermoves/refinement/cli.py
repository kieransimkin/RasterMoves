"""Command-line integration; none of these helpers imports a diffusion runtime."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from ..errors import UpscaleError
from ..pipeline import Upscaler
from ..tracing import span
from .api import config_refiner, paths_for, refinement_identity, run_workflow, validate_paths
from .specs import BUILTINS, get_spec

OPTION_FIELDS = ("strength", "steps", "guidance", "control_scale", "tile", "overlap", "blend", "mask_feather")


def add_arguments(commands, upscale):
    refiners = commands.add_parser("refiners", help="List pinned same-size refinement workflows (no downloads).")
    refiners.add_argument("--json", action="store_true")
    download = commands.add_parser("download-refiner", help="Prefetch one refiner's pinned components; no inference.")
    download.add_argument("refiner", choices=list(BUILTINS))
    download.add_argument("--offline", action="store_true")
    download.add_argument("--strict-checksums", action="store_true")
    refine = commands.add_parser("refine", help="Refine existing images at unchanged dimensions; save lossless baselines.")
    refine.add_argument("input", type=Path)
    refine.add_argument("-o", "--output", type=Path)
    refine.add_argument("--format", choices=["png", "webp", "tiff"], default="png")
    refine.add_argument("--recursive", action="store_true")
    refine.add_argument("--continue-on-error", action="store_true")
    refine.add_argument("--overwrite", action="store_true")
    refine.add_argument("--resume", action="store_true")
    refine.add_argument("--dry-run", action="store_true")
    refine.add_argument("--offline", action="store_true")
    refine.add_argument("--strict-checksums", action="store_true")
    refine.add_argument("--max-output-mp", type=float, default=64)
    refine.add_argument("--trace", action="store_true")
    refine.add_argument("--trace-file", type=Path)
    refine.add_argument("--trace-interval", type=float)
    # Common trace planner compatibility; no inference meaning on refine.
    refine.set_defaults(model_file=None, all_models=False, backend="none", device="auto", precision="auto",
                        tile=None, overlap=None, tile_pad=None)
    for command in (upscale, refine):
        command.add_argument("--refiner", default="sd15-tile" if command is refine else None,
                             help="Optional same-size workflow: sd15-tile, sdxl-tile, or an explicitly enabled Python plugin.")
        command.add_argument("--refine-strength", type=float, help="Img2img strength, 0..1; default 0.22. Zero bypasses generation.")
        command.add_argument("--refine-steps", type=int, help="Scheduler steps before strength truncation; default 40.")
        command.add_argument("--refine-guidance", type=float, help="Classifier-free guidance; default 5.")
        command.add_argument("--refine-control-scale", type=float, help="ControlNet conditioning scale; default 1.")
        command.add_argument("--refine-tile", type=int, help="Diffusion tile in final-image pixels, multiple of 8; default 512/1024.")
        command.add_argument("--refine-overlap", type=int, help="Diffusion crop overlap; default 64 pixels.")
        command.add_argument("--refine-blend", type=float, help="Blend generated RGB with baseline, 0..1; default 1.")
        command.add_argument("--refine-color-match", action="store_true", help="Match global RGB means before compositing.")
        command.add_argument("--refine-mask-feather", type=float, help="Outward protection-mask feather radius in pixels; default 0.")
        command.add_argument("--protect-mask", type=Path, help="Working-size opaque mask: white protects, black permits changes.")
        command.add_argument("--seed", type=int, help="Reproducible root seed; default 0 (tile seeds are coordinate-derived).")
        command.add_argument("--refine-prompt", "--prompt", dest="refine_prompt", help="Optional content description; default empty.")
        command.add_argument("--refine-negative-prompt", "--negative-prompt", dest="refine_negative_prompt")
        command.add_argument("--refine-device", help="auto/cpu/mps/cuda/cuda:N, independently of the upscaler.")
        command.add_argument("--refine-precision", choices=["auto", "fp16", "fp32"])
        command.add_argument("--refine-offload", choices=["none", "model", "sequential"])
        command.add_argument("--refine-scheduler", choices=["ddim", "euler"])
        command.add_argument("--refine-no-vae-tiling", action="store_true")


def validate_selection(args):
    if args.command != "upscale" or args.refiner:
        return
    for key, value in vars(args).items():
        if (key.startswith("refine_") and value is not None and value is not False) or (key in {"seed", "protect_mask"} and value is not None):
            raise UpscaleError(f"--{key.replace('_', '-')} requires --refiner.")


def configuration(args) -> dict:
    options = {key: getattr(args, "refine_" + key) for key in OPTION_FIELDS
               if getattr(args, "refine_" + key) is not None}
    options.update(max_output_mp=args.max_output_mp, color_match=args.refine_color_match)
    for key, attr in (("seed", "seed"), ("prompt", "refine_prompt"), ("negative_prompt", "refine_negative_prompt")):
        if getattr(args, attr) is not None:
            options[key] = getattr(args, attr)
    config = {"refiner": args.refiner, "options": options, "cache_dir": args.cache_dir, "offline": args.offline,
              "strict_checksums": args.strict_checksums, "external_plugins": args.external_plugins,
              "vae_tiling": not args.refine_no_vae_tiling}
    for key in ("device", "precision", "offload", "scheduler"):
        if getattr(args, "refine_" + key) is not None:
            config[key] = getattr(args, "refine_" + key)
    if args.protect_mask:
        config["protect_mask"] = str(args.protect_mask.expanduser().absolute())
    config_refiner(config)  # Validation occurs before any model or catalogue download.
    return config


def run_metadata(args) -> int:
    if args.command == "refiners":
        data = [spec.to_dict() for spec in BUILTINS.values()]
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            for spec in BUILTINS.values():
                print(f"{spec.id:16} {spec.family:6} same-size; diffusion tile {spec.default_tile}")
                for component in (spec.base, spec.controlnet):
                    print(f"  {component.role}: {component.repo_id}@{component.revision} | {component.license}")
            print("Generative detail is not verified recovery. Model component terms apply separately.")
        return 0
    from .diffusers_backend import download_bundle
    spec, _ = get_spec(args.refiner)
    _, inventory = download_bundle(spec, cache_dir=args.cache_dir, offline=args.offline,
                                   strict_checksums=args.strict_checksums)
    print(json.dumps({"refiner": spec.id, "components": inventory}, indent=2))
    return 0


def run_refinement(args) -> int:
    from ..cli import batch_jobs
    config = configuration(args)
    chained = args.command == "upscale"
    if chained and args.sync_models:
        raise UpscaleError("--sync-models requires --all-models.")
    jobs = batch_jobs(args.input, args.output, recursive=args.recursive, format=args.format,
                      suffix="upscaled" if chained else "refined")
    if args.resume and args.overwrite:
        raise UpscaleError("Choose --resume or --overwrite.")
    for source, output in jobs:
        validate_paths(source, paths_for(output), mask=args.protect_mask)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "stage": "upscale-then-refine" if chained else "refine-only",
                          "refinement": refinement_identity(config),
                          "jobs": [{"input": str(s), **{k: str(v) for k, v in paths_for(o).items()}} for s, o in jobs]}, indent=2))
        return 0
    failures = 0
    with config_refiner(config) as refiner:
        for i, (source, output) in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {source} -> {output} ({args.refiner})", file=sys.stderr)
            up = None
            try:
                with span("refinement_workflow", input_name=source.name, refiner=args.refiner):
                    options = {}
                    if chained:
                        # The next upscaler must not overlap the previous image's diffusion session.
                        refiner.close()
                        up = Upscaler(args.model, model_file=args.model_file, native_scale=args.native_scale,
                                      cache_dir=args.cache_dir, device=args.device, backend=args.backend,
                                      precision=args.precision, offline=args.offline, strict_checksums=args.strict_checksums,
                                      extra_arches=args.extra_arches, model_dirs=args.plugin_dir,
                                      external_plugins=args.external_plugins)
                        options = {key: getattr(args, key) for key in ("tile", "overlap", "tile_pad", "scale", "width",
                                   "height", "long_edge", "alpha", "max_output_mp", "force_tiling")}
                    result = run_workflow(source, output, refiner, upscaler=up, image_options=options,
                                          protect_mask=args.protect_mask, overwrite=args.overwrite, resume=args.resume)
                print(result)
            except (UpscaleError, OSError, ValueError) as e:
                if not args.continue_on_error:
                    raise
                refiner.close()
                failures += 1
                print(f"ERROR: {source}: {e}", file=sys.stderr)
            finally:
                if up is not None:
                    up.close()
    return 1 if failures else 0
