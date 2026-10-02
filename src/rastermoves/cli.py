from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
import sys

from . import __version__
from .catalog import sync_catalog
from .downloads import Downloader, source_kind
from .errors import UpscaleError
from .paths import cache_dir
from .pipeline import DEFAULT_MODEL, SUPPORTED_IMAGES, Upscaler
from .registry import Registry


def parser():
    p = argparse.ArgumentParser(prog="rastermoves", description="RasterMoves: modular AI image upscaling for DanceFlow, powered by OpenModelDB model plugins.")
    p.add_argument("--version", action="version", version=f"RasterMoves {__version__}")
    p.add_argument("--cache-dir", help="Cache root (or RASTERMOVES_CACHE). Place global flags before the command.")
    p.add_argument("--plugin-dir", action="append", default=[], help="Directory of JSON model manifests; repeatable.")
    p.add_argument("--external-plugins", action="store_true", help="Enable trusted installed Python plugin entry points.")
    p.add_argument("--verbose", action="store_true")
    commands = p.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models", help="Search the locally available model plugins.")
    models.add_argument("query", nargs="?", default="")
    models.add_argument("--architecture")
    models.add_argument("--scale", type=int)
    models.add_argument("--tag")
    models.add_argument("--json", action="store_true")
    info = commands.add_parser("info", help="Inspect a model, sources, licence and checksum.")
    info.add_argument("model")
    info.add_argument("--offline", action="store_true")
    sync = commands.add_parser("sync", help="Refresh the entire OpenModelDB catalogue, without downloading weights.")
    sync.add_argument("--source", help="OpenModelDB export JSON, model directory, repository checkout, or HTTPS URL.")
    sync.add_argument("--offline", action="store_true")
    download = commands.add_parser("download", help="Prefetch one model's weights; reuse cached weights thereafter.")
    download.add_argument("model")
    download.add_argument("--backend", default="auto")
    download.add_argument("--offline", action="store_true")
    download.add_argument("--strict-checksums", action="store_true")
    commands.add_parser("doctor", help="Show installed runtime versions and available devices.")
    up = commands.add_parser("upscale", help="Upscale one image or a directory; weights are downloaded on first use.")
    up.add_argument("input", type=Path)
    up.add_argument("-o", "--output", type=Path)
    selection = up.add_mutually_exclusive_group()
    selection.add_argument("-m", "--model", help=f"Model ID or OpenModelDB page; default {DEFAULT_MODEL}.")
    selection.add_argument("--all-models", action="store_true", help="Try every registered model on one image; -o must name a folder.")
    # --model + --model-file is intentionally supported for checksum validation.
    up.add_argument("--model-file", type=Path, help="Use a local state-dict/safetensors/ONNX model.")
    up.add_argument("--sync-models", action="store_true", help="With --all-models, refresh the full OpenModelDB catalogue first (metadata only).")
    up.add_argument("--dry-run", action="store_true", help="With --all-models, print the model/output plan without downloading weights.")
    up.add_argument("--resume", action="store_true", help="With --all-models, reuse verified successes and retry failed/interrupted models.")
    up.add_argument("--native-scale", type=int, help="Required for a standalone local ONNX file.")
    up.add_argument("--backend", default="auto", help="auto, spandrel, onnx, or an external backend name.")
    up.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N; also mps with PyTorch.")
    up.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    up.add_argument("--tile", type=int, default=256, help="Core tile size in input pixels; 0 = whole image.")
    up.add_argument("--overlap", type=int, default=32, help="Feathered core overlap, in input pixels.")
    up.add_argument("--tile-pad", type=int, default=16, help="Extra context halo around each tile, in input pixels.")
    up.add_argument("--force-tiling", action="store_true", help="Override a model's global-context tiling guidance.")
    sizing = up.add_mutually_exclusive_group()
    sizing.add_argument("--scale", type=float, help="Final scale, after the native neural pass and Lanczos resizing.")
    sizing.add_argument("--width", type=int)
    sizing.add_argument("--height", type=int)
    sizing.add_argument("--long-edge", type=int)
    up.add_argument("--max-output-mp", type=float, default=64, help="Native AND final output pixel limit (default 64 MP).")
    up.add_argument("--alpha", choices=["lanczos", "model"], default="lanczos")
    up.add_argument("--recursive", action="store_true")
    up.add_argument("--format", choices=["png", "webp", "jpg", "tiff"], default="png", help="Directory/default output format.")
    up.add_argument("--overwrite", action="store_true")
    up.add_argument("--continue-on-error", action="store_true", help="Process remaining batch files; still exit nonzero on failures.")
    up.add_argument("--report", action="store_true", help="Write a JSON provenance sidecar next to each output.")
    up.add_argument("--offline", action="store_true")
    up.add_argument("--strict-checksums", action="store_true")
    up.add_argument("--extra-arches", action="store_true", help="Opt in to installed extra architectures with additional licences.")
    return p


def _registry(args):
    return Registry(args.cache_dir, model_dirs=args.plugin_dir, external_plugins=args.external_plugins)


def _doctor():
    data = {"rastermoves": __version__, "python": sys.version.split()[0], "packages": {}}
    for package in ("torch", "torchvision", "spandrel", "safetensors", "onnxruntime", "onnxruntime-gpu", "gdown", "huggingface-hub"):
        try:
            data["packages"][package] = version(package)
        except PackageNotFoundError:
            data["packages"][package] = None
    try:
        import torch
        data["torch_devices"] = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    except Exception as e:
        data["torch_error"] = str(e)
    try:
        import onnxruntime as ort
        data["onnx_providers"] = ort.get_available_providers()
    except Exception as e:
        data["onnx_error"] = str(e)
    return data


def batch_jobs(source: Path, output: Path | None, *, recursive=False, format="png"):
    if source.is_file():
        dest = output or source.with_name(f"{source.name}_upscaled.{format}")
        if dest.is_dir():
            dest = dest / f"{source.name}_upscaled.{format}"
        return [(source, dest)]
    if not source.is_dir():
        raise UpscaleError(f"Input does not exist: {source}")
    dest_root = output or source.with_name(source.name + "_upscaled")
    if dest_root.resolve() == source.resolve():
        raise UpscaleError("Batch output directory must differ from the input directory.")
    if dest_root.is_file():
        raise UpscaleError("A directory input requires a directory output.")
    paths = sorted(source.rglob("*") if recursive else source.glob("*"))
    result = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGES:
            continue
        if path.resolve().is_relative_to(dest_root.resolve()):
            continue
        relative = path.relative_to(source)
        # Keep the original extension in the output stem so foo.jpg and foo.png cannot collide.
        dest = dest_root / relative.parent / f"{relative.name}_upscaled.{format}"
        result.append((path, dest))
    if not result:
        raise UpscaleError("No supported images found.")
    return result


def _all_models(args) -> int:
    from .sweep import output_name, run_all_models
    if args.model_file or args.native_scale is not None or args.recursive:
        raise UpscaleError("--all-models cannot be combined with --model-file, --native-scale or --recursive.")
    if args.output is None:
        raise UpscaleError("--all-models requires -o/--output with a named output folder.")
    if not args.input.is_file():
        raise UpscaleError("--all-models requires one input image file, not a directory.")
    if args.output.exists() and not args.output.is_dir():
        raise UpscaleError("--all-models output must be a named directory, not a file.")
    if args.resume and args.overwrite:
        raise UpscaleError("Choose --resume or --overwrite, not both.")
    offline = Downloader(args.cache_dir, offline=args.offline).offline
    if args.sync_models:
        if offline:
            raise UpscaleError("--sync-models requires network access; omit it to use the cached catalogue offline.")
        print("Refreshing OpenModelDB metadata (no weights yet)...", file=sys.stderr)
        sync_catalog(cache_dir(args.cache_dir))
    registry = _registry(args)
    specs = registry.search()
    print(f"Selected {len(specs)} model plugins. Weights download as needed; the full catalogue can use substantial disk space.", file=sys.stderr)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "input": str(args.input), "output_directory": str(args.output),
                          "model_count": len(specs), "models": [
                              {"id": s.id, "license": s.license, "native_scale": s.scale,
                               "output": str(args.output / output_name(s, args.format)),
                               "resource_sizes_bytes": [r.size for r in s.resources]} for s in specs]}, indent=2))
        return 0
    sessions = {"cache_dir": args.cache_dir, "device": args.device, "backend": args.backend,
                "precision": args.precision, "offline": offline, "strict_checksums": args.strict_checksums,
                "extra_arches": args.extra_arches, "model_dirs": args.plugin_dir, "external_plugins": args.external_plugins}
    options = {key: getattr(args, key) for key in (
        "tile", "overlap", "tile_pad", "scale", "width", "height", "long_edge", "alpha", "max_output_mp", "force_tiling")}
    def progress(index, total, model, status):
        print(f"[{index}/{total}] {model}: {status}", file=sys.stderr)
    summary = run_all_models(args.input, args.output, specs, session_options=sessions,
                             image_options=options, format=args.format, registry=registry, overwrite=args.overwrite,
                             resume=args.resume, progress=progress)
    for row in summary["results"]:
        if row["status"] in {"success", "reused"}:
            print(Path(summary["output_directory"]) / row["output"])
        else:
            print(f"ERROR: {row['model_id']}: {row.get('error', row.get('cleanup_error', row['status']))}", file=sys.stderr)
    print(f"Summary: {Path(summary['output_directory']) / 'summary.json'}", file=sys.stderr)
    return 1 if summary["counts"]["failed"] else 0


def run(args) -> int:
    if args.command == "doctor":
        print(json.dumps(_doctor(), indent=2))
    elif args.command == "sync":
        offline = Downloader(args.cache_dir, offline=args.offline).offline
        print(json.dumps(sync_catalog(cache_dir(args.cache_dir), args.source, offline=offline), indent=2))
    elif args.command == "models":
        items = _registry(args).search(args.query, architecture=args.architecture, scale=args.scale, tag=args.tag)
        if args.json:
            print(json.dumps([s.to_dict() for s in items], indent=2))
        else:
            print(f"{'MODEL ID':42} {'SCALE':5} {'ARCHITECTURE':14} {'FORMATS':20} LICENCE")
            for s in items:
                formats = ",".join(dict.fromkeys(r.format for r in s.resources)) or "none"
                print(f"{s.id:42} {s.scale:<5} {s.architecture:14} {formats:20} {s.license}")
            print(f"\n{len(items)} model plugins. Catalogue presence does not guarantee runtime compatibility. Use sync for more.")
    elif args.command == "info":
        offline = Downloader(args.cache_dir, offline=args.offline).offline
        spec = _registry(args).get(args.model, auto_fetch=True, offline=offline).spec
        data = spec.to_dict()
        data["source_types"] = sorted({source_kind(u) for r in spec.resources for u in r.urls})
        data["compatibility"] = "Requires a successful load with an installed backend; catalogue membership alone is not validation."
        print(json.dumps(data, indent=2))
    elif args.command == "download":
        dl = Downloader(args.cache_dir, offline=args.offline, strict_checksums=args.strict_checksums)
        plugin = _registry(args).get(args.model, auto_fetch=True, offline=dl.offline)
        errors = []
        for resource in plugin.resources(args.backend):
            if resource.backend not in {"spandrel", "onnx"} and not args.external_plugins:
                continue
            try:
                print(dl.get(resource))
                return 0
            except UpscaleError as e:
                errors.append(str(e))
        raise UpscaleError("No model resource downloaded. " + "; ".join(errors))
    elif args.command == "upscale":
        if args.all_models:
            return _all_models(args)
        if args.sync_models or args.dry_run or args.resume:
            raise UpscaleError("--sync-models, --dry-run and --resume require --all-models.")
        jobs = batch_jobs(args.input, args.output, recursive=args.recursive, format=args.format)
        failures = 0
        with Upscaler(args.model, model_file=args.model_file, native_scale=args.native_scale, cache_dir=args.cache_dir,
                      device=args.device, backend=args.backend, precision=args.precision,
                      offline=args.offline, strict_checksums=args.strict_checksums, extra_arches=args.extra_arches,
                      model_dirs=args.plugin_dir, external_plugins=args.external_plugins) as up:
            for index, (source, output) in enumerate(jobs, 1):
                print(f"[{index}/{len(jobs)}] {source} -> {output}", file=sys.stderr)
                try:
                    up.upscale_file(source, output, overwrite=args.overwrite, report=args.report,
                                    tile=args.tile, overlap=args.overlap, tile_pad=args.tile_pad,
                                    scale=args.scale, width=args.width, height=args.height, long_edge=args.long_edge,
                                    alpha=args.alpha, max_output_mp=args.max_output_mp, force_tiling=args.force_tiling)
                    print(output)
                except (UpscaleError, OSError, ValueError) as e:
                    if not args.continue_on_error:
                        raise
                    failures += 1
                    print(f"ERROR: {source}: {e}", file=sys.stderr)
        return 1 if failures else 0
    return 0


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return run(args)
    except (UpscaleError, OSError, ValueError) as e:
        if args.verbose:
            logging.exception("Failed")
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
