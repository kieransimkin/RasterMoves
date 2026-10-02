"""Same-resolution refinement sessions and durable two-stage image workflows."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from filelock import FileLock, Timeout
import numpy as np
from PIL import Image, ImageOps

from .. import __version__
from ..downloads import sha256_file
from ..errors import UpscaleError
from ..network import atomic_json
from ..pipeline import Upscaler, _prepare
from ..tracing import current_trace, event, span
from .diffusers_backend import runtime_versions
from .specs import RefineOptions, get_spec
from .tiles import protection_mask, refine_tiles


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _image_digest(image: Image.Image) -> str:
    return hashlib.sha256(f"{image.mode}:{image.size}:".encode() + image.tobytes()).hexdigest()


def paths_for(output: str | Path) -> dict[str, Path]:
    output = Path(output).expanduser().absolute()
    return {"output": output, "report": output.with_suffix(output.suffix + ".json"),
            "baseline": output.with_name(output.stem + ".baseline.png"),
            "baseline_report": output.with_name(output.stem + ".baseline.png.json")}


def validate_paths(source: Path, paths: dict[str, Path], *, mask: Path | None = None):
    if not source.is_file():
        raise UpscaleError(f"Input does not exist: {source}")
    if paths["output"].suffix.lower() not in {".png", ".webp", ".tif", ".tiff"}:
        raise UpscaleError("Refinement output must be lossless PNG, WebP or TIFF (no JPEG).")
    outputs = [p.resolve() for p in paths.values()]
    if len(set(outputs)) != len(outputs):
        raise UpscaleError("Refinement output paths collide.")
    for inp in (source, mask):
        if inp is not None and inp.resolve() in outputs:
            raise UpscaleError("An output overlaps the input or protection mask; choose another output name.")
    for path in paths.values():
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise UpscaleError(f"Refinement outputs must be regular files, not directories or symlinks: {path}")
    trace = current_trace()
    if trace:
        trace.protect_paths([source, mask, *paths.values()])


def save_image(image: Image.Image, destination: Path, *, overwrite=False) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".rastermoves-refine-", suffix=destination.suffix, dir=destination.parent)
    os.close(fd)
    try:
        options = {"icc_profile": image.info["icc_profile"]} if image.info.get("icc_profile") else {}
        if destination.suffix.lower() == ".webp":
            options.update(lossless=True, exact=True)
        with span("refine_write"):
            image.save(name, **options)
        with FileLock(str(destination) + ".lock", timeout=600):
            if destination.is_symlink() or (destination.exists() and not destination.is_file()):
                raise UpscaleError("Refinement destination changed to a symlink/directory during processing.")
            if destination.exists() and not overwrite:
                raise UpscaleError(f"Refinement destination appeared during processing: {destination}.")
            os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)
    return sha256_file(destination)


def read_mask(path: str | Path | None) -> Image.Image | None:
    if path is None:
        return None
    with Image.open(path) as original:
        if getattr(original, "n_frames", 1) != 1:
            raise UpscaleError("Protection mask must be a single image.")
        if original.mode not in {"1", "L", "RGB"}:
            raise UpscaleError("Protection mask must be opaque 8-bit grayscale/RGB.")
        return ImageOps.exif_transpose(original).copy()


class Refiner:
    """Optional same-size generative enhancement; not a scale-changing model.

    Model loading is lazy, and skipped for zero strength/blend or fully protected
    images. A session is reusable but deliberately not thread-safe.
    """
    def __init__(self, refiner="sd15-tile", *, options: RefineOptions | None = None,
                 cache_dir=None, offline=False, device="auto", precision="auto", offload="none",
                 vae_tiling=True, scheduler="ddim", strict_checksums=False, external_plugins=False):
        self.spec, self.factory = get_spec(refiner, external_plugins=external_plugins)
        self.options = (options or RefineOptions()).validate(self.spec)
        if device not in {"auto", "cpu", "mps", "cuda"} and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise UpscaleError("Invalid refinement device.")
        if precision not in {"auto", "fp16", "fp32"} or scheduler not in {"ddim", "euler"}:
            raise UpscaleError("Invalid refinement precision or scheduler.")
        if offload not in {"none", "model", "sequential"}:
            raise UpscaleError("Invalid refinement offload policy.")
        if offload != "none" and device in {"cpu", "mps"}:
            raise UpscaleError("CPU offload requires a CUDA refinement device.")
        if device == "cpu" and precision == "fp16":
            raise UpscaleError("CPU refinement requires fp32.")
        self.runtime = dict(cache_dir=cache_dir, offline=offline, device=device, precision=precision,
                            offload=offload, vae_tiling=vae_tiling, scheduler=scheduler,
                            strict_checksums=strict_checksums)
        self.loaded = None
        self.last_report = None
        self.reused = False

    def identity(self) -> dict:
        runtime = {k: v for k, v in self.runtime.items() if k not in {"cache_dir", "offline", "strict_checksums"}}
        return {"spec": self.spec.to_dict(), "options": asdict(self.options), "runtime": runtime,
                "runtime_versions": runtime_versions()}

    def _predict(self, image, **kwargs):
        if self.loaded is None:
            from .diffusers_backend import DiffusersTileBackend
            self.loaded = (self.factory or DiffusersTileBackend)(self.spec, **self.runtime)
        return self.loaded.predict(image, **kwargs)

    def refine_image(self, image: Image.Image, *, protect_mask: Image.Image | None = None) -> Image.Image:
        started = time.perf_counter()
        with span("refine_prepare"):
            base, alpha, icc = _prepare(image)
        try:
            if base.width * base.height > self.options.max_output_mp * 1_000_000:
                raise UpscaleError("Refinement image exceeds --max-output-mp.")
            protection = protection_mask(protect_mask, base.size, self.options.mask_feather)
            if alpha is not None:
                invisible = np.asarray(alpha) == 0
                if protection is None:
                    protection = invisible.astype(np.float32)
                else:
                    protection = np.maximum(protection, invisible)
            with span("refine_image", refiner=self.spec.id):
                output, stats = refine_tiles(base, self._predict, self.options,
                                             tile=self.options.tile or self.spec.default_tile, protection=protection)
            if alpha is not None:
                output.putalpha(alpha)
            if icc:
                output.info["icc_profile"] = icc
            self.last_report = {
                "kind": "same-size-generative-refinement", "generative": not bool(stats.get("skipped")),
                "warning": "Generated detail is plausible synthesis, not verified recovered information.",
                "refiner_id": self.spec.id, "input_size": list(base.size), "output_size": list(output.size),
                "baseline_pixel_sha256": _image_digest(base), "config": self.identity(),
                "mask_pixel_sha256": _image_digest(protect_mask) if protect_mask else None,
                "alpha": "unchanged from baseline", "tiling": "frozen-input-spatial-overlap-v1",
                "effective_tile": self.options.tile or self.spec.default_tile,
                "scheduler": self.runtime["scheduler"], "metrics": stats,
                "elapsed_seconds": time.perf_counter() - started,
                "components": getattr(self.loaded, "inventory", []),
                "device": getattr(self.loaded, "device", "not-loaded"),
                "precision": getattr(self.loaded, "precision", "not-loaded"),
            }
            return output
        finally:
            base.close()
            if alpha is not None:
                alpha.close()

    def refine_file(self, source, output, *, protect_mask=None, overwrite=False, resume=False) -> Path:
        return run_workflow(source, output, self, protect_mask=protect_mask, overwrite=overwrite, resume=resume)

    def close(self):
        if self.loaded is not None:
            try:
                self.loaded.close()
            finally:
                self.loaded = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def config_refiner(config: dict) -> Refiner:
    data = dict(config)
    data.pop("protect_mask", None)
    options = RefineOptions(**data.pop("options", {}))
    return Refiner(options=options, **data)


def refinement_identity(config: dict) -> dict:
    refiner = config_refiner(config)
    mask = config.get("protect_mask")
    return {**refiner.identity(), "mask_sha256": sha256_file(Path(mask)) if mask else None}


def _check_artifact(path: Path, record: dict | None) -> bool:
    return bool(isinstance(record, dict) and path.is_file() and sha256_file(path) == record.get("sha256"))


def run_workflow(source, output, refiner: Refiner, *, upscaler: Upscaler | None = None,
                 image_options: dict | None = None, protect_mask=None, overwrite=False, resume=False) -> Path:
    """Save a lossless baseline before refinement; journal failures and resume safely.

    For chained processing, unload the upscaler before loading diffusion. Successful
    final output reuse requires baseline, baseline provenance and final checksums.
    """
    source = Path(source).expanduser().absolute()
    mask_path = Path(protect_mask).expanduser().absolute() if protect_mask else None
    paths = paths_for(output)
    validate_paths(source, paths, mask=mask_path)
    if resume and overwrite:
        raise UpscaleError("Choose --resume or --overwrite, not both.")
    mask = read_mask(mask_path)
    try:
        # Decode and reject unsupported input precision/animations before loading weights.
        with Image.open(source) as image:
            base, alpha, _ = _prepare(image)
            base.close()
            if alpha is not None:
                alpha.close()
        up_identity = None
        if upscaler:
            up_identity = {"spec": upscaler.spec.to_dict(), "options": dict(image_options or {}),
                           "backend": upscaler.backend, "device": upscaler.device, "precision": upscaler.precision,
                           "local_weight_sha256": sha256_file(upscaler.local_file) if upscaler.local_file else None}
        config = {"software_version": __version__, "input_sha256": sha256_file(source),
                  "output_format": paths["output"].suffix.lower(), "refinement": refiner.identity(),
                  "mask_sha256": sha256_file(mask_path) if mask_path else None, "upscaler": up_identity}
        fingerprint = digest(config)
        paths["output"].parent.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(paths["output"]) + ".workflow.lock", timeout=0):
                return _execute(source, paths, fingerprint, config, refiner, upscaler, image_options or {},
                                mask, mask_path, overwrite, resume)
        except Timeout as e:
            raise UpscaleError("Another refinement workflow is using this output.") from e
    finally:
        if mask is not None:
            mask.close()


def _execute(source, paths, fingerprint, config, refiner, upscaler, image_options, mask, mask_path, overwrite, resume):
    refiner.reused = False
    previous = {}
    if resume:
        if not paths["report"].is_file():
            raise UpscaleError("--resume requires this workflow's existing output JSON report.")
        try:
            previous = json.loads(paths["report"].read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            raise UpscaleError("Cannot read workflow resume report.") from e
        if (not isinstance(previous, dict) or previous.get("schema_version") != 1 or previous.get("kind") != "refinement-workflow"
                or previous.get("fingerprint") != fingerprint):
            raise UpscaleError("Cannot resume: input, mask, pipeline, version, runtime or refinement settings changed.")
    elif not overwrite and any(p.exists() for p in paths.values()):
        raise UpscaleError("Workflow output/baseline/report already exists. Use --resume, --overwrite or a new output.")
    artifacts = previous.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise UpscaleError("Invalid workflow artifact records.")
    baseline_ok = (_check_artifact(paths["baseline"], artifacts.get("baseline"))
                   and _check_artifact(paths["baseline_report"], artifacts.get("baseline_report")))
    if resume and previous.get("status") == "success" and baseline_ok and _check_artifact(paths["output"], artifacts.get("output")):
        refiner.last_report = previous.get("refinement")
        refiner.reused = True
        event("refinement_reused", output_name=paths["output"].name)
        return paths["output"]
    report = {"schema_version": 1, "kind": "refinement-workflow", "software": "RasterMoves",
              "software_version": __version__, "created_at": datetime.now(timezone.utc).isoformat(),
              "fingerprint": fingerprint, "config": config, "input": str(source),
              "status": "running", "artifacts": {}}
    if baseline_ok:
        report["artifacts"].update({k: artifacts[k] for k in ("baseline", "baseline_report")})
    trace = current_trace()
    if trace:
        report["trace_file"] = str(trace.path)
    atomic_json(paths["report"], report)
    try:
        if not baseline_ok:
            with span("workflow_baseline"):
                if upscaler:
                    upscaler.upscale_file(source, paths["baseline"], overwrite=overwrite or resume, report=True, **image_options)
                else:
                    with Image.open(source) as original:
                        base, alpha, icc = _prepare(original)
                    try:
                        if alpha is not None:
                            base.putalpha(alpha)
                        if icc:
                            base.info["icc_profile"] = icc
                        save_image(base, paths["baseline"], overwrite=overwrite or resume)
                    finally:
                        base.close()
                        if alpha is not None:
                            alpha.close()
                    atomic_json(paths["baseline_report"], {"kind": "lossless-srgb-baseline", "input_sha256": config["input_sha256"]})
            for key in ("baseline", "baseline_report"):
                report["artifacts"][key] = {"file": paths[key].name, "sha256": sha256_file(paths[key])}
            atomic_json(paths["report"], report)
        else:
            event("refinement_baseline_reused", baseline_name=paths["baseline"].name)
        if upscaler:
            upscaler.close()  # Never keep feed-forward and diffusion weights resident together.
        with Image.open(paths["baseline"]) as baseline:
            result = refiner.refine_image(baseline, protect_mask=mask)
        try:
            # Recheck inputs before publishing, detecting accidental edits during long runs.
            if sha256_file(source) != config["input_sha256"]:
                raise UpscaleError("Input changed during the workflow; refined output was not published.")
            if mask_path and sha256_file(mask_path) != config["mask_sha256"]:
                raise UpscaleError("Protection mask changed during the workflow; refined output was not published.")
            output_hash = save_image(result, paths["output"], overwrite=overwrite or resume)
        finally:
            result.close()
        report.update(status="success", refinement=refiner.last_report)
        report["artifacts"]["output"] = {"file": paths["output"].name, "sha256": output_hash}
    except BaseException as e:
        report.update(status="interrupted" if isinstance(e, (KeyboardInterrupt, SystemExit)) else "failed",
                      error=f"{type(e).__name__}: {e}")
        atomic_json(paths["report"], report)
        raise
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(paths["report"], report)
    return paths["output"]


class WorkflowUpscaler:
    """Adapter used by the existing all-model sweep: exactly one chosen refiner."""
    def __init__(self, model, *, refinement, **session_options):
        self.upscaler = Upscaler(model, **session_options)
        self.spec = self.upscaler.spec
        self.config = dict(refinement)
        self.refiner = config_refiner(self.config)
        self.last_output = None

    def upscale_file(self, source, output, *, overwrite=False, resume=False, report=True, **options):
        del report  # Workflow provenance is mandatory.
        result = run_workflow(source, output, self.refiner, upscaler=self.upscaler,
                              image_options=options, protect_mask=self.config.get("protect_mask"),
                              overwrite=overwrite, resume=resume)
        self.last_output = paths_for(result)
        return result

    def artifacts(self) -> dict:
        return {k: {"file": self.last_output[k].name, "sha256": sha256_file(self.last_output[k])}
                for k in ("baseline", "baseline_report")} if self.last_output else {}

    def close(self):
        try:
            self.upscaler.close()
        finally:
            self.refiner.close()
