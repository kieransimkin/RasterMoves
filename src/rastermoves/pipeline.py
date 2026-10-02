from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import logging
import math
import os
from pathlib import Path
import tempfile
import time

from filelock import FileLock
import numpy as np
from PIL import Image, ImageCms, ImageOps

from .downloads import Downloader, sha256_file
from . import __version__
from .errors import UpscaleError
from .network import atomic_json
from .plugins import backend_factories
from .registry import Registry
from .specs import ModelSpec
from .tiling import upscale_array
from .tracing import current_trace, span, traced

DEFAULT_MODEL = "4x-realesr-general-x4v3"
log = logging.getLogger(__name__)
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def target_size(width, height, native_scale, *, scale=None, target_width=None, target_height=None, long_edge=None):
    count = sum(v is not None for v in (scale, target_width, target_height, long_edge))
    if count > 1:
        raise UpscaleError("Choose only one final-size option: scale, width, height, or long-edge.")
    factor = float(native_scale)
    if scale is not None:
        factor = float(scale)
    elif target_width is not None:
        factor = target_width / width
    elif target_height is not None:
        factor = target_height / height
    elif long_edge is not None:
        factor = long_edge / max(width, height)
    if not math.isfinite(factor) or factor <= 0:
        raise UpscaleError("Final size/scale must be finite and positive.")
    return max(1, round(width * factor)), max(1, round(height * factor))


def _high_bit_depth(image: Image.Image) -> bool:
    if image.mode in {"I", "F"} or image.mode.startswith("I;16"):
        return True
    # Pillow can expose 16-bit RGB PNG/TIFF as RGB after implicitly reducing precision.
    # Check their source metadata before that lossy conversion becomes invisible.
    if image.format == "TIFF" and hasattr(image, "tag_v2"):
        bits = image.tag_v2.get(258, (8,))
        bits = (bits,) if isinstance(bits, int) else bits
        return any(n > 8 for n in bits)
    if image.format == "PNG" and getattr(image, "fp", None) is not None:
        file = image.fp
        position = file.tell()
        try:
            file.seek(0)
            header = file.read(25)
            return header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) == 25 and header[24] > 8
        finally:
            file.seek(position)
    return False


@traced("preprocess")
def _prepare(image: Image.Image):
    if getattr(image, "n_frames", 1) != 1:
        raise UpscaleError("Animated/multi-page images are not supported. Extract frames first.")
    if _high_bit_depth(image):
        raise UpscaleError("High-bit-depth/HDR inputs are not supported; convert explicitly to 8-bit sRGB first.")
    image = ImageOps.exif_transpose(image)
    image.load()
    alpha = image.convert("RGBA").getchannel("A") if "A" in image.getbands() or "transparency" in image.info else None
    icc = image.info.get("icc_profile")
    if icc:
        try:
            srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
            source = ImageCms.ImageCmsProfile(BytesIO(icc))
            # Preserve CMYK while doing profile conversion; convert palette and alpha images to RGB.
            color = image if image.mode in {"RGB", "CMYK", "L"} else image.convert("RGB")
            color = ImageCms.profileToProfile(color, source, srgb, outputMode="RGB")
            icc = srgb.tobytes()
        except (ValueError, OSError, ImageCms.PyCMSError) as e:
            raise UpscaleError("Could not convert the embedded colour profile to sRGB; convert the input explicitly first.") from e
    else:
        color = image.convert("RGB")
    return color, alpha, icc


class Upscaler:
    """Reusable session; weights download/load once and all inference is local.

    Use as a context manager to release model memory after a batch. Not thread-safe.
    """
    @traced("session_init")
    def __init__(self, model: str | None = None, *, model_file=None, native_scale=None, cache_dir=None,
                 device="auto", backend="auto", precision="fp32", offline=False, strict_checksums=False,
                 extra_arches=False, model_dirs=(), external_plugins=False, registry: Registry | None = None):
        self.downloader = Downloader(cache_dir, offline=offline, strict_checksums=strict_checksums)
        self.registry = registry if registry is not None else Registry(
            self.downloader.root, model_dirs=model_dirs, external_plugins=external_plugins)
        self.device, self.backend, self.precision = device, backend, precision
        self.extra_arches, self.external_plugins = extra_arches, external_plugins
        self.local_file = Path(model_file).expanduser() if model_file else None
        self.validate_metadata = True
        if self.local_file and not self.local_file.is_file():
            raise UpscaleError(f"Model file does not exist: {self.local_file}")
        if model is not None or self.local_file is None:
            self.plugin = self.registry.get(model or DEFAULT_MODEL, auto_fetch=True, offline=self.downloader.offline)
            self.spec = self.plugin.spec
        else:
            if self.local_file.suffix.lower() == ".onnx" and native_scale is None:
                raise UpscaleError("A standalone ONNX model requires --native-scale (for example --native-scale 4).")
            self.spec = ModelSpec(id="local-model", name=self.local_file.name, scale=native_scale or 1)
            self.plugin = None
            self.validate_metadata = False
        self.loaded = None
        self.weight_path = None
        self.resource = None
        self.last_report = None

    @traced("load_model")
    def load(self):
        if self.loaded is not None:
            return self.loaded
        if self.local_file:
            suffix = self.local_file.suffix.lower()
            kind = self.backend if self.backend != "auto" else ("onnx" if suffix == ".onnx" else "spandrel")
            factory = backend_factories(external=self.external_plugins).get(kind)
            if factory is None:
                raise UpscaleError(f"No backend named {kind}.")
            if self.plugin:
                matches = [r for r in self.spec.resources if r.format == suffix.lstrip(".") and r.backend == kind]
                if matches:
                    errors = []
                    for r in matches:
                        try:
                            if self.downloader.strict_checksums and not r.sha256:
                                raise UpscaleError("No publisher checksum available for the supplied local weights.")
                            self.downloader._verify(self.local_file, r)
                            self.resource = r
                            break
                        except UpscaleError as e:
                            errors.append(str(e))
                    else:
                        raise UpscaleError("Local weights do not match the selected model. " + "; ".join(errors))
                elif self.downloader.strict_checksums:
                    raise UpscaleError("No matching checksum manifest for local weights.")
            elif self.downloader.strict_checksums:
                raise UpscaleError("Standalone local weights have no publisher checksum; select a model manifest.")
            with span("backend_load", backend=kind, model_id=self.spec.id):
                self.loaded = factory(self.local_file, self.spec, device=self.device, precision=self.precision,
                                      extra_arches=self.extra_arches, validate_metadata=self.validate_metadata)
            self.weight_path = self.local_file
        else:
            self.loaded, self.weight_path, self.resource = self.plugin.load(
                self.downloader, backend=self.backend, device=self.device, precision=self.precision,
                extra_arches=self.extra_arches, external_backends=self.external_plugins)
        return self.loaded

    @traced("upscale_image")
    def upscale_image(self, image: Image.Image, *, tile=256, overlap=32, tile_pad=16,
                      scale=None, width=None, height=None, long_edge=None, alpha="lanczos",
                      max_output_mp=64, force_tiling=False, progress=None) -> Image.Image:
        if not math.isfinite(max_output_mp) or max_output_mp <= 0:
            raise UpscaleError("max_output_mp must be finite and positive.")
        if alpha not in {"lanczos", "model"}:
            raise UpscaleError("Alpha method must be lanczos or model.")
        started = time.perf_counter()
        color, alpha_channel, icc = _prepare(image)
        loaded = self.load()
        target = target_size(*color.size, loaded.scale, scale=scale, target_width=width,
                             target_height=height, long_edge=long_edge)
        max_pixels = int(max_output_mp * 1_000_000)
        if target[0] * target[1] > max_pixels:
            raise UpscaleError("Final output exceeds --max-output-mp.")
        native_size = (color.width * loaded.scale, color.height * loaded.scale)
        if target[0] > native_size[0] or target[1] > native_size[1]:
            log.warning("Final target exceeds native model scale; additional enlargement is Lanczos, not another AI pass.")
        with span("input_to_array"):
            if loaded.input_channels == 1:
                color = color.convert("L")
            array = np.asarray(color, dtype=np.float32) / 255.0
            if array.ndim == 2:
                array = array[..., None]
        opts = dict(tile=tile, overlap=overlap, pad=tile_pad, max_output_pixels=max_pixels,
                    force_tiling=force_tiling, progress=progress)
        result = upscale_array(array, loaded, **opts)
        with span("postprocess"):
            pixels = np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8)
            if loaded.output_channels == 1:
                pixels = pixels[..., 0]
            output = Image.fromarray(pixels)
            if output.size != target:
                output = output.resize(target, Image.Resampling.LANCZOS)
            if alpha_channel is not None:
                if alpha == "model":
                    a = np.asarray(alpha_channel, dtype=np.float32)[..., None] / 255.0
                    if loaded.input_channels == 3:
                        a = np.repeat(a, 3, axis=2)
                    a = upscale_array(a, loaded, **opts)
                    a = np.mean(a, axis=2)
                    alpha_channel = Image.fromarray(np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8))
                alpha_channel = alpha_channel.resize(target, Image.Resampling.LANCZOS)
                output.putalpha(alpha_channel)
            if icc and loaded.output_channels == 3:
                output.info["icc_profile"] = icc
        # EXIF is intentionally not copied: orientation is applied; stale dimensions/GPS are omitted.
        with span("report_checksum"):
            weight_digest = sha256_file(self.weight_path) if self.weight_path else None
        self.last_report = {
            "software": "RasterMoves", "software_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(), "model_id": self.spec.id,
            "model_license": self.spec.license, "model_page": self.spec.source_page,
            "weight_sha256": weight_digest,
            "publisher_verified": bool(self.resource and self.resource.sha256),
            "backend": type(loaded).__name__, "device": loaded.device, "precision": loaded.precision,
            "input_size": list(color.size), "native_scale": loaded.scale, "output_size": list(output.size),
            "requested_tile": tile, "overlap": overlap, "tile_pad": tile_pad, "alpha_method": alpha,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        return output

    @traced("upscale_file")
    def upscale_file(self, source, output, *, overwrite=False, report=False, **kwargs) -> Path:
        source, output = Path(source), Path(output)
        trace = current_trace()
        if trace:
            trace.protect_paths([source, output, output.with_suffix(output.suffix + ".json")])
        if source.resolve() == output.resolve():
            raise UpscaleError("Input and output must be different files, even with --overwrite.")
        if output.exists() and not overwrite:
            raise UpscaleError(f"Output already exists: {output}. Use --overwrite to replace it.")
        formats = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP", ".tif": "TIFF", ".tiff": "TIFF"}
        fmt = formats.get(output.suffix.lower())
        if fmt is None:
            raise UpscaleError("Output extension must be PNG, JPEG, WebP or TIFF.")
        with Image.open(source) as original:
            if fmt == "JPEG" and ("A" in original.getbands() or "transparency" in original.info):
                raise UpscaleError("JPEG cannot preserve alpha. Select PNG/WebP/TIFF or flatten the input explicitly.")
            result = self.upscale_image(original, **kwargs)
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".rastermoves-", suffix=output.suffix, dir=output.parent)
        os.close(fd)
        try:
            options = {}
            if "icc_profile" in result.info:
                options["icc_profile"] = result.info["icc_profile"]
            if fmt == "JPEG":
                options.update(quality=95, subsampling=0)
            elif fmt == "WEBP":
                options.update(lossless=True)
            with span("write_image"):
                result.save(name, format=fmt, **options)
            with FileLock(str(output) + ".lock", timeout=600):
                if output.exists() and not overwrite:
                    raise UpscaleError(f"Output appeared while processing: {output}.")
                os.replace(name, output)
                if report:
                    payload = {**self.last_report, "input": str(source), "output": str(output)}
                    with span("write_provenance"):
                        atomic_json(output.with_suffix(output.suffix + ".json"), payload)
        finally:
            Path(name).unlink(missing_ok=True)
            result.close()
        return output

    @traced("model_cleanup")
    def close(self):
        if self.loaded is not None:
            self.loaded.close()
            self.loaded = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
