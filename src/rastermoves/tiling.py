"""Halo tiles with overlapping feathered cores; only one input tile lives on the device."""
from __future__ import annotations

import logging
import math
import numpy as np

from .errors import BackendOOM, UpscaleError

log = logging.getLogger(__name__)


def positions(length: int, tile: int, overlap: int) -> list[int]:
    if tile >= length:
        return [0]
    step = tile - overlap
    values = list(range(0, length - tile + 1, step))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def feather(length: int, overlap: int, at_start: bool, at_end: bool) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    n = min(overlap, length // 2)
    if n:
        ramp = np.arange(1, n + 1, dtype=np.float32) / (n + 1)
        if not at_start:
            weights[:n] = ramp
        if not at_end:
            weights[-n:] = ramp[::-1]
    return weights


def _predict(model, tile: np.ndarray) -> np.ndarray:
    output = model.predict(tile)
    expected = (tile.shape[0] * model.scale, tile.shape[1] * model.scale, model.output_channels)
    if not isinstance(output, np.ndarray) or output.shape != expected:
        raise UpscaleError(f"Backend returned {getattr(output, 'shape', None)}, expected {expected}.")
    if not np.isfinite(output).all():
        raise UpscaleError("Model returned NaN/Inf values. Use fp32, or try a different checkpoint/backend.")
    return np.asarray(output, dtype=np.float32)


def _once(image, model, tile, overlap, pad, progress):
    h, w, _ = image.shape
    s = model.scale
    if tile == 0 or (h <= tile and w <= tile):
        result = _predict(model, image)
        if progress:
            progress(1, 1)
        return np.clip(result, 0, 1)
    ys, xs = positions(h, tile, overlap), positions(w, tile, overlap)
    accumulator = np.zeros((h * s, w * s, model.output_channels), np.float32)
    weights = np.zeros((h * s, w * s, 1), np.float32)
    total = len(ys) * len(xs)
    completed = 0
    for y in ys:
        for x in xs:
            y2, x2 = min(y + tile, h), min(x + tile, w)
            py, px = max(0, y - pad), max(0, x - pad)
            py2, px2 = min(h, y2 + pad), min(w, x2 + pad)
            prediction = _predict(model, image[py:py2, px:px2])
            core = prediction[(y - py) * s:(y2 - py) * s, (x - px) * s:(x2 - px) * s]
            fy = feather((y2 - y) * s, overlap * s, y == 0, y2 == h)
            fx = feather((x2 - x) * s, overlap * s, x == 0, x2 == w)
            mask = (fy[:, None] * fx[None, :])[..., None]
            accumulator[y * s:y2 * s, x * s:x2 * s] += core * mask
            weights[y * s:y2 * s, x * s:x2 * s] += mask
            completed += 1
            if progress:
                progress(completed, total)
    if not np.all(weights > 0):
        raise UpscaleError("Tiling left uncovered pixels.")
    np.divide(accumulator, weights, out=accumulator)
    np.clip(accumulator, 0, 1, out=accumulator)
    return accumulator


def upscale_array(image: np.ndarray, model, *, tile=256, overlap=32, pad=16,
                  max_output_pixels=64_000_000, force_tiling=False, progress=None) -> np.ndarray:
    if any(type(v) is not int or v < 0 for v in (tile, overlap, pad)):
        raise UpscaleError("Tile, overlap and padding must be non-negative integers.")
    if tile and overlap >= tile:
        raise UpscaleError("Overlap must be smaller than the tile size.")
    if image.ndim != 3 or image.shape[2] != model.input_channels or min(image.shape[:2]) < 1:
        raise UpscaleError("Expected a nonempty HWC image matching the model's input channels.")
    if not np.issubdtype(image.dtype, np.floating) or not np.isfinite(image).all():
        raise UpscaleError("Input must be a finite floating-point image.")
    if np.min(image) < 0 or np.max(image) > 1:
        raise UpscaleError("Input pixels must be normalized to [0,1].")
    pixels = image.shape[0] * image.shape[1] * model.scale**2
    if pixels > max_output_pixels:
        raise UpscaleError(f"Native AI output would have {pixels / 1e6:.1f} MP, above the configured "
                           f"{max_output_pixels / 1e6:.1f} MP limit. Resize the input, choose a smaller native scale, "
                           "or raise --max-output-mp only with enough system RAM.")
    if model.tiling != "supported" and not force_tiling:
        if tile:
            log.warning("Model tiling is %s; using a whole-image pass. --force-tiling overrides this guidance.", model.tiling)
        tile = 0
    while True:
        try:
            return _once(image, model, tile, overlap, pad, progress)
        except BackendOOM:
            # Leave the exception scope before retrying so failed tensor/array frames are released.
            if (model.tiling != "supported" and not force_tiling) or (tile and tile <= 16):
                raise UpscaleError("Device memory exhausted. Use CPU, a lighter model, or a smaller image. "
                                   "Models requiring global context may not support automatic tiling.") from None
        model.clear_cache()
        tile = min(256, max(16, math.ceil(max(image.shape[:2]) / 2))) if tile == 0 else max(16, tile // 2)
        overlap = min(overlap, tile // 4)
        pad = min(pad, tile // 4)
        log.warning("Device memory exhausted; restarting with tile=%d, overlap=%d, pad=%d.", tile, overlap, pad)
