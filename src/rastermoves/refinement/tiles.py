"""Frozen-input overlapping img2img tiles; NOT shared-latent MultiDiffusion."""
from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageFilter

from ..errors import UpscaleError
from ..tracing import span
from .specs import RefineOptions


def positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tile_seed(seed: int, x: int, y: int) -> int:
    # Stable across Python hash randomization and skipped protected tiles.
    return int.from_bytes(hashlib.sha256(f"{seed}:{x}:{y}".encode()).digest()[:8], "big") % 2**63


def protection_mask(mask: Image.Image | None, size: tuple[int, int], feather: float) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.size != size:
        raise UpscaleError("Protection mask must match the working image dimensions; it is never resized silently.")
    if mask.mode not in {"1", "L", "RGB"}:
        raise UpscaleError("Protection masks must be opaque 8-bit grayscale/RGB, not alpha or high-bit-depth images.")
    gray = mask.convert("L")
    hard = np.asarray(gray, dtype=np.float32) / 255.0
    if feather:
        # Feather outward only. Fully protected (white) pixels remain exactly protected.
        soft = np.asarray(gray.filter(ImageFilter.GaussianBlur(feather)), dtype=np.float32) / 255.0
        return np.maximum(hard, soft)
    return hard


def _weight(length: int, start: int, extent: int, overlap: int) -> np.ndarray:
    weight = np.ones(length, dtype=np.float32)
    ramp = min(overlap, length)
    if ramp:
        edge = (np.arange(ramp, dtype=np.float32) + 1) / (ramp + 1)
        if start > 0:
            weight[:ramp] *= edge
        if start + length < extent:
            weight[-ramp:] *= edge[::-1]
    return weight


def refine_tiles(image: Image.Image, predict: Callable, options: RefineOptions, *, tile: int,
                 protection: np.ndarray | None = None) -> tuple[Image.Image, dict]:
    """Process stable crops from the baseline, blend spatial overlap, then composite.

    Keeps output buffers in host RAM. Tile size bounds diffusion work, not total RAM.
    Pad with edge pixels to the next multiple of 8, crop back afterwards; never rescale.
    """
    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = base.shape[:2]
    if options.strength == 0 or options.blend == 0 or (protection is not None and np.all(protection == 1)):
        return Image.fromarray(base.copy()), {"skipped": "no editable pixels or zero strength/blend", "tiles": [],
                                              "executed_denoising_steps": 0}
    acc = np.zeros((h, w, 3), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    rows = []
    for y in positions(h, tile, options.overlap):
        for x in positions(w, tile, options.overlap):
            ch, cw = min(tile, h - y), min(tile, w - x)
            seed = tile_seed(options.seed, x, y)
            src = base[y:y + ch, x:x + cw]
            row = {"x": x, "y": y, "width": cw, "height": ch, "seed": seed}
            if protection is not None and np.all(protection[y:y + ch, x:x + cw] == 1):
                out = src.astype(np.float32)
                row.update(skipped="fully protected", executed_denoising_steps=0)
            else:
                ph, pw = max(64, (ch + 7) // 8 * 8), max(64, (cw + 7) // 8 * 8)
                padded = Image.fromarray(np.pad(src, ((0, ph - ch), (0, pw - cw), (0, 0)), mode="edge"))
                generated = None
                try:
                    with span("refine_tile", x=x, y=y, width=cw, height=ch, seed=seed):
                        generated, stats = predict(padded, options=options, seed=seed)
                    if not isinstance(generated, Image.Image) or generated.mode != "RGB" or generated.size != padded.size:
                        raise UpscaleError("Refiner returned invalid tile geometry/mode; expected same-size RGB.")
                    out = np.array(generated, dtype=np.float32)[:ch, :cw].copy()
                    row.update(stats)
                finally:
                    padded.close()
                    if isinstance(generated, Image.Image):
                        generated.close()
            weight = _weight(ch, y, h, options.overlap)[:, None] * _weight(cw, x, w, options.overlap)[None, :]
            acc[y:y + ch, x:x + cw] += out * weight[..., None]
            weights[y:y + ch, x:x + cw] += weight
            rows.append(row)
    if not np.all(weights > 0) or not np.isfinite(acc).all():
        raise UpscaleError("Invalid/nonfinite refinement blend buffers.")
    with span("refine_composite"):
        acc /= weights[..., None]
        if options.color_match:
            # Per-channel global mean shift only, not a hidden perceptual correction.
            acc += base.mean(axis=(0, 1), dtype=np.float64) - acc.mean(axis=(0, 1), dtype=np.float64)
        np.clip(acc, 0, 255, out=acc)
        amount = options.blend if protection is None else options.blend * (1 - protection[..., None])
        acc = base + (acc - base) * amount
        output = np.rint(np.clip(acc, 0, 255)).astype(np.uint8)
        if protection is not None:
            output[protection == 1] = base[protection == 1]
    return Image.fromarray(output), {"tiles": rows, "tile_count": len(rows),
                                     "executed_denoising_steps": sum(r.get("executed_denoising_steps", 0) for r in rows)}
