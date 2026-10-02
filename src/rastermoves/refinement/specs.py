"""Pinned workflow descriptors, separate from scale-changing model manifests.

No model repository code is imported. Python extensions are installed locally and
require the same explicit opt-in as RasterMoves' existing backend plugins.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import entry_points
import math
import re
from typing import Protocol

from PIL import Image

from ..errors import UpscaleError


@dataclass(frozen=True)
class Component:
    role: str
    repo_id: str
    revision: str
    license: str
    format: str = "safetensors"

    def __post_init__(self):
        if self.role not in {"base", "controlnet"}:
            raise UpscaleError("Component role must be base or controlnet.")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.repo_id, flags=re.ASCII):
            raise UpscaleError("Component source must be a Hugging Face owner/repository ID.")
        if not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise UpscaleError("Component revisions must be full immutable 40-character commit hashes.")
        if self.format not in {"safetensors", "restricted-bin"}:
            raise UpscaleError("Unsupported component weight format.")
        if self.role == "base" and self.format != "safetensors":
            raise UpscaleError("Base pipelines require safetensors.")

    def to_dict(self) -> dict:
        return {**asdict(self), "model_card": f"https://huggingface.co/{self.repo_id}"}


@dataclass(frozen=True)
class RefinerSpec:
    id: str
    family: str
    base: Component
    controlnet: Component
    default_tile: int

    def __post_init__(self):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", self.id):
            raise UpscaleError("Unsafe refiner ID.")
        if self.family not in {"sd15", "sdxl"}:
            raise UpscaleError("Built-in refiners support SD 1.5 or SDXL, not mixed families.")
        if self.base.role != "base" or self.controlnet.role != "controlnet":
            raise UpscaleError("Workflow components have incorrect roles.")
        if self.default_tile < 64 or self.default_tile % 8:
            raise UpscaleError("Default diffusion tile must be >=64 and a multiple of 8.")

    def to_dict(self) -> dict:
        return {"id": self.id, "family": self.family, "kind": "same-size-refiner",
                "default_tile": self.default_tile,
                "components": [self.base.to_dict(), self.controlnet.to_dict()]}


BUILTINS = {
    "sd15-tile": RefinerSpec(
        "sd15-tile", "sd15",
        Component("base", "stable-diffusion-v1-5/stable-diffusion-v1-5",
                  "451f4fe16113bff5a5d2269ed5ad43b0592e9a14", "CreativeML-OpenRAIL-M"),
        Component("controlnet", "lllyasviel/control_v11f1e_sd15_tile",
                  "3f877705c37010b7221c3d10743307d6b5b6efac", "OpenRAIL", "restricted-bin"), 512),
    "sdxl-tile": RefinerSpec(
        "sdxl-tile", "sdxl",
        Component("base", "stabilityai/stable-diffusion-xl-base-1.0",
                  "462165984030d82259a11f4367a4eed129e94a7b", "CreativeML-OpenRAIL++-M"),
        Component("controlnet", "xinsir/controlnet-tile-sdxl-1.0",
                  "1ae8d9529efe58f7362a987363ff86a7904dc84f", "Apache-2.0"), 1024),
}


@dataclass(frozen=True)
class RefineOptions:
    strength: float = 0.22
    steps: int = 40
    guidance: float = 5.0
    control_scale: float = 1.0
    seed: int = 0
    prompt: str = ""
    negative_prompt: str = ""
    tile: int | None = None
    overlap: int = 64
    blend: float = 1.0
    color_match: bool = False
    mask_feather: float = 0.0
    max_output_mp: float = 64.0

    def validate(self, spec: RefinerSpec) -> RefineOptions:
        for key in ("strength", "guidance", "control_scale", "blend", "mask_feather", "max_output_mp"):
            if not math.isfinite(getattr(self, key)):
                raise UpscaleError(f"{key} must be finite.")
        if not 0 <= self.strength <= 1 or not 0 <= self.blend <= 1:
            raise UpscaleError("Refinement strength and blend must be between 0 and 1.")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or not 1 <= self.steps <= 1000:
            raise UpscaleError("Refinement steps must be an integer between 1 and 1000.")
        if self.strength > 0 and int(self.steps * self.strength) < 1:
            raise UpscaleError("Refinement strength * steps must permit at least one denoising step.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise UpscaleError("Seed must be an integer from 0 through 2**63-1.")
        tile = self.tile if self.tile is not None else spec.default_tile
        if isinstance(tile, bool) or not isinstance(tile, int) or tile < 64 or tile % 8:
            raise UpscaleError("Diffusion tile must be >=64 and a multiple of 8; it is separate from --tile.")
        if (isinstance(self.overlap, bool) or not isinstance(self.overlap, int)
                or not 0 <= self.overlap < tile):
            raise UpscaleError("Refinement overlap must be an integer >=0 and smaller than its tile.")
        if self.guidance < 0 or self.control_scale < 0 or self.mask_feather < 0 or self.max_output_mp <= 0:
            raise UpscaleError("Guidance, control scale and feather must be nonnegative; output limit must be positive.")
        if not isinstance(self.prompt, str) or not isinstance(self.negative_prompt, str):
            raise UpscaleError("Prompts must be strings.")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


class RefinerBackend(Protocol):
    """Locally installed backend contract. Returned tiles must be RGB and same-size."""
    def predict(self, image: Image.Image, *, options: RefineOptions, seed: int) -> tuple[Image.Image, dict]: ...
    def close(self) -> None: ...


def get_spec(refiner: str, *, external_plugins: bool = False) -> tuple[RefinerSpec, object | None]:
    if refiner in BUILTINS:
        return BUILTINS[refiner], None
    if external_plugins:
        matches = [e for e in entry_points(group="rastermoves.refiners") if e.name == refiner]
        if len(matches) > 1:
            raise UpscaleError(f"Duplicate installed refiner: {refiner}")
        if matches:
            # Trusted local entry point returns (RefinerSpec, backend constructor).
            spec, factory = matches[0].load()()
            if not isinstance(spec, RefinerSpec) or spec.id != refiner or not callable(factory):
                raise UpscaleError("Refiner factory must return (matching RefinerSpec, backend constructor).")
            return spec, factory
    raise UpscaleError(f"Unknown refiner {refiner!r}. Available built-ins: {', '.join(BUILTINS)}.")
