from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any
from urllib.parse import urlparse

from .errors import UpscaleError

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")


def validate_id(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value) or value.endswith("."):
        raise UpscaleError(f"Invalid model ID: {value!r}")
    return value


def validate_url(value: str) -> str:
    p = urlparse(value)
    if p.scheme != "https" or not p.hostname or p.username or p.password:
        raise UpscaleError("Model sources must be HTTPS URLs without embedded credentials.")
    return value


@dataclass(frozen=True)
class Resource:
    format: str
    urls: tuple[str, ...]
    sha256: str | None = None
    size: int | None = None
    backend: str = "spandrel"

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.format):
            raise UpscaleError("Invalid resource format.")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.backend):
            raise UpscaleError("Invalid backend name.")
        if self.sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", self.sha256):
            raise UpscaleError("Resource SHA-256 must contain exactly 64 hexadecimal digits.")
        if self.size is not None and (type(self.size) is not int or self.size <= 0):
            raise UpscaleError("Resource size must be a positive integer.")
        for url in self.urls:
            validate_url(url)
        object.__setattr__(self, "urls", tuple(self.urls))
        if self.sha256:
            object.__setattr__(self, "sha256", self.sha256.lower())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Resource:
        return cls(format=d["format"], urls=tuple(d.get("urls", [])),
                   sha256=d.get("sha256"), size=d.get("size"),
                   backend=d.get("backend", "spandrel"))


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    scale: int
    architecture: str = "auto"
    license: str = "unknown"
    author: str = "unknown"
    description: str = ""
    tags: tuple[str, ...] = ()
    resources: tuple[Resource, ...] = ()
    source_page: str | None = None
    input_channels: int = 3
    output_channels: int = 3
    # ONNX has no universal padding metadata. Overrides belong to each plugin.
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        validate_id(self.id)
        if type(self.scale) is not int or self.scale < 1 or self.scale > 64:
            raise UpscaleError(f"{self.id}: scale must be an integer in 1..64.")
        if not isinstance(self.name, str) or not self.name:
            raise UpscaleError("A model name is required.")
        for label in ("architecture", "license", "author", "description"):
            if not isinstance(getattr(self, label), str):
                raise UpscaleError(f"{self.id}: {label} must be text.")
        if any(not isinstance(t, str) for t in self.tags):
            raise UpscaleError(f"{self.id}: tags must be text values.")
        for channels in (self.input_channels, self.output_channels):
            if type(channels) is not int or channels < 1:
                raise UpscaleError(f"{self.id}: channel counts must be positive integers.")
        if not isinstance(self.options, dict):
            raise UpscaleError("Model options must be a JSON object.")
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "resources", tuple(self.resources))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelSpec:
        return cls(id=d["id"], name=d["name"], scale=d["scale"],
                   architecture=d.get("architecture", "auto"), license=d.get("license", "unknown"),
                   author=d.get("author", "unknown"), description=d.get("description", ""),
                   tags=tuple(d.get("tags", [])),
                   resources=tuple(Resource.from_dict(r) for r in d.get("resources", [])),
                   source_page=d.get("source_page"), input_channels=d.get("input_channels", 3),
                   output_channels=d.get("output_channels", 3), options=d.get("options", {}))
