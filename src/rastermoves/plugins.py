from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from importlib.util import find_spec
from pathlib import Path

from .errors import UpscaleError
from .specs import ModelSpec, Resource
from .tracing import span


def backend_factories(*, external=False) -> dict:
    from .backends.spandrel_backend import SpandrelModel
    from .backends.onnx_backend import OnnxModel
    result = {"spandrel": SpandrelModel, "onnx": OnnxModel}
    if external:
        for ep in entry_points(group="rastermoves.backends"):
            if ep.name in result:
                raise UpscaleError(f"Duplicate backend entry point: {ep.name}")
            result[ep.name] = ep.load()
    return result


@dataclass
class ModelPlugin:
    """One independently addressable plugin per model; override load() for custom model logic."""
    spec: ModelSpec

    def resources(self, backend="auto") -> list[Resource]:
        items = [r for r in self.spec.resources if backend == "auto" or r.backend == backend]
        preferred = "spandrel" if find_spec("spandrel") else "onnx" if find_spec("onnxruntime") else "spandrel"
        return sorted(items, key=lambda r: (
            r.backend != preferred,
            {"safetensors": 0, "pth": 1, "pt": 2, "ckpt": 3, "onnx": 4}.get(r.format, 99)))

    def load(self, downloader, *, backend="auto", device="auto", precision="fp32",
             extra_arches=False, external_backends=False):
        factories = backend_factories(external=external_backends)
        errors = []
        for resource in self.resources(backend):
            factory = factories.get(resource.backend)
            if factory is None:
                errors.append(f"No backend plugin for {resource.backend}")
                continue
            if resource.backend == "spandrel" and resource.format not in {"pth", "pt", "ckpt", "safetensors"}:
                errors.append(f"Unsupported checkpoint format: {resource.format}")
                continue
            if resource.backend == "onnx" and resource.format != "onnx":
                continue
            # Fail before downloading hundreds of MB when a built-in runtime is absent.
            module = {"spandrel": "spandrel", "onnx": "onnxruntime"}.get(resource.backend)
            if module and find_spec(module) is None:
                errors.append(f"{resource.backend} runtime is not installed")
                continue
            try:
                path = downloader.get(resource)
                with span("backend_load", backend=resource.backend, model_id=self.spec.id):
                    loaded = factory(path, self.spec, device=device, precision=precision, extra_arches=extra_arches)
                return loaded, path, resource
            except UpscaleError as e:
                errors.append(f"{resource.backend}/{resource.format}: {e}")
        raise UpscaleError(f"Could not load {self.spec.id}. " + "; ".join(errors or ["No matching weight resources."]))
