"""Explicitly installed Python plugin contracts, without loading any remote code."""
from types import SimpleNamespace

import pytest

from rastermoves.errors import UpscaleError
from rastermoves.plugins import ModelPlugin
from rastermoves.registry import Registry
from rastermoves.specs import ModelSpec, Resource
import rastermoves.plugins as plugin_module
import rastermoves.registry as registry_module


def test_python_model_discovery_requires_opt_in(tmp_path, monkeypatch):
    seen = []
    plugin = ModelPlugin(ModelSpec("external-model", "External model", 2))
    def entry_points(*, group):
        seen.append(group)
        return [SimpleNamespace(name="example", load=lambda: lambda: plugin)]
    monkeypatch.setattr(registry_module, "entry_points", entry_points)
    Registry(tmp_path)
    assert seen == []
    registry = Registry(tmp_path, external_plugins=True)
    assert registry.get("external-model") is plugin
    assert seen == ["rastermoves.models"]


def test_python_model_case_insensitive_collision(tmp_path, monkeypatch):
    plugin = ModelPlugin(ModelSpec("4X-REALESRGAN-X4PLUS", "Duplicate", 4))
    monkeypatch.setattr(registry_module, "entry_points", lambda **_: [
        SimpleNamespace(name="duplicate", load=lambda: lambda: plugin)])
    with pytest.raises(UpscaleError, match="duplicates"):
        Registry(tmp_path, external_plugins=True)


def test_custom_backend_loads_through_model_plugin(tmp_path, monkeypatch, repeat_model):
    path = tmp_path / "local.testweights"
    path.write_bytes(b"test-only")
    resource = Resource(format="testweights", backend="test-backend", urls=())
    spec = ModelSpec("custom", "Custom backend model", 2, resources=(resource,))
    received = {}
    def factory(file, model, **options):
        received.update(file=file, model=model, options=options)
        return repeat_model
    monkeypatch.setattr(plugin_module, "entry_points", lambda **_: [
        SimpleNamespace(name="test-backend", load=lambda: factory)])
    downloader = SimpleNamespace(get=lambda r: path)
    loaded, actual_path, actual_resource = ModelPlugin(spec).load(downloader, external_backends=True)
    assert loaded is repeat_model and actual_path == path and actual_resource == resource
    assert received["model"] == spec
    assert received["options"]["precision"] == "fp32"
