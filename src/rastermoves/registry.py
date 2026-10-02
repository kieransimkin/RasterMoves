from __future__ import annotations

from importlib import resources
from importlib.metadata import entry_points
import json
from pathlib import Path
from urllib.parse import urlparse, unquote

from .catalog import fetch_one
from .errors import UpscaleError
from .paths import cache_dir, plugin_dirs
from .plugins import ModelPlugin
from .specs import ModelSpec, validate_id


class Registry:
    def __init__(self, cache=None, *, model_dirs=(), external_plugins=False):
        self.cache = cache_dir(cache)
        self.plugins: dict[str, ModelPlugin] = {}
        for item in resources.files("rastermoves").joinpath("models").iterdir():
            if item.name.endswith(".json"):
                self._add(ModelSpec.from_dict(json.loads(item.read_text(encoding="utf-8"))))
        catalog = self.cache / "catalog.json"
        if catalog.exists():
            try:
                for item in json.loads(catalog.read_text(encoding="utf-8"))["models"]:
                    self._add(ModelSpec.from_dict(item))
            except (ValueError, KeyError, TypeError) as e:
                raise UpscaleError("Cached catalogue is invalid; remove catalog.json and run sync again.") from e
        dirs = [self.cache / "imported-models", *plugin_dirs(), *(Path(p) for p in model_dirs)]
        for directory in dirs:
            for path in sorted(directory.glob("*.json")):
                try:
                    self._add(ModelSpec.from_dict(json.loads(path.read_text(encoding="utf-8"))))
                except (ValueError, TypeError, KeyError, UpscaleError) as e:
                    raise UpscaleError(f"Invalid model plugin {path}: {e}") from e
        # JSON manifests are passive data. Third-party Python code requires explicit opt-in.
        if external_plugins:
            for ep in entry_points(group="rastermoves.models"):
                plugin = ep.load()()
                if not isinstance(plugin, ModelPlugin):
                    raise UpscaleError(f"Entry point {ep.name} must return a ModelPlugin instance.")
                if any(k.casefold() == plugin.spec.id.casefold() for k in self.plugins):
                    raise UpscaleError(f"External plugin duplicates an existing model ID: {plugin.spec.id}")
                self.plugins[plugin.spec.id] = plugin

    def _add(self, spec):
        key = spec.id.casefold()
        for old in list(self.plugins):
            if old.casefold() == key and old != spec.id:
                raise UpscaleError(f"Model IDs differ only by case: {old} and {spec.id}")
        self.plugins[spec.id] = ModelPlugin(spec)

    def get(self, model_id: str, *, auto_fetch=False, offline=False) -> ModelPlugin:
        if model_id.startswith("https://"):
            p = urlparse(model_id)
            if p.hostname != "openmodeldb.info" or not p.path.startswith("/models/"):
                raise UpscaleError("Model URL must be an OpenModelDB /models/ page.")
            model_id = unquote(p.path.rstrip("/").rsplit("/", 1)[-1])
        validate_id(model_id)
        for key, plugin in self.plugins.items():
            if key.casefold() == model_id.casefold():
                return plugin
        if auto_fetch:
            spec = fetch_one(model_id, self.cache, offline=offline)
            self._add(spec)
            return self.plugins[spec.id]
        raise UpscaleError(f"Unknown model: {model_id}. Run 'rastermoves sync' to import the full catalogue.")

    def search(self, query="", *, architecture=None, scale=None, tag=None):
        matches = []
        for plugin in self.plugins.values():
            s = plugin.spec
            text = " ".join([s.id, s.name, s.author, s.architecture, s.description, *s.tags]).casefold()
            if query.casefold() not in text:
                continue
            if architecture and architecture.casefold() != s.architecture.casefold():
                continue
            if scale is not None and scale != s.scale:
                continue
            if tag and tag.casefold() not in [t.casefold() for t in s.tags]:
                continue
            matches.append(s)
        return sorted(matches, key=lambda s: s.id.casefold())
