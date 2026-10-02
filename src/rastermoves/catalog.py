"""Import OpenModelDB's exported JSON, individual entries, or a checked-out data directory."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from filelock import FileLock

from .errors import DownloadError, UpscaleError
from .network import atomic_json, stream_download
from .specs import ModelSpec, Resource, validate_id

API_URL = "https://openmodeldb.info/api/v1/models.json"
ARCHIVE_URL = "https://codeload.github.com/OpenModelDB/open-model-database/tar.gz/refs/heads/main"
RAW_BASE = "https://raw.githubusercontent.com/OpenModelDB/open-model-database/main/data/models/"


def from_openmodeldb(model_id: str, data: dict) -> ModelSpec:
    validate_id(model_id)
    resources = []
    for r in data.get("resources", []):
        fmt = str(r.get("type", "unknown")).lower().lstrip(".")
        platform = str(r.get("platform", "unknown")).lower()
        backend = {"pytorch": "spandrel", "onnx": "onnx"}.get(platform, platform)
        resources.append(Resource(format=fmt, urls=tuple(r.get("urls", [])),
                                  sha256=r.get("sha256"), size=r.get("size"), backend=backend))
    author = data.get("author", "unknown")
    if isinstance(author, list):
        author = ", ".join(str(x) for x in author)
    license_value = data.get("license", "unknown")
    if not isinstance(license_value, str):
        license_value = json.dumps(license_value, ensure_ascii=False)
    return ModelSpec(id=model_id, name=data["name"], scale=data["scale"],
                     architecture=data.get("architecture", "unknown"), license=license_value,
                     author=str(author), description=data.get("description", ""),
                     tags=tuple(data.get("tags", [])), resources=tuple(resources),
                     source_page=f"https://openmodeldb.info/models/{model_id}",
                     input_channels=data.get("inputChannels", 3),
                     output_channels=data.get("outputChannels", 3))


def convert_catalog(raw) -> tuple[list[ModelSpec], list[str]]:
    if not isinstance(raw, dict):
        raise UpscaleError("Expected an OpenModelDB JSON object keyed by model ID.")
    if "models" in raw and isinstance(raw["models"], dict):
        raw = raw["models"]
    converted, skipped = [], []
    for key, value in raw.items():
        try:
            if not isinstance(value, dict):
                raise UpscaleError("model entry is not an object")
            converted.append(from_openmodeldb(key, value))
        except (KeyError, TypeError, ValueError, UpscaleError) as e:
            skipped.append(f"{key}: {e}")
    if not converted:
        raise UpscaleError("No valid models found; the previous catalogue was not changed.")
    return converted, skipped


def _read_archive(path: Path) -> dict:
    data, expanded = {}, 0
    with tarfile.open(path, "r:gz") as archive:
        # Read model JSON entries only. Do not extract any paths or execute repository code.
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if (len(parts) != 4 or parts[1:3] != ("data", "models")
                    or not parts[-1].endswith(".json") or not member.isfile()):
                continue
            if member.size > 1024 * 1024:
                raise UpscaleError("Unexpectedly large model metadata entry.")
            expanded += member.size
            if expanded > 64 * 1024**2 or len(data) > 20000:
                raise UpscaleError("Catalogue archive exceeds metadata limits.")
            f = archive.extractfile(member)
            if f is not None:
                with f:
                    data[parts[-1][:-5]] = json.load(f)
    return data


def sync_catalog(cache: Path, source: str | None = None, *, offline=False) -> dict:
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    origin = source or API_URL
    with FileLock(str(cache / "catalog.lock"), timeout=600), tempfile.TemporaryDirectory() as tmp:
        path = Path(origin).expanduser() if not origin.startswith("https://") else None
        if path is not None and path.is_dir():
            data_dir = path / "data" / "models" if (path / "data" / "models").is_dir() else path
            raw = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(data_dir.glob("*.json"))}
        elif path is not None and path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
        elif path is not None:
            raise UpscaleError(f"Catalogue source does not exist: {path}")
        else:
            if offline:
                raise DownloadError("Offline mode forbids catalogue network requests; use a local JSON file.")
            dest = Path(tmp) / "catalog.json"
            try:
                stream_download(origin, dest, max_bytes=64 * 1024**2)
                raw = json.loads(dest.read_text(encoding="utf-8"))
            except (DownloadError, ValueError):
                if source is not None:
                    raise
                origin = ARCHIVE_URL
                dest = Path(tmp) / "catalog.tar.gz"
                stream_download(origin, dest, max_bytes=128 * 1024**2)
                raw = _read_archive(dest)
        specs, skipped = convert_catalog(raw)
        payload = {"schema_version": 1, "synced_at": datetime.now(timezone.utc).isoformat(),
                   "source": origin, "models": [s.to_dict() for s in specs], "skipped": skipped}
        atomic_json(cache / "catalog.json", payload)
        return {"imported": len(specs), "skipped": skipped, "source": origin}


def fetch_one(model_id: str, cache: Path, *, offline=False) -> ModelSpec:
    validate_id(model_id)
    if offline:
        raise DownloadError(f"{model_id} is not in the local catalogue; run sync online first.")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        stream_download(RAW_BASE + model_id + ".json", path, max_bytes=1024**2)
        spec = from_openmodeldb(model_id, json.loads(path.read_text(encoding="utf-8")))
        atomic_json(Path(cache) / "imported-models" / f"{model_id}.json", spec.to_dict())
        return spec
