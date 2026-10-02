import io
import json
from pathlib import Path
import tarfile

import pytest

from rastermoves.catalog import from_openmodeldb, convert_catalog, sync_catalog, _read_archive, fetch_one
from rastermoves.errors import UpscaleError, DownloadError
from rastermoves.registry import Registry
from rastermoves.specs import ModelSpec, Resource, validate_id


def test_bundled_plugins_have_verified_resources(tmp_path):
    registry = Registry(tmp_path)
    assert len(registry.search()) == 8
    assert {s.architecture for s in registry.search()} == {"esrgan", "compact", "hat", "dat", "span"}
    for s in registry.search():
        assert s.resources and all(r.sha256 and r.size for r in s.resources)
        assert s.source_page.startswith("https://openmodeldb.info/models/")
        assert ModelSpec.from_dict(s.to_dict()) == s


def test_registry_lookup_filter_and_url(tmp_path):
    r = Registry(tmp_path)
    assert r.get("4X-ULTRASHARPV2").spec.id == "4x-UltraSharpV2"
    assert r.get("https://openmodeldb.info/models/4x-LexicaHAT/").spec.architecture == "hat"
    assert len(r.search(architecture="compact", scale=4)) == 2
    assert len(r.search("", tag="ai-generated")) == 2
    assert len(r.search("nomos", scale=2)) == 1
    with pytest.raises(UpscaleError):
        r.get("https://evil.example/models/4x-LexicaHAT")


@pytest.mark.parametrize("value", ["../x", "x/y", "x\\y", "", ".dot", "with spaces", "trailing.", "a"*161])
def test_invalid_ids(value):
    with pytest.raises(UpscaleError):
        validate_id(value)


@pytest.mark.parametrize("kwargs", [
    {"format": "../pth", "urls": []}, {"format": "pth", "urls": ["http://example.com/a"]},
    {"format": "pth", "urls": ["https://user:password@example.com/a"]},
    {"format": "pth", "urls": [], "sha256": "bad"}, {"format": "pth", "urls": [], "size": -1},
])
def test_invalid_resources(kwargs):
    with pytest.raises(UpscaleError):
        Resource(**kwargs)


def test_import_original_schema(raw_model):
    s = from_openmodeldb("2x-Test", raw_model)
    assert s.scale == 2 and s.resources[0].backend == "spandrel"
    assert s.resources[0].sha256 == "a" * 64
    raw_model["author"] = ["one", "two"]
    assert from_openmodeldb("2x-Test", raw_model).author == "one, two"


def test_partial_catalog_validation(raw_model):
    models, skipped = convert_catalog({"2x-good": raw_model, "../bad": raw_model, "not-an-object": 5})
    assert len(models) == 1 and len(skipped) == 2


def test_sync_local_export_and_keep_old_on_error(tmp_path, raw_model):
    cache = tmp_path / "cache"
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"2x-Test": raw_model}))
    report = sync_catalog(cache, str(source), offline=True)
    assert report["imported"] == 1
    assert Registry(cache).get("2x-Test").spec.name == "Example"
    before = (cache / "catalog.json").read_bytes()
    source.write_text('{}')
    with pytest.raises(UpscaleError):
        sync_catalog(cache, str(source), offline=True)
    assert (cache / "catalog.json").read_bytes() == before


def test_sync_repository_directory(tmp_path, raw_model):
    source = tmp_path / "repo" / "data" / "models"
    source.mkdir(parents=True)
    (source / "2x-Test.json").write_text(json.dumps(raw_model))
    assert sync_catalog(tmp_path / "cache", str(tmp_path / "repo"))["imported"] == 1


def test_user_manifest_overrides_bundle(tmp_path):
    p = tmp_path / "plugins"
    p.mkdir()
    spec = Registry(tmp_path / "cache").get("4x-LexicaHAT").spec.to_dict()
    spec["description"] = "local override"
    (p / "custom.json").write_text(json.dumps(spec))
    assert Registry(tmp_path / "cache", model_dirs=[p]).get(spec["id"]).spec.description == "local override"


def test_archive_reads_models_without_extracting(tmp_path, raw_model):
    archive = tmp_path / "archive.tgz"
    content = json.dumps(raw_model).encode()
    with tarfile.open(archive, "w:gz") as tar:
        for name in ["repo-main/data/models/2x-Test.json", "../../escaped.json", "repo-main/package.json"]:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    assert list(_read_archive(archive)) == ["2x-Test"]
    assert not (tmp_path / "escaped.json").exists()


def test_default_sync_falls_back_to_archive(tmp_path, raw_model, monkeypatch):
    content = json.dumps(raw_model).encode()
    calls = []
    def download(url, dest, **kwargs):
        calls.append(url)
        if url.endswith("models.json"):
            raise DownloadError("unavailable")
        with tarfile.open(dest, "w:gz") as tar:
            info = tarfile.TarInfo("repo-main/data/models/2x-Test.json")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    monkeypatch.setattr("rastermoves.catalog.stream_download", download)
    assert sync_catalog(tmp_path)["imported"] == 1
    assert len(calls) == 2


def test_unknown_offline_does_not_request_network(tmp_path, monkeypatch):
    monkeypatch.setattr("rastermoves.catalog.stream_download", lambda *a, **k: pytest.fail("network access"))
    with pytest.raises(DownloadError, match="local catalogue"):
        Registry(tmp_path).get("4x-Missing", auto_fetch=True, offline=True)


def test_lazy_model_import_is_cached(tmp_path, raw_model, monkeypatch):
    def download(url, dest, **kwargs):
        dest.write_text(json.dumps(raw_model))
    monkeypatch.setattr("rastermoves.catalog.stream_download", download)
    assert Registry(tmp_path).get("2x-Test", auto_fetch=True).spec.scale == 2
    monkeypatch.setattr("rastermoves.catalog.stream_download", lambda *a, **k: pytest.fail("network access"))
    assert Registry(tmp_path).get("2x-Test", auto_fetch=True, offline=True).spec.scale == 2
