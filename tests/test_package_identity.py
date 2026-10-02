"""Regression checks for the distribution's public name and migration boundaries."""
from importlib import resources
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

import rastermoves
from rastermoves import Registry, Upscaler
from rastermoves import paths, plugins
from rastermoves.cli import _doctor, main
from rastermoves.downloads import Downloader
from rastermoves.errors import DownloadError
from rastermoves.network import session


def test_public_api():
    assert rastermoves.__version__ == "0.1.1"
    assert Registry.__module__ == "rastermoves.registry"
    assert Upscaler.__module__ == "rastermoves.pipeline"
    assert rastermoves.__all__ == ["Upscaler", "Registry", "__version__"]
    with pytest.raises(AttributeError):
        getattr(rastermoves, "not_an_api")


def test_root_import_does_not_load_neural_runtimes():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", (
            "import rastermoves, sys; "
            "assert rastermoves.__version__ == '0.1.1'; "
            "assert not ({'torch', 'spandrel', 'onnxruntime'} & set(sys.modules))"
        )],
        env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_cli_version_has_new_name(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert capsys.readouterr().out.strip() == "RasterMoves 0.1.1"


@pytest.mark.parametrize("command", [None, "models", "info", "sync", "download", "doctor", "upscale"])
def test_cli_help_uses_new_executable(command, capsys):
    args = [command, "--help"] if command else ["--help"]
    with pytest.raises(SystemExit) as caught:
        main(args)
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage: rastermoves" in help_text
    assert "upscalelab" not in help_text.casefold()


def test_doctor_version_key():
    data = _doctor()
    assert data["rastermoves"] == rastermoves.__version__
    assert "upscalelab" not in data


def test_http_user_agent_tracks_package_version():
    with session() as http:
        assert http.headers["User-Agent"] == f"RasterMoves/{rastermoves.__version__}"


def test_cache_environment_and_explicit_override(tmp_path, monkeypatch):
    configured = tmp_path / "configured-cache"
    explicit = tmp_path / "explicit-cache"
    monkeypatch.setenv("RASTERMOVES_CACHE", str(configured))
    assert paths.cache_dir() == configured
    assert configured.is_dir()
    assert paths.cache_dir(explicit) == explicit
    assert explicit.is_dir()


def test_default_platform_directories_use_new_name(tmp_path, monkeypatch):
    calls = []

    def cache_path(name):
        calls.append(("cache", name))
        return tmp_path / "cache" / name

    def config_path(name):
        calls.append(("config", name))
        return tmp_path / "config" / name

    monkeypatch.delenv("RASTERMOVES_CACHE", raising=False)
    monkeypatch.delenv("RASTERMOVES_PLUGIN_PATH", raising=False)
    monkeypatch.setattr(paths, "user_cache_path", cache_path)
    monkeypatch.setattr(paths, "user_config_path", config_path)
    assert paths.cache_dir() == tmp_path / "cache" / "rastermoves"
    assert paths.plugin_dirs() == [tmp_path / "config" / "rastermoves" / "models"]
    assert calls == [("cache", "rastermoves"), ("config", "rastermoves")]


def test_plugin_directory_environment(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setenv("RASTERMOVES_PLUGIN_PATH", os.pathsep.join([str(first), str(second)]))
    assert paths.plugin_dirs()[-2:] == [first, second]


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("YES", True), ("0", False)])
def test_offline_environment_name(tmp_path, monkeypatch, value, expected):
    monkeypatch.setenv("RASTERMOVES_OFFLINE", value)
    assert Downloader(tmp_path).offline is expected


def test_legacy_environment_is_not_silently_used(tmp_path, monkeypatch):
    monkeypatch.delenv("RASTERMOVES_CACHE", raising=False)
    monkeypatch.delenv("RASTERMOVES_OFFLINE", raising=False)
    monkeypatch.setenv("UPSCALELAB_CACHE", str(tmp_path / "old"))
    monkeypatch.setenv("UPSCALELAB_OFFLINE", "1")
    monkeypatch.setattr(paths, "user_cache_path", lambda _: tmp_path / "new")
    assert paths.cache_dir() == tmp_path / "new"
    assert not Downloader(tmp_path).offline


def test_reuse_existing_verified_cache_offline(tmp_path, weight_data):
    blob, resource = weight_data
    old_root = tmp_path / "existing-cache"
    downloader = Downloader(old_root, offline=True)
    # This is the same cache layout used before the rename.
    path = downloader.target(resource)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    assert downloader.get(resource) == path


def test_uncached_offline_error_uses_new_command(tmp_path, weight_data):
    _, resource = weight_data
    with pytest.raises(DownloadError, match="rastermoves download MODEL"):
        Downloader(tmp_path, offline=True).get(resource)


def test_eight_model_manifests_packaged_under_new_namespace(tmp_path):
    bundled = resources.files("rastermoves").joinpath("models")
    records = [json.loads(p.read_text(encoding="utf-8")) for p in bundled.iterdir() if p.name.endswith(".json")]
    assert len(records) == 8
    registry = Registry(tmp_path)
    for record in records:
        assert registry.get(record["id"]).spec.id == record["id"]
    assert resources.files("rastermoves").joinpath("py.typed").is_file()


def test_backend_entry_point_group_uses_new_name(monkeypatch):
    groups = []

    def entries(*, group):
        groups.append(group)
        return [SimpleNamespace(name="example", load=lambda: object)]

    monkeypatch.setattr(plugins, "entry_points", entries)
    plugins.backend_factories(external=False)
    assert groups == []
    assert plugins.backend_factories(external=True)["example"] is object
    assert groups == ["rastermoves.backends"]


def test_output_provenance_identifies_rastermoves(tmp_path, repeat_model):
    with Upscaler(cache_dir=tmp_path) as up:
        up.loaded = repeat_model
        up.upscale_image(Image.new("RGB", (6, 5), (25, 80, 120)), tile=0)
        assert up.last_report["software"] == "RasterMoves"
        assert up.last_report["software_version"] == rastermoves.__version__
