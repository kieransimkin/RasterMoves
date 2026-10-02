from dataclasses import replace
import json

import pytest

from rastermoves.downloads import Downloader, huggingface_parts, normalize_url, source_kind
from rastermoves.errors import DownloadError, IntegrityError, UpscaleError
from rastermoves.network import stream_download


def test_download_checksum_cache_and_offline(tmp_path, weight_data, monkeypatch):
    blob, resource = weight_data
    dl = Downloader(tmp_path)
    calls = []
    def fetch(url, dest, r):
        calls.append(url)
        dest.write_bytes(blob)
    monkeypatch.setattr(dl, "_fetch", fetch)
    path = dl.get(resource)
    assert path.read_bytes() == blob
    assert dl.get(resource) == path and len(calls) == 1
    offline = Downloader(tmp_path, offline=True)
    monkeypatch.setattr(offline, "_fetch", lambda *a: pytest.fail("network access"))
    assert offline.get(resource) == path


def test_corrupt_cache_is_repaired_online_and_rejected_offline(tmp_path, weight_data, monkeypatch):
    blob, resource = weight_data
    dl = Downloader(tmp_path)
    monkeypatch.setattr(dl, "_fetch", lambda url, dest, r: dest.write_bytes(blob))
    path = dl.get(resource)
    path.write_bytes(b"x" * len(blob))
    with pytest.raises(IntegrityError):
        Downloader(tmp_path, offline=True).get(resource)
    assert dl.get(resource).read_bytes() == blob


def test_mirror_fallback_does_not_publish_bad_file(tmp_path, weight_data, monkeypatch):
    blob, resource = weight_data
    resource = replace(resource, urls=("https://bad.example/a.pth", "https://good.example/a.pth"))
    dl = Downloader(tmp_path)
    def fetch(url, dest, r):
        dest.write_bytes(b"x" * len(blob) if "bad." in url else blob)
    monkeypatch.setattr(dl, "_fetch", fetch)
    assert dl.get(resource).read_bytes() == blob
    assert not list((tmp_path / "weights").glob("*.part"))


def test_all_failures_leave_no_weight(tmp_path, weight_data, monkeypatch):
    blob, resource = weight_data
    dl = Downloader(tmp_path)
    monkeypatch.setattr(dl, "_fetch", lambda url, dest, r: dest.write_bytes(b"bad"))
    with pytest.raises(DownloadError):
        dl.get(resource)
    assert not dl.target(resource).exists()
    assert not list((tmp_path / "weights").glob("*.part"))


def test_unknown_checksum_receipt_and_strict_mode(tmp_path, weight_data, monkeypatch):
    blob, resource = weight_data
    resource = replace(resource, sha256=None)
    dl = Downloader(tmp_path)
    monkeypatch.setattr(dl, "_fetch", lambda url, dest, r: dest.write_bytes(blob))
    path = dl.get(resource)
    assert dl.get(resource) == path
    receipt = json.loads(path.with_suffix(".pth.json").read_text())
    assert not receipt["publisher_verified"]
    with pytest.raises(IntegrityError):
        Downloader(tmp_path, strict_checksums=True).get(resource)
    path.write_bytes(b"x" * len(blob))
    with pytest.raises(IntegrityError):
        Downloader(tmp_path, offline=True).get(resource)


@pytest.mark.parametrize("content", [b"<!DOCTYPE html><html>login", b"<html>login</html>",
                                     b"version https://git-lfs.github.com/spec/v1\n"])
def test_html_lfs_rejected(tmp_path, weight_data, content):
    _, resource = weight_data
    resource = replace(resource, size=None, sha256=None)
    p = tmp_path / "weights.pth"
    p.write_bytes(content)
    with pytest.raises(IntegrityError):
        Downloader(tmp_path)._verify(p, resource)


def test_size_limit_prevents_fetch(tmp_path, weight_data, monkeypatch):
    _, resource = weight_data
    dl = Downloader(tmp_path, max_bytes=1)
    monkeypatch.setattr(dl, "_fetch", lambda *a: pytest.fail("network access"))
    with pytest.raises(DownloadError):
        dl.get(resource)


def test_hf_blob_and_resolve_urls():
    assert huggingface_parts("https://huggingface.co/a/b/blob/main/sub/model%20one.pth?download=true") == (
        "a/b", "main", "sub/model one.pth", "model")
    assert huggingface_parts("https://huggingface.co/datasets/a/b/resolve/refs%2Fpr%2F2/m.onnx") == (
        "a/b", "refs/pr/2", "m.onnx", "dataset")
    with pytest.raises(DownloadError):
        huggingface_parts("https://huggingface.co/a/b")


def test_sources():
    assert source_kind("https://mega.nz/file/123") == "manual"
    assert source_kind("https://drive.google.com/drive/folders/123") == "manual"
    assert source_kind("https://drive.google.com/file/d/123/view") == "gdrive"
    assert source_kind("https://example.org/model.zip") == "manual"
    assert normalize_url("https://github.com/a/b/blob/main/m.pth") == "https://github.com/a/b/raw/main/m.pth"
    assert "dl=1" in normalize_url("https://www.dropbox.com/s/x/model.pth?dl=0")


class Response:
    def __init__(self, blocks=(), headers=None, redirect=False):
        self.blocks = blocks
        self.headers = headers or {}
        self.is_redirect = redirect
    def close(self): pass
    def raise_for_status(self): pass
    def iter_content(self, n): return iter(self.blocks)
    def __enter__(self): return self
    def __exit__(self, *args): pass


class HTTP:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
    def get(self, url, **kwargs):
        self.calls.append(url)
        return next(self.responses)


def test_stream_download_bounded_and_valid(tmp_path):
    http = HTTP([Response([b"123", b"45"])])
    p = tmp_path / "data"
    stream_download("https://example.org/file", p, max_bytes=5, http=http)
    assert p.read_bytes() == b"12345"
    with pytest.raises(DownloadError):
        stream_download("https://example.org/file", p, max_bytes=4, http=HTTP([Response([b"12345"])]))


def test_stream_rejects_html_and_https_downgrade(tmp_path):
    with pytest.raises(DownloadError):
        stream_download("https://example.org/file", tmp_path / "f", max_bytes=10,
                        http=HTTP([Response(headers={"Content-Type": "text/html"})]))
    http = HTTP([Response(headers={"Location": "http://evil.example/file"}, redirect=True)])
    with pytest.raises(UpscaleError):
        stream_download("https://example.org/file", tmp_path / "f", max_bytes=10, http=http)
    assert len(http.calls) == 1


def test_hf_client_receives_exact_repository_revision_and_file(tmp_path, monkeypatch, weight_data):
    import huggingface_hub
    blob, resource = weight_data
    source = tmp_path / "hf-cached.pth"
    source.write_bytes(blob)
    captured = {}
    def hf_download(**kwargs):
        captured.update(kwargs)
        return str(source)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_download)
    resource = replace(resource, urls=("https://huggingface.co/owner/repo/blob/release/models/test%20file.pth",))
    dl = Downloader(tmp_path / "app-cache")
    assert dl.get(resource).read_bytes() == blob
    assert captured["repo_id"] == "owner/repo"
    assert captured["revision"] == "release"
    assert captured["filename"] == "models/test file.pth"
    assert captured["repo_type"] == "model"
    assert captured["local_files_only"] is False
    assert "token" not in captured
