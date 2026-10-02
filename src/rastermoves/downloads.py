from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse, unquote

from filelock import FileLock

from .errors import DownloadError, IntegrityError
from .network import atomic_json, stream_download
from .paths import cache_dir
from .specs import Resource

log = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def huggingface_parts(url: str):
    p = urlparse(url)
    parts = p.path.strip("/").split("/")
    if p.hostname not in {"huggingface.co", "www.huggingface.co"}:
        return None
    repo_type = "model"
    if parts and parts[0] in {"datasets", "spaces"}:
        repo_type = {"datasets": "dataset", "spaces": "space"}[parts.pop(0)]
    if len(parts) < 5 or parts[2] not in {"resolve", "blob"}:
        raise DownloadError("Hugging Face sources must point to one file, not a repository or folder.")
    return "/".join(parts[:2]), unquote(parts[3]), unquote("/".join(parts[4:])), repo_type


def source_kind(url: str) -> str:
    p = urlparse(url)
    host = p.hostname or ""
    if host in {"huggingface.co", "www.huggingface.co"}:
        return "huggingface"
    if host == "drive.google.com":
        return "manual" if "/folders/" in p.path else "gdrive"
    if host in {"mega.nz", "mega.co.nz", "u.pcloud.link", "icedrive.net", "1drv.ms"} or "sharepoint.com" in host:
        return "manual"
    if p.path.lower().endswith((".zip", ".7z", ".rar", ".tar", ".gz")):
        return "manual"
    return "https"


def normalize_url(url: str) -> str:
    p = urlparse(url)
    if p.hostname == "github.com" and "/blob/" in p.path:
        return url.replace("/blob/", "/raw/", 1)
    if p.hostname in {"dropbox.com", "www.dropbox.com"}:
        q = parse_qs(p.query)
        q["dl"] = ["1"]
        return urlunparse(p._replace(query=urlencode(q, doseq=True)))
    return url


class Downloader:
    """Content-addressed model cache. Verified on every read, locked and atomically published."""
    def __init__(self, cache=None, *, offline=False, strict_checksums=False,
                 max_bytes=4 * 1024**3):
        self.root = cache_dir(cache)
        self.offline = offline or os.environ.get("RASTERMOVES_OFFLINE", "").lower() in {"1", "true", "yes"}
        self.strict_checksums = strict_checksums
        self.max_bytes = max_bytes
        if max_bytes <= 0:
            raise DownloadError("max_bytes must be positive.")

    def target(self, r: Resource) -> Path:
        key = r.sha256 or hashlib.sha256(json.dumps(r.urls).encode()).hexdigest()
        return self.root / "weights" / f"{key}.{r.format}"

    def _verify(self, path: Path, r: Resource, *, cached=False) -> str:
        n = path.stat().st_size
        if n == 0 or n > self.max_bytes or (r.size is not None and n != r.size):
            raise IntegrityError("Model file size does not match the manifest or configured limit.")
        with path.open("rb") as f:
            prefix = f.read(256).lstrip().lower()
        if prefix.startswith((b"<!doctype html", b"<html", b"version https://git-lfs.github.com/spec")):
            raise IntegrityError("Received HTML or a Git LFS pointer instead of model weights.")
        digest = sha256_file(path)
        if r.sha256 and digest != r.sha256:
            raise IntegrityError("Model SHA-256 does not match the model manifest.")
        if cached and not r.sha256:
            receipt = path.with_suffix(path.suffix + ".json")
            try:
                expected = json.loads(receipt.read_text(encoding="utf-8"))["sha256"]
            except (OSError, ValueError, KeyError) as e:
                raise IntegrityError("Unverified cached file has no valid integrity receipt.") from e
            if digest != expected:
                raise IntegrityError("Cached file differs from its original download.")
        return digest

    def get(self, r: Resource) -> Path:
        if self.strict_checksums and not r.sha256:
            raise IntegrityError("No publisher checksum available; strict checksum mode refuses this model.")
        if r.size and r.size > self.max_bytes:
            raise DownloadError("Model exceeds the configured download limit.")
        path = self.target(r)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock", timeout=600):
            if path.exists():
                try:
                    self._verify(path, r, cached=True)
                    return path
                except IntegrityError:
                    if self.offline:
                        raise
                    log.warning("Cached weights failed verification; downloading a fresh copy.")
            if self.offline:
                raise DownloadError("Weights are not cached. Run 'rastermoves download MODEL' online first.")
            if not r.sha256:
                log.warning("This model has no publisher checksum; using a local integrity receipt only.")
            errors = []
            temp = path.with_suffix(path.suffix + ".part")
            for url in sorted(r.urls, key=lambda u: {"huggingface": 0, "https": 1, "gdrive": 2, "manual": 3}[source_kind(u)]):
                try:
                    temp.unlink(missing_ok=True)
                    self._fetch(url, temp, r)
                    digest = self._verify(temp, r)
                    os.replace(temp, path)
                    atomic_json(path.with_suffix(path.suffix + ".json"), {
                        "sha256": digest, "publisher_verified": bool(r.sha256), "size": path.stat().st_size,
                        "source_host": urlparse(url).hostname,
                    })
                    return path
                except (DownloadError, OSError) as e:
                    errors.append(f"{urlparse(url).hostname}: {e}")
                finally:
                    temp.unlink(missing_ok=True)
            raise DownloadError("No usable model source. " + "; ".join(errors or ["No URLs listed."]) +
                                " Supply a local checkpoint with --model-file or add an explicit mirror to a plugin.")

    def _fetch(self, url: str, dest: Path, r: Resource):
        kind = source_kind(url)
        log.info("Downloading weights from %s", urlparse(url).hostname)
        if kind == "manual":
            raise DownloadError("This host, archive or folder needs a direct file mirror/manual download.")
        if kind == "huggingface":
            from huggingface_hub import hf_hub_download
            repo, revision, filename, repo_type = huggingface_parts(url)
            try:
                # HF_TOKEN is read by huggingface_hub; never passed to other hosts.
                source = Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                              repo_type=repo_type, cache_dir=self.root / "huggingface",
                              local_files_only=self.offline))
                if source.stat().st_size > self.max_bytes:
                    raise DownloadError("Hugging Face file exceeds the configured size limit.")
                shutil.copyfile(source, dest)
            except DownloadError:
                raise
            except Exception as e:
                raise DownloadError(f"Hugging Face download failed ({type(e).__name__}). Check access and HF_TOKEN.") from e
        elif kind == "gdrive":
            try:
                import gdown
            except ImportError as e:
                raise DownloadError("Install the gdrive extra from the source directory: python -m pip install -e '.[gdrive]'.") from e
            try:
                result = gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True, use_cookies=False)
                if result is None or not dest.exists():
                    raise DownloadError("Google Drive file unavailable or over quota.")
            except DownloadError:
                raise
            except Exception as e:
                raise DownloadError(f"Google Drive download failed ({type(e).__name__}).") from e
        else:
            stream_download(normalize_url(url), dest, max_bytes=min(self.max_bytes, r.size or self.max_bytes))
