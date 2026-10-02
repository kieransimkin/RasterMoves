"""Bounded HTTPS reads, retries and atomic writes. Never downloads source code for execution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import __version__
from .errors import DownloadError
from .specs import validate_url


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "HEAD"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = f"RasterMoves/{__version__}"
    return s


def stream_download(url: str, dest: Path, *, max_bytes: int, http=None) -> None:
    validate_url(url)
    own = http is None
    http = http or session()
    try:
        # Follow redirects explicitly so HTTPS is enforced at every hop.
        for _ in range(11):
            response = http.get(url, stream=True, timeout=(15, 90), allow_redirects=False)
            if response.is_redirect:
                from urllib.parse import urljoin
                next_url = urljoin(url, response.headers["Location"])
                response.close()
                validate_url(next_url)
                url = next_url
                continue
            break
        else:
            raise DownloadError("Too many download redirects.")
        with response:
            response.raise_for_status()
            length = response.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise DownloadError("Download exceeds the configured size limit.")
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise DownloadError("Host returned a web/login page, not a downloadable file.")
            n = 0
            with dest.open("wb") as f:
                for block in response.iter_content(1024 * 1024):
                    if not block:
                        continue
                    n += len(block)
                    if n > max_bytes:
                        raise DownloadError("Download exceeds the configured size limit.")
                    f.write(block)
                f.flush()
                os.fsync(f.fileno())
            if not n:
                raise DownloadError("Downloaded an empty file.")
    except requests.RequestException as e:
        # Do not echo signed URLs, private tokens or response bodies into logs.
        raise DownloadError(f"HTTPS request failed for {urlparse(url).hostname} ({type(e).__name__}).") from e
    finally:
        if own:
            http.close()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
