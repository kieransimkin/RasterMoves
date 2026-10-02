"""Release validation and deterministic archives; run with Python 3.10+ plus dev extras."""
from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import gzip
import hashlib
import io
import os
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, only for developer tooling.
    import tomli as tomllib

TAG = re.compile(r"^v([0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+|\.dev[0-9]+|\.post[0-9]+)?)$")
ROOT = Path(__file__).resolve().parents[1]


def version_from_source(root: Path = ROOT) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project["name"] != "rastermoves":
        raise ValueError("Distribution name must remain rastermoves.")
    tree = ast.parse((root / "src/rastermoves/__init__.py").read_text(encoding="utf-8"))
    versions = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)]
    if versions != [project["version"]]:
        raise ValueError("pyproject.toml and rastermoves.__version__ disagree.")
    if not TAG.fullmatch("v" + project["version"]):
        raise ValueError("Use a canonical X.Y.Z version, optionally with a/b/rc, .dev or .post suffix.")
    return project["version"]


def release_metadata(tag: str, repository: str, root: Path = ROOT, *, check_git=True, allow_untagged=False) -> dict[str, str]:
    match = TAG.fullmatch(tag)
    if not match and not (allow_untagged and not tag):
        raise ValueError("Release tag must be vX.Y.Z (or a canonical Python prerelease, such as v0.3.0rc1).")
    version = version_from_source(root)
    if match and match.group(1) != version:
        raise ValueError(f"Tag {tag} does not match package version {version}; update both version declarations first.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid owner/repository.")
    result = {"tag": tag, "version": version, "image": "ghcr.io/" + repository.lower(),
              "prerelease": str(bool(re.search(r"(?:a|b|rc|\.dev)[0-9]+$", version))).lower()}
    if check_git:
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
        head = git("rev-parse", "HEAD")
        if tag and git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}") != head:
            raise ValueError("Checked-out commit is not the requested release tag.")
        result.update(commit=head, source_date_epoch=git("show", "-s", "--format=%ct", "HEAD"))
    return result


def normalize_sdist(path: Path, epoch: int) -> None:
    """Normalize tar/gzip timestamps and ownership so the same tag rebuilds identically."""
    buffer = io.BytesIO()
    with tarfile.open(path, "r:gz") as source:
        with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=epoch) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as target:
                for info in sorted(source.getmembers(), key=lambda member: member.name):
                    if info.issym() or info.islnk():
                        raise ValueError("Unexpected link in source distribution.")
                    info.mtime, info.uid, info.gid = epoch, 0, 0
                    info.uname = info.gname = ""
                    info.pax_headers = {k: v for k, v in info.pax_headers.items() if k not in {"mtime", "atime", "ctime"}}
                    data = source.extractfile(info) if info.isfile() else None
                    target.addfile(info, data)
    path.write_bytes(buffer.getvalue())


def validate_dist(directory: Path, root: Path = ROOT) -> list[Path]:
    version = version_from_source(root)
    wheels, sdists = sorted(directory.glob("*.whl")), sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution.")
    expected = {"Name": "rastermoves", "Version": version}
    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata_paths = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("Wheel metadata missing or ambiguous.")
        metadata = BytesParser().parsebytes(wheel.read(metadata_paths[0]))
        if any(metadata[k] != v for k, v in expected.items()):
            raise ValueError("Wheel name/version does not match the tagged source.")
        if "rastermoves/sweep.py" not in wheel.namelist() or "rastermoves/py.typed" not in wheel.namelist():
            raise ValueError("Wheel is missing required package data.")
        if len([n for n in wheel.namelist() if n.startswith("rastermoves/models/") and n.endswith(".json")]) != 8:
            raise ValueError("Expected all eight bundled model manifests.")
    with tarfile.open(sdists[0], "r:gz") as archive:
        metadata_paths = [n for n in archive.getnames() if n.count("/") == 1 and n.endswith("/PKG-INFO")]
        if len(metadata_paths) != 1:
            raise ValueError("Source distribution metadata missing or ambiguous.")
        metadata = BytesParser().parsebytes(archive.extractfile(metadata_paths[0]).read())
        if any(metadata[k] != v for k, v in expected.items()):
            raise ValueError("Source distribution name/version does not match the tagged source.")
        prefix = metadata_paths[0].rsplit("/", 1)[0]
        required = ("docs/RELEASING.md", "Dockerfile", "scripts/release_tools.py", ".github/workflows/release.yml")
        if any(f"{prefix}/{p}" not in archive.getnames() for p in required):
            raise ValueError("Source distribution is missing release tooling or documentation.")
    return wheels + sdists


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("metadata")
    check.add_argument("--tag", required=True)
    check.add_argument("--repository", required=True)
    check.add_argument("--github-output", type=Path)
    check.add_argument("--allow-untagged", action="store_true", help="TestPyPI rehearsal only; production always needs a tag.")
    dist = sub.add_parser("dist")
    dist.add_argument("directory", type=Path, default=Path("dist"), nargs="?")
    dist.add_argument("--normalize", action="store_true")
    dist.add_argument("--checksums", type=Path)
    args = parser.parse_args(argv)
    if args.command == "metadata":
        result = release_metadata(args.tag, args.repository, allow_untagged=args.allow_untagged)
        text = "".join(f"{k}={v}\n" for k, v in result.items())
        print(text, end="")
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as f:
                f.write(text)
    else:
        if args.normalize:
            for path in args.directory.glob("*.tar.gz"):
                normalize_sdist(path, int(os.environ["SOURCE_DATE_EPOCH"]))
        paths = validate_dist(args.directory)
        if args.checksums:
            args.checksums.parent.mkdir(parents=True, exist_ok=True)
            args.checksums.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in paths),
                                      encoding="utf-8")
        print("Validated:", ", ".join(p.name for p in paths))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"Release validation failed: {e}") from e
