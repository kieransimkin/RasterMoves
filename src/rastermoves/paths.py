from pathlib import Path
import os
from platformdirs import user_config_path


def project_root() -> Path:
    """Locate a RasterMoves source checkout; packaged installs fall back to the working directory."""
    override = os.environ.get("RASTERMOVES_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "rastermoves").is_dir():
            return parent
    return Path.cwd().resolve()


def cache_dir(value=None) -> Path:
    # Keep large/partial model downloads with the project by default rather than in
    # an opaque per-user system cache. Explicit CLI/env overrides still win.
    path = Path(value or os.environ.get("RASTERMOVES_CACHE") or project_root() / "models")
    path = path.expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def plugin_dirs() -> list[Path]:
    paths = [Path(user_config_path("rastermoves")) / "models"]
    paths += [Path(p).expanduser() for p in os.environ.get("RASTERMOVES_PLUGIN_PATH", "").split(os.pathsep) if p]
    return paths
