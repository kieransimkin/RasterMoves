from pathlib import Path
import os
from platformdirs import user_cache_path, user_config_path


def cache_dir(value=None) -> Path:
    path = Path(value or os.environ.get("RASTERMOVES_CACHE") or user_cache_path("rastermoves"))
    path = path.expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def plugin_dirs() -> list[Path]:
    paths = [Path(user_config_path("rastermoves")) / "models"]
    paths += [Path(p).expanduser() for p in os.environ.get("RASTERMOVES_PLUGIN_PATH", "").split(os.pathsep) if p]
    return paths
