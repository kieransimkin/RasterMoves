#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

echo "[RasterMoves] Preparing GUI environment..."

if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
else
    echo "[RasterMoves] uv was not found. Installing uv..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "[RasterMoves] ERROR: installing uv requires curl or wget." >&2
        exit 1
    fi

    if [[ -x "$HOME/.local/bin/uv" ]]; then
        UV="$HOME/.local/bin/uv"
    elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
        UV="$HOME/.cargo/bin/uv"
    elif command -v uv >/dev/null 2>&1; then
        UV="$(command -v uv)"
    else
        echo "[RasterMoves] ERROR: uv installation completed but uv could not be located." >&2
        exit 1
    fi
fi

echo "[RasterMoves] Ensuring managed Python 3.12 is available..."
"$UV" python install 3.12

if [[ ! -f ".venv/pyvenv.cfg" ]]; then
    echo "[RasterMoves] Creating local .venv..."
    "$UV" venv --python 3.12 .venv
fi

echo "[RasterMoves] Installing/updating GUI, upscaling and refinement dependencies..."
"$UV" pip install --python .venv -e ".[all,refine,extra-arches,gui]"

echo "[RasterMoves] Starting GUI..."
if [[ -x ".venv/bin/rastermoves-gui" ]]; then
    exec ".venv/bin/rastermoves-gui" "$@"
fi

echo "[RasterMoves] ERROR: GUI entry point was not installed into .venv." >&2
exit 1
