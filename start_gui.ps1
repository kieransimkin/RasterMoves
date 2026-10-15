param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $GuiArgs
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Uv {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $HOME ".local\bin\uv.exe"),
        (Join-Path $HOME ".cargo\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }
    return $null
}

try {
    Write-Host "[RasterMoves] Preparing GUI environment..."
    $uv = Find-Uv

    if (-not $uv) {
        Write-Host "[RasterMoves] uv was not found. Installing uv..."
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $uv = Find-Uv
    }

    if (-not $uv) {
        throw "uv installation completed but uv.exe could not be located."
    }

    Write-Host "[RasterMoves] Ensuring managed Python 3.12 is available..."
    & $uv python install 3.12
    if ($LASTEXITCODE -ne 0) { throw "uv could not install Python 3.12." }

    if (-not (Test-Path ".venv\pyvenv.cfg")) {
        Write-Host "[RasterMoves] Creating local .venv..."
        & $uv venv --python 3.12 .venv
        if ($LASTEXITCODE -ne 0) { throw "uv could not create .venv." }
    }

    Write-Host "[RasterMoves] Installing/updating GUI, upscaling and refinement dependencies..."
    & $uv pip install --python .venv -e ".[all,refine,extra-arches,gui]"
    if ($LASTEXITCODE -ne 0) { throw "uv could not install RasterMoves dependencies." }

    Write-Host "[RasterMoves] Starting GUI..."
    & ".\.venv\Scripts\rastermoves-gui.exe" @GuiArgs
    exit $LASTEXITCODE
}
catch {
    Write-Error "[RasterMoves] Startup failed: $($_.Exception.Message)"
    exit 1
}
