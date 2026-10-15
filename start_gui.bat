@echo off
setlocal
cd /d "%~dp0"

echo [RasterMoves] Preparing GUI environment...

set "UV_EXE="
for /f "delims=" %%I in ('where uv 2^>nul') do if not defined UV_EXE set "UV_EXE=%%I"

if not defined UV_EXE (
    echo [RasterMoves] uv was not found. Installing uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression"
    if errorlevel 1 goto :failed

    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
    if not defined UV_EXE if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.cargo\bin\uv.exe"
    if not defined UV_EXE if exist "%LOCALAPPDATA%\Programs\uv\uv.exe" set "UV_EXE=%LOCALAPPDATA%\Programs\uv\uv.exe"
)

if not defined UV_EXE (
    for /f "delims=" %%I in ('where uv 2^>nul') do if not defined UV_EXE set "UV_EXE=%%I"
)

if not defined UV_EXE (
    echo [RasterMoves] ERROR: uv installation completed but uv.exe could not be located.
    goto :failed
)

echo [RasterMoves] Ensuring managed Python 3.12 is available...
"%UV_EXE%" python install 3.12
if errorlevel 1 goto :failed

if not exist ".venv\pyvenv.cfg" (
    echo [RasterMoves] Creating local .venv...
    "%UV_EXE%" venv --python 3.12 .venv
    if errorlevel 1 goto :failed
)

echo [RasterMoves] Installing/updating GUI, upscaling and refinement dependencies...
"%UV_EXE%" pip install --python .venv -e ".[all,refine,extra-arches,gui]"
if errorlevel 1 goto :failed

echo [RasterMoves] Starting GUI...
".venv\Scripts\rastermoves-gui.exe" %*
exit /b %ERRORLEVEL%

:failed
echo [RasterMoves] Startup failed.
pause
exit /b 1
