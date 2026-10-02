# Changelog

## 0.2.0 — 2026-10-02

- Add `upscale --all-models -o DIRECTORY`, with `--sync-models` for the full
  OpenModelDB catalogue and `--dry-run` to inspect the selection before downloading.
- Process models sequentially with isolated failures, deterministic per-model
  filenames, provenance sidecars and an incremental `summary.json`.
- Add checksum-validated `--resume`, folder locking, explicit overwrite controls,
  interruption recording and nonzero exit codes for incomplete comparisons.
- Add tag-driven CI/CD for PyPI Trusted Publishing, GitHub Release assets/checksums,
  GHCR CPU containers, optional TestPyPI rehearsals and optional Docker Hub mirroring.
- Gate publication on Linux/Windows/macOS tests, optional-runtime tests, wheel/sdist
  validation and isolated-install smoke checks; smoke-test containers before pushing.
- Pin third-party Actions to verified commit SHAs and configure Dependabot updates.
- Add version/tag guards, deterministic source archives, release documentation,
  Dockerfile, and package/repository/author metadata.


## 0.1.1 — 2026-10-02

### RasterMoves rename

- Rename the UpscaleLab distribution, executable, and Python package to `rastermoves`.
- Identify RasterMoves as the image-upscaling component of the DanceFlow family.
- Rename Python plugin entry-point groups to `rastermoves.models` and
  `rastermoves.backends`; update imports, plugin documentation, and examples.
- Rename environment variables to `RASTERMOVES_CACHE`, `RASTERMOVES_PLUGIN_PATH`,
  `RASTERMOVES_OFFLINE`, and `RASTERMOVES_LIVE`.
- Use RasterMoves-specific default cache/config directories, temporary output prefixes,
  HTTP user agent, CLI help, diagnostics, and test workflow settings.
- Include the tool name and version in JSON output provenance reports.
- Add rename regression tests and a migration guide, including reuse of existing
  downloaded weights without requiring another download.

The eight bundled model definitions, model IDs, publisher checksums, source URLs,
licences, supported backends, and image-processing options are unchanged. This is a
local source/wheel release, not a publication to PyPI or GitHub. Legacy command/import
aliases are not installed; see [MIGRATION.md](docs/MIGRATION.md).

## 0.1.0 — 2026-10-02

Initial implementation under the working name UpscaleLab: per-model JSON plugins,
OpenModelDB catalogue import, automatic downloads and caching, Spandrel/PyTorch and
ONNX adapters, tiled inference, CLI and Python API, and regression tests.
