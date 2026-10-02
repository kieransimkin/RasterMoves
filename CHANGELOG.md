# Changelog

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
