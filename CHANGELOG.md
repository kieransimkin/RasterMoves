# Changelog

## 0.3.4 - 2026-10-08

- Add explicit agent capability, improvement, validation and upstream PR guidance to the README and contributor instructions.

## 0.3.3 - 2026-10-08

- Apply the reviewed capability description and author website to OCI container labels as well as language packages. Preserve previously published versions and original logo bytes.


## 0.3.2 - 2026-10-08

- Add an original tool-specific vector logo and PNG companion in the shared DanceFlow visual style.
- Clarify package descriptions from reviewed documentation and KeywordMoves literal-source evidence, without claims of measured search demand.
- Link package descriptions and READMEs to Kieran Simkin’s website and retain branding files in installable packages.


## 0.3.1 - 2026-10-08

- Include the reviewed use-and-improve workflow, evaluation checklist and improvement-loop diagram.
- Document mixed-runtime comparison recovery and clarify the current optional model routes.
- No image-processing algorithm or default-model changes.

## 0.3.0 — 2026-10-02

- Add optional same-size SD1.5/SDXL ControlNet Tile img2img workflows, with pinned
  Hugging Face component revisions, safe checkpoint loading and explicit offline prefetch.
- Add `refiners`, `download-refiner`, standalone `refine`, and `upscale --refiner`.
- Keep original upscaler plugins/defaults unchanged; the heavyweight `refine` extra
  is separate from base, `all` and the published CPU container.
- Use frozen-input spatial tiles, independent diffusion tile controls and coordinate-derived
  seeds; report actual denoising steps instead of implying configured steps all execute.
- Preserve lossless baselines, alpha and protected pixels; add outward mask feathering,
  explicit blending and optional global RGB mean matching.
- Add hash-verified resume, per-workflow journals, failure/interruption baseline retention,
  output collision guards and explicit one-refiner all-model comparisons.
- Extend stage traces with VAE encode/decode, denoising and compositing; record CUDA
  allocator peaks when running the native CUDA refinement backend.
- Add local refiner entry points, offline tests, optional random-weight Diffusers runtime
  CI, opt-in pretrained smoke tests, and installed-artifact workflow smoke checks.
- This is the first ControlNet Tile implementation, not MultiDiffusion/shared-latent
  inference or a SeedVR2/VOSR/SUPIR restoration integration.

## 0.2.1 — 2026-10-02

- Add opt-in `--trace`, `--trace-file` and `--trace-interval` options for single-image,
  batch, all-model and dry-run commands.
- Stream Chrome JSON timelines with nested stage durations, process CPU seconds /
  utilization, RSS/VMS and thread counters, plus system available-memory context.
- Include per-model metrics in comparison summaries; preserve failure/interruption
  traces, mark resumed results as reused rather than inventing new inference timings.
- Distinguish download/cache verification, backend load, inference attempts (including
  OOM retry markers), postprocessing, writes and cleanup.
- Add a lazy optional `trace` dependency extra (`psutil`), also included in `all`/`dev`.
- Never overwrite existing traces; choose numbered defaults on subsequent runs.
- Add tracing documentation, regression tests and installed-artifact tracing smoke tests.
- No model/plugin interface, checkpoint source, architecture or publication-credential changes.

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
