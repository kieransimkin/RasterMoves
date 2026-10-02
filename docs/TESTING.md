# RasterMoves 0.2.0 validation

Base repository: `kieransimkin/RasterMoves`, commit
`dd1d61afc582d23c2108b72bea1e9686e555b9f3`, checked on 2026-10-02.
The local baseline's complete Git tree matches upstream exactly:
`fc023effd330fa26eb87eb1169e29cb55c7b1083`.

## Local results

- **180 tests passed, 2 optional live-model tests skipped**, Python 3.13.5 on Linux.
- All-model CLI → sweep → actual image pipeline/tiler/output-writer integration tested
  across all eight starter IDs using a deterministic substitute model (not pretrained
  neural-network quality testing). RGBA and native-scale output dimensions verified.
- Failure isolation, cleanup, Ctrl+C handling, incremental summaries, resume checksum
  verification, corrupt/missing results, changed input/settings/model manifests,
  new models, output collisions, folder locking and registry-snapshot reuse tested.
- Tag/version validation, prerelease classification, optional untagged TestPyPI
  rehearsals, lowercase container names, exact-tag commit checks, distributable
  metadata/data validation and reproducible source-archive normalization tested.
- Workflow YAML parsed; job dependencies, publishing gates, least-privilege settings,
  exact Action SHA pins and shell-script syntax tested. This is not a GitHub-hosted
  workflow execution or an actionlint run.
- Wheel and source distribution built locally using the installed setuptools backend;
  metadata and bundled model/typing/release files checked. Their package installations
  and CLI smoke checks run outside the source checkout, reusing preinstalled third-party
  dependencies because this environment cannot fetch new packages. CI uses fresh venvs
  and performs ordinary dependency installation plus strict Twine checks.
- Patch application checked against a clean worktree of the exact upstream commit;
  tests also run from that patched worktree.

## Not executed here

No GitHub-hosted workflow, real PyPI/TestPyPI upload, GitHub release creation,
Docker/GHCR/Docker Hub build/push or end-to-end pretrained-model upscale was performed.
Docker, actionlint, the build frontend, Twine, Spandrel and ONNX Runtime are not available
in this execution environment; the installed setuptools backend is used for local
package builds. Tests that exercise backend interfaces use substitutes as described
above and in the original test notes. PyTorch synthetic-checkpoint tests do run locally.

The workflows include fresh installation, strict Twine validation, optional-runtime
checks and a network-disabled container smoke test before pushing. The two live-model
tests require explicit opt-in (`RASTERMOVES_LIVE=1`) and real downloads/runtimes. A
catalogue entry is not a promise that its checkpoint is supported by installed runtimes.

Use **Re-run failed jobs** after fixing publisher configuration; do not treat a skipped
optional publication job as a successful upload. The patch and release tooling do not
reserve package names or change account/environment settings.

---

## Historical 0.1.1 validation notes

# RasterMoves validation record — 0.1.1

Date: 2026-10-02. These are observed results from the rename build, not claims about
all operating systems, devices, download hosts, or pretrained models.

## Regression suite

The original 0.1.0 archive was extracted and tested before editing:
**94 passed, 2 skipped**. After renaming and adding 25 name/migration checks:

```text
...................................................ss................... [ 59%]
.................................................                        [100%]
119 passed, 2 skipped in 2.56s
```

Command: `python -m pytest -q`, run inside the source project. The additional tests
check public imports, lightweight package import, CLI version/help, diagnostic output,
HTTP user agent, default cache/config names, new environment variables, entry-point
groups, bundled resources, cached-weight reuse, offline errors, and provenance branding.

All eight model-manifest files were also compared byte-for-byte against the original
archive. They are unchanged, including source URLs, model IDs, publisher checksums,
and recorded licences. Model source release numbers were not changed to the tool's
new version number.

## What the inherited suite exercises

The local tests exercise model manifest loading and overrides, explicit Python plugin
opt-in and backend dispatch, catalogue parsing and archive fallback with local
fixtures, download/cache integrity with stubbed transports, Hugging Face client
arguments, mirror failures, size limits, HTML/LFS rejection, and offline behavior.

They also exercise tile geometry, edge coverage, overlapping blending, context halos,
smaller-tile OOM retries, invalid output rejection, transparency, orientation, final
resizing, atomic output writing, sidecars, non-overwrite behavior, CLI model listing,
and batch path handling.

**Actual installed PyTorch/safetensors operations:** tensor checkpoint save/restricted
load, rejection of an executable-pickle fixture without running it, a synthetic Conv2d
network compared between tiled and whole-image execution, tensor conversion/inference
through the Spandrel adapter with a stub architecture detector, and safetensors save/load.

**Stub runtime contracts:** ONNX shape/layout/precision/provider checks and padding/
cropping, plus Spandrel descriptor integration and wrapper normalization. These are
regression checks, not pretrained-model validation.

## Wheel installation checks

The wheel was built locally with `setuptools.build_meta.build_wheel`, without fetching
build dependencies. It was installed with `pip --no-index --no-deps --target` into a
new directory, using the already installed core dependencies. The smoke checks ran
outside the source checkout and asserted the imported file came from that installed
wheel, not from the source tree. This was not a fresh dependency-resolution test.

Passed checks:

- Distribution metadata reports `rastermoves` version `0.1.1`.
- Both the actual generated `rastermoves` executable and `python -m rastermoves`
  report `RasterMoves 0.1.1`.
- The wheel provides no legacy executable or legacy Python import package.
- The eight packaged JSON models and `py.typed` marker are present.
- Model listing, offline model inspection, and `doctor` run successfully.
- A synthetic repeat-pixel backend performs a tiled RGBA file upscale from 13x9 to
  26x18 and writes a JSON sidecar identifying RasterMoves 0.1.1.

The synthetic smoke image is not presented as a neural-upscaling quality example.
The complete source ZIP and setuptools source distribution include tests,
documentation, model manifests, and the migration guide.

## Not executed in this rename validation

The two opt-in `live` tests were skipped. They download and run RealESR AnimeVideo v3
through Spandrel and SPANkendata through ONNX Runtime. Those two runtimes are absent
from this environment; no model weights were downloaded and no runtime installation
was attempted during the rename. No GPU or Apple MPS device is available here.

No end-to-end pretrained upscale, live catalogue sync, Hugging Face/GitHub/Drive
weight download, provider authentication/quota test, Windows/macOS integration, or
visual-quality comparison is claimed. The included GitHub Actions workflow was
updated, but was not executed on GitHub. Nothing was uploaded to PyPI or a repository.

## Reproduce

```bash
python -m pip install -e ".[torch,onnx,dev]"
python -m pytest -q
# Linux / macOS: explicitly enable the two download-and-inference tests.
RASTERMOVES_LIVE=1 python -m pytest -q tests/test_live.py
```

On Windows PowerShell, set `$env:RASTERMOVES_LIVE = "1"` before the final pytest
command. Each live test uses a temporary cache and checks the recorded publisher
checksum before inference.

## Observed environment

```json
{
  "rastermoves": "0.1.1",
  "python": "3.13.5",
  "packages": {
    "torch": "2.10.0+cpu",
    "torchvision": "0.25.0+cpu",
    "spandrel": null,
    "safetensors": "0.7.0",
    "onnxruntime": null,
    "onnxruntime-gpu": null,
    "gdown": null,
    "huggingface-hub": "1.16.1"
  },
  "torch_devices": {
    "cuda": false,
    "mps": false
  },
  "onnx_error": "No module named 'onnxruntime'"
}
```
