# Generative refinement — RasterMoves 0.3.0

## What this release implements

Two native Diffusers image-to-image workflows, `sd15-tile` and `sdxl-tile`, use a
matching ControlNet Tile to add plausible detail **without changing image dimensions**.
They are separate from the existing scale-changing upscaler model/backend interfaces.
The base image supplies both img2img initial latents and ControlNet image conditioning.

This is optional generative enhancement, not a faithful reconstruction guarantee.
Faces, geometry, small objects and lettering can change. Always compare the saved
non-generative baseline. Protection masks provide exact compositing protection, rather
than assuming diffusion will reproduce important text or identities.

This release does **not** implement SeedVR2, VOSR, PiSA-SR, FiDeSR, SUPIR, HYPIR, FLUX,
a ComfyUI bridge, the SDXL refiner checkpoint, a shared-latent MultiDiffusion solver,
or a separate seam-fixing diffusion pass. Those remain separate future integrations,
not aliases that secretly execute this pipeline. No pretrained visual-quality claims
are made by the offline tests supplied with this patch.

## Install

Use Python 3.10 or newer. Install the PyTorch build appropriate to your accelerator,
then, from the source checkout:

```bash
python -m pip install -e ".[refine,trace]"
rastermoves doctor
rastermoves refiners
```

Existing upscaler runtimes are unchanged. For a fresh installation needing both all
existing upscalers and diffusion, explicitly select `.[all,refine]`. The `refine`
extra is not included in `all`, `dev`, base dependencies or the default CPU container.
A published wheel supports the equivalent `rastermoves[refine,trace]` extra once the
maintainer publishes this version; this patch does not publish it.

Diffusers is pinned to 0.35.2 for a known API contract. Transformers is constrained
below 5, and Accelerate below 2. Updating those boundaries requires running the real
runtime tests and a full checkpoint smoke test, not merely editing the dependency text.
The tested entry-point semantics were checked against upstream 0.35.2 source; local
pretrained execution is not represented as verified when the runtime was unavailable.

## Commands

```bash
# Refine an already upscaled file. No additional enlargement occurs.
rastermoves refine upscaled.png -o refined.png --refiner sd15-tile --seed 123 --trace

# Select the larger SDXL workflow.
rastermoves refine upscaled.png -o refined-xl.png --refiner sdxl-tile --seed 123

# Existing upscaler, final resizing, then same-size refinement.
rastermoves upscale input.png -o cover.png -m 4x-realesr-general-x4v3 \
  --width 3000 --refiner sd15-tile --refine-strength 0.22 --seed 123 --trace

# Process a directory, preserving its relative structure.
rastermoves refine originals -o refined --recursive --continue-on-error

# Inspect configuration/outputs without weight downloads or inference.
rastermoves refine upscaled.png -o refined.png --refiner sdxl-tile --dry-run

# Apply exactly one selected refiner after each registered upscaler.
rastermoves upscale input.png --all-models -o comparison --refiner sd15-tile --trace
```

Add `--sync-models` only to explicitly refresh a full all-model catalogue. Refiners are
not imported from OpenModelDB scale-model manifests. `--all-models` does not silently
try every refiner or every seed. Failed attempts are recorded and remaining models run.
An entire catalogue sweep with diffusion may be extremely expensive. Start with one
input/model or use dry-run first. Use `--width` to give differently scaled upscalers a
common target when comparing them with a single fixed-size protection mask.

Global flags such as `--cache-dir` and `--external-plugins` precede the subcommand.
On `upscale`, supplying refinement-specific options without `--refiner` is an error,
not a silently ignored request. `refine` selects `sd15-tile` by default because the
command itself explicitly requests refinement.

### Controls and defaults

| Flag | Default | Meaning |
|---|---|---|
| `--refiner` | `sd15-tile` on `refine`; none on `upscale` | Pinned workflow |
| `--refine-strength` | 0.22 | Img2img noise strength in [0,1] |
| `--refine-steps` | 40 | Scheduler steps before strength truncation |
| `--refine-guidance` | 5 | Classifier-free guidance scale |
| `--refine-control-scale` | 1 | ControlNet conditioning weight |
| `--seed` | 0 | Integer root seed; recorded per-tile seeds derive from coordinates |
| `--refine-prompt` / `--prompt` | empty | Optional content description |
| `--refine-negative-prompt` / `--negative-prompt` | empty | Optional negative prompt |
| `--refine-tile` | 512 for SD1.5; 1024 for SDXL | Independent diffusion tile in working-image pixels |
| `--refine-overlap` | 64 | Overlap between diffusion crops, less than tile size |
| `--refine-blend` | 1 | Generated/baseline RGB blend in [0,1] |
| `--refine-color-match` | off | Match global RGB means before final compositing |
| `--protect-mask` | none | White protects; black allows changes |
| `--refine-mask-feather` | 0 | Outward mask feather radius in working pixels |
| `--refine-device` | auto | CUDA, else MPS, else CPU; explicit cuda:N supported |
| `--refine-precision` | auto | FP16 on CUDA, FP32 otherwise; CPU FP16 rejected |
| `--refine-offload` | none | none/model/sequential; offload requires CUDA |
| `--refine-scheduler` | ddim | ddim or euler |
| `--refine-no-vae-tiling` | absent | Disable otherwise-enabled VAE tiling |
| `--max-output-mp` | 64 | Reject an image over the pixel limit |

These defaults are conservative starting parameters, not visually benchmarked quality
presets. Positive strength must give `int(strength * steps) >= 1`. For example, a
40-step schedule at strength 0.22 executes a shortened schedule, not 40 denoising
iterations. The report counts real callback invocations separately for each tile.

Strength zero, blend zero or a fully protected image skips runtime/model loading and
returns the baseline. This is useful for testing I/O, masks and trace plumbing without
downloading models. It is **not** a demonstration of successful diffusion inference.

### Tiling and memory

Each crop comes from the same immutable baseline. A stable SHA-256-derived seed based
on `(root_seed,x,y)` avoids dependence on skipped tiles or Python's randomised hash.
Crops are edge-padded to a multiple of eight and at least 64 pixels, then cropped back.
Generated RGB tiles are accumulated using positive feather weights. The final image
is composited with the baseline, blend factor and protection mask. It is never resized
inside refinement. In a chained workflow the existing upscaler owns final resizing.

This is **spatial overlap/blend img2img**, not shared-latent MultiDiffusion. Independent
crops may disagree in texture, lighting or semantics; overlap reduces but cannot
eliminate seams. ControlNet Tile is a conditioning model, not an automatic diffusion
memory scheduler. VAE tiling handles encode/decode memory separately. There is no
hidden pre-downscale, automatic OOM retile or seed change. An OOM reports an error and
retains the baseline; change settings explicitly before rerunning with `--overwrite`
or a new output path because changed settings intentionally invalidate `--resume`.

```bash
rastermoves refine upscaled.png -o refined.png --refiner sdxl-tile \
  --refine-device cuda --refine-tile 512 --refine-overlap 64 --refine-offload model
```

Smaller-than-training tiles can reduce useful context/quality. Model offload trades
transfer overhead for resident GPU memory; sequential offload is more aggressive.
No universal minimum VRAM or speed is promised. CPU is likely unsuitable for large
comparisons. Full-image float32 accumulation buffers remain in host RAM even when
using small tiles. Feed-forward and diffusion weights are not intentionally kept on
the GPU together: the upscaler closes before refinement, and a previous refinement
session closes before a new upscaler in a batch. In standalone batches a refiner is
reused to avoid repeated loading.

## Output contract, masks and alpha

For `-o enhanced.png`, the workflow writes:

```text
enhanced.baseline.png       lossless, pre-refinement working image
enhanced.baseline.png.json  source/upscaler provenance
enhanced.png                final refined image
enhanced.png.json           mandatory workflow journal and detailed provenance
enhanced.png.trace.json     optional --trace output (numbered if already present)
```

Outputs must be PNG, lossless WebP or TIFF. JPEG is rejected to avoid claiming exact
pixel protection after a lossy re-encode. Baselines are always PNG. Inputs follow the
existing 8-bit sRGB preparation: EXIF orientation is applied; colour profiles are
converted; unsupported HDR/high-bit-depth and animated inputs are rejected. “Exact
protected pixels” means the normalised/upscaled baseline pixels, not unchanged file
bytes or original CMYK values. The original input file is never overwritten.

Protection masks must be opaque 8-bit grayscale/RGB and match the **working/final
baseline size**, not necessarily the original low-resolution source. No implicit mask
resizing occurs. A dimension mismatch is an error before diffusion model loading.
White (255) restores baseline RGB exactly; intermediate values blend continuously;
black (0) permits refinement. Feathering expands protection outward while keeping
original white pixels white. Colour matching and global blending happen before exact
white-pixel restoration. All-protected tiles skip inference without affecting seeds
elsewhere. The baseline alpha channel is copied unchanged. Fully transparent pixels
are protected automatically; diffusion never generates a replacement alpha channel.

```bash
rastermoves refine cover-upscaled.png -o cover-refined.png \
  --protect-mask lettering-mask.png --refine-mask-feather 4 --refine-strength 0.2
```

Use a lossless PNG result for easiest pixel-by-pixel comparisons. No separate raw,
unblended diffusion image is exported in this version; rerun under another name with
blend=1 and colour matching off to compare that setting. Do not describe the saved
baseline as an unblended generative result: it is the non-generative input to refinement.

## Pinned downloads, offline use and model terms

Use `rastermoves refiners --json` for the exact machine-readable component descriptors.
The built-ins pin these revisions:

| Workflow | Role and repository | Revision |
|---|---|---|
| sd15-tile | base: stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 |
| sd15-tile | controlnet: lllyasviel/control_v11f1e_sd15_tile | 3f877705c37010b7221c3d10743307d6b5b6efac |
| sdxl-tile | base: stabilityai/stable-diffusion-xl-base-1.0 | 462165984030d82259a11f4367a4eed129e94a7b |
| sdxl-tile | controlnet: xinsir/controlnet-tile-sdxl-1.0 | 1ae8d9529efe58f7362a987363ff86a7904dc84f |

Each workflow contains its own VAE, text encoder(s), tokenizer(s), scheduler and
ControlNet, not just one checkpoint. Downloads go under RasterMoves' existing cache
root in `diffusion-hub/`, using Hugging Face's file cache and lock handling. Only an
allowlist of required data/weight files is downloaded. First validate the base class,
UNet and ControlNet configuration, then download weights. SD1.5 and SDXL components
cannot be mixed. All subsequent pipeline loading is local-only and uses explicit
Diffusers pipeline classes; repository Python and custom loader classes are not used.

All base and SDXL ControlNet weights use safetensors. The original SD1.5 Tile repository
publishes `.bin`; it is pinned to publisher SHA-256
`eb05b4c3665bd76dad70a90652014a9b3aab391abd8a5bb484e860330f9492fb`
and loaded with `torch.load(weights_only=True)` then checked for named tensors only.
There is no unrestricted pickle fallback. Cached SHA-256/LFS and Git-blob content
addresses are verified when available; `--strict-checksums` requires every file to
have a verifiable content address. Systems with copied rather than symlinked HF cache
files may lack that address information for some files, causing strict mode to fail
rather than pretending those hashes have been independently verified. Reports always
record the computed file hashes. Pinned source/integrity checks reduce risk but are not
a sandbox for native parsers or locally installed third-party packages.

```bash
rastermoves --cache-dir ./model-cache download-refiner sd15-tile
rastermoves --cache-dir ./model-cache download-refiner sdxl-tile
rastermoves --cache-dir ./model-cache refine upscaled.png -o refined.png --offline
```

`HF_TOKEN` is handled by huggingface_hub; do not put tokens in model IDs or command-line
prompts. An upstream gated/private model may require account access/terms acceptance.
`RASTERMOVES_OFFLINE` remains honoured through the existing downloader. Offline mode
requires all exact pinned files. Full-precision safetensors are downloaded and may be
cast for FP16 inference; GPU precision is not a download-size setting. No automatic
model revision upgrades or repository-code installs occur.

Base SD1.5 is CreativeML OpenRAIL-M, base SDXL CreativeML OpenRAIL++-M, the original
SD1.5 Tile model is labelled OpenRAIL, and the selected Xinsir model is Apache-2.0.
These are component metadata, not a blanket commercial-use/legal clearance for the
whole workflow. Read the actual model cards and licence texts; RasterMoves' code
licence does not replace them. Model-card links are included in `refiners` JSON and
[SOURCES.md](SOURCES.md). The SD1.5 base safety checker remains enabled. A flagged tile
fails the refinement workflow rather than saving a black tile as a successful result.

## Provenance, resume and failure handling

The output JSON is mandatory, even without `--report`. It fingerprints source bytes,
mask bytes, software version, workflow/component pins, prompts, root seed, scheduler,
all refinement settings, requested runtime settings and installed dependency versions.
Chained workflows also include the selected upscaler manifest, local weight hash when
applicable, and upscaler processing settings. Component inventories include actual
file hashes/sizes. Refiner reports show resolved device/precision and per-tile seeds,
geometry, configured steps and actual executed steps.

Fixed seeds improve repeatability on a consistent software/hardware stack; bitwise
reproducibility across GPUs, devices or dependency versions is not promised. The resume
fingerprint includes requested device selection, not a fingerprint of all installed
hardware or every external plugin's source code. Use an explicit device and a fresh
output folder after changing hardware or third-party plugin implementations.

```bash
rastermoves refine upscaled.png -o refined.png --seed 123 --resume
```

Resume checks the journal and hashes of the baseline, baseline report and final result.
A completed verified result is reused without fake new inference timings. After a
failed/interrupted refinement, an intact baseline is reused and refinement is retried.
It resumes at the stage/image level, **not** from partially generated tiles or denoising
latents. A changed input, mask, setting, model pin, package/runtime version requires
`--overwrite` or a new path; no implicit stale result reuse occurs. Changing only tracing,
cache location or offline permission does not invalidate an otherwise matching run.
In all-model mode the overall summary also checks artifact hashes; an explicitly
tracked corrupted result is rebuilt. Upgrading 0.2.1 to 0.3.0 changes comparison identity.

Existing unrelated outputs, baselines, reports, directories or symlinks are not silently
overwritten. The workflow has a single-writer lock, image writes are atomic, and Python
exceptions/Ctrl+C update its journal. An OS kill, disk error or hardware failure can
leave an incomplete journal or temporary files; it is not a transactional filesystem.
Traces are independently named and never overwritten by image `--overwrite`.
The input and protection-mask files are rechecked before final publication to detect
edits during a long run. Reports contain local paths and prompts: review them before
sharing; they do not intentionally store HF credentials.

## Resource measurements

`--trace` / `--trace-file` / `--trace-interval` are available unchanged. Added spans
include workflow baseline preparation, component downloads/verification, refinement
loading, image preparation, tiles, VAE encode, denoising, VAE decode, compositing,
writing and cleanup. Process CPU/RSS/VMS remain sampled by the existing tracer.

The native refinement backend synchronises GPU work at stage boundaries. It does not
synchronise every denoising callback. Per-tile reports include wall/process-CPU seconds
and VAE/denoising durations. CUDA reports add peak PyTorch allocated and reserved bytes,
reset per tile. These are process allocator measurements including model weights,
not NVML device-wide utilisation, total VRAM, or per-layer allocation attribution.
They do not instrument the existing feed-forward CUDA backend or external workers.
MPS/CPU do not receive invented CUDA counters. Measurements carry instrumentation
overhead. A resumed result marks reuse; its prior timings remain historical.

## Python API and extension layer

```python
from rastermoves.refinement import Refiner, RefineOptions

options = RefineOptions(strength=0.22, steps=40, seed=123)
with Refiner("sd15-tile", options=options, device="auto") as refiner:
    refiner.refine_file("upscaled.png", "enhanced.png", protect_mask="protect.png")
    print(refiner.last_report)
```

For chaining, use `run_workflow(source, output, refiner, upscaler=upscaler,
image_options={...})`; it closes the upscaler before loading diffusion. A reusable
`Upscaler` reloads lazily if used afterwards. See `examples/refine_api.py`.
Use `refine_image(PIL_image, protect_mask=PIL_mask)` for an in-memory result; in this
case callers own file writes, baseline retention and resume. Sessions are not thread-safe.

Installed Python extensions register in `rastermoves.refiners` and must be explicitly
enabled with `--external-plugins`. Their factory returns `(RefinerSpec, backend_factory)`;
see [PLUGINS.md](PLUGINS.md). This first schema models SD1.5/SDXL same-size refiners,
not arbitrary original-input restoration workflows. Existing `rastermoves.models`
and `rastermoves.backends` interfaces remain unchanged.

## Verification

Run `python -m pytest -q` for offline regression tests. A separate CI job installs the
`refine` extra and executes tiny randomly initialised real SD1.5/SDXL Diffusers pipelines
on CPU with no checkpoint downloads. It sets `RASTERMOVES_REQUIRE_DIFFUSERS=1` so a
missing runtime cannot silently pass by skipping. These runtime tests still do not
validate the full published checkpoints or image quality.

Optional full pretrained smoke tests download multi-GB models only when explicitly
requested:

```bash
# POSIX shell; select one family before downloading both.
RASTERMOVES_REFINE_LIVE=1 RASTERMOVES_REFINE_DEVICE=cuda \
  python -m pytest -q tests/test_refinement_live.py -k sd15
```

In PowerShell, set the two environment variables with `$env:NAME="value"`, then run
pytest normally. See [TESTING.md](TESTING.md) for what was and was not actually executed
for this patch. None of the synthetic/no-op tests is a visual quality benchmark.
