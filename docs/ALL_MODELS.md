# All-model image comparisons

## Run one image through all registered models

```bash
rastermoves upscale input.png --all-models -o comparison
```

`-o` is required and always means a **directory** in this mode. One image is accepted,
not a directory or recursive batch. All registered IDs are attempted in sorted order,
including 1x restoration models. One suitable resource/backend is chosen per model by
its existing plugin; the command does not run every mirror or every format of the same
checkpoint. `--model`, `--model-file`, `--native-scale`, and `--recursive` are incompatible.

The registry combines bundled definitions, previously imported OpenModelDB metadata,
custom manifest directories, and explicitly enabled Python entry-point plugins.

```bash
# Include your manifest plugins in the same run.
rastermoves --plugin-dir ./my-models upscale input.png --all-models -o comparisons

# Attempt every model in the current remote catalogue.
rastermoves upscale input.png --all-models --sync-models -o openmodeldb-comparison

# Alternatively, refresh metadata once, then run against that local snapshot.
rastermoves sync
rastermoves upscale input.png --all-models -o openmodeldb-comparison
```

No registry entry is silently treated as compatible. Missing runtimes, unsupported
architectures, missing URLs, unavailable weights, unsupported channels, invalid size
constraints and other load/inference errors are recorded for the affected model.
Later models are still attempted. Automatic resource fallback remains the same as in
single-model mode. There is no automatic package installation or remote-code execution.

## Plan downloads before running

```bash
rastermoves upscale input.png --all-models --sync-models --dry-run -o comparison
```

Dry-run emits JSON with IDs, licences, native scales, proposed output paths and declared
resource sizes. It **does not download weights, load models or create the results
folder**. `--sync-models` still fetches catalogue metadata; omit it for a fully local
plan. Listed sizes describe resources, not a prediction of total disk usage: mirrors,
backend choices, cached files and runtime allocations differ. A catalogue-wide run can
require substantial bandwidth, disk space and processing. Review licence metadata.

## Files and progress

```text
comparison/
  model-4x-realesr-general-x4v3.png
  model-4x-realesr-general-x4v3.png.json
  model-4x-UltraSharpV2.png
  model-4x-UltraSharpV2.png.json
  summary.json
```

The `model-` prefix keeps otherwise valid IDs such as `CON` safe as Windows filenames.
The existing output writer also uses `.lock` files; these contain no image results.

Successful outputs are written atomically. Provenance sidecars are always enabled for
all-model runs, regardless of `--report`. `summary.json` includes software version,
input SHA-256, processing settings, model licence/native scale/manifest hash, relative
output name, status, elapsed time, result/sidecar checksums and error details. It is
atomically refreshed before and after each attempt, so progress remains available if
one model fails. Progress and diagnostics go to stderr; successful output paths go to
stdout. A folder lock prevents concurrent sweeps from writing to the same folder.

Models are loaded sequentially, closed and garbage-collected before the next attempt.
This limits simultaneously loaded models, not the size of one model or the full output
array. Tiling reduces inference memory, but native-resolution outputs still occupy RAM.
A model whose whole-image requirements exceed memory may still fail.

Exit codes: **0** all successful/reused; **1** at least one failed or invalid run setup;
**2** argparse usage errors; **130** Ctrl+C. Native-process crashes, forced termination,
system shutdown and an OS OOM-kill cannot be caught by Python; the last saved summary
can still be resumed. Per-model Python errors do not require `--continue-on-error`.

## Output settings

```bash
# Same final width for visual comparison; native neural scales may differ.
rastermoves upscale cover.png --all-models -o width-3000 --width 3000

# Lower inference memory usage; explicit CPU selection.
rastermoves upscale input.png --all-models -o small-tiles \
  --tile 128 --overlap 24 --tile-pad 16 --device cpu

# Cached definitions/weights only; uncached models are reported as failed.
rastermoves upscale input.png --all-models -o offline-comparison --offline
```

`--format`, `--precision`, `--backend`, `--device`, `--alpha`, final-size options,
`--max-output-mp`, `--strict-checksums`, `--extra-arches` and `--force-tiling` retain their
single-model meanings. `--backend onnx` still selects all model IDs, so models without
an ONNX resource fail explicitly rather than disappearing from the summary. RGBA is
preserved with PNG/WebP/TIFF; transparent input with JPEG is rejected before loading.
The native AND final output pixel limit still applies.

## Resume and overwriting

```bash
rastermoves upscale input.png --all-models -o comparison --resume
```

Resume requires this folder's existing `summary.json`. Keep the **same input bytes,
RasterMoves version and processing settings**. Both output and sidecar checksums, plus
the model's manifest hash, must match before a successful result is reused. Failed,
interrupted, missing or changed results are rerun. A changed model manifest invalidates
only that model. Newly registered models can be appended on resume. Existing files not
tracked by the earlier run are not overwritten. A changed input/version/settings needs
a new folder or an explicit `--overwrite` run instead.

`--resume` may replace damaged or outdated outputs belonging to that saved run.
`--overwrite` starts a new summary and permits replacement of planned filenames, but
never replaces the input file. Neither option empties the folder or removes outputs
for models no longer selected. Failed replacements can leave an older image present;
**use the latest summary, not directory presence, to determine success**. Preserve
valuable results before choosing overwrite. Resume and overwrite are mutually exclusive.

Cache-directory and offline settings can change between resumptions. Environment/runtime
upgrades or changes inside third-party Python plugin code are not fully fingerprinted;
use a fresh folder for controlled benchmarks after such changes. A successful result's
provenance describes the runtime actually used. Hard termination during output writing
may leave a temporary file; a resumed run does not mistake that file for a verified result.
