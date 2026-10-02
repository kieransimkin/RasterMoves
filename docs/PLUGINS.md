# Plugin development

## Two levels of modularity

A **model plugin** describes one set of weights: name, architecture, native scale,
channels, licence, source URLs, and verification hashes. A **backend plugin** implements
loading and inference for a weight format or runtime. There is no requirement to copy
an inference implementation for every model that shares an architecture.

Every bundled or imported JSON manifest creates its own `ModelPlugin` object. The
registry resolves plugins by case-insensitive ID. Different spellings that differ only
by case are rejected. Explicit custom manifests may override a matching ID; prefer a
new ID when the weights or behavior differ.

Load precedence is bundled manifests, cached catalogue, individually imported models,
configured user directories, then explicit `--plugin-dir` directories. Model IDs must
be safe ASCII identifiers, not paths.

## Add a model using JSON

Copy `examples/models/example-ultrasharp-v2.json` into a directory you control. It is
a complete working-format manifest referencing the same author-supplied model as the
bundled UltraSharpV2 entry, under a different ID. It is not a new trained model.

```bash
rastermoves --plugin-dir ./examples/models models
rastermoves --plugin-dir ./examples/models info example-ultrasharp-v2
rastermoves --plugin-dir ./examples/models upscale input.png -o output.png -m example-ultrasharp-v2
```

The manifest schema is:

```json
{
  "id": "my-model-id",
  "name": "My model display name",
  "scale": 4,
  "architecture": "dat",
  "license": "LICENSE-IDENTIFIER-FROM-AUTHOR",
  "author": "Author name",
  "description": "Purpose and relevant requirements",
  "tags": ["photo"],
  "input_channels": 3,
  "output_channels": 3,
  "resources": [],
  "options": {}
}
```

This schema sketch deliberately has no download resources. For each real resource,
include `format`, `backend` (`spandrel` or `onnx`), `urls` (HTTPS URLs), and preferably
`sha256` and `size` (exact byte count). The example file shows a complete verified-source
entry. The runtime uses the source SHA-256 for integrity, not a similar filename.

Hugging Face `blob` and `resolve` URLs are both recognized. For reproducible releases,
use a full commit revision in the URL and an author-verified SHA-256. URL-encode any
slash inside a revision name. Do not put credentials in URLs. `HF_TOKEN` is the supported
authentication mechanism for Hub downloads.

`RASTERMOVES_PLUGIN_PATH` accepts platform-separated directories (`:` on POSIX, `;` on
Windows). User config defaults use `platformdirs.user_config_path("rastermoves")/models`.
Custom metadata loading alone never executes Python.

### ONNX-specific contracts

The built-in ONNX backend supports one normalized floating-point image input/output,
with rank four, batch one, and one or three channels. Add `options` for deviations
from the default RGB NCHW image contract:

```json
{
  "layout": "NHWC",
  "channel_order": "RGB",
  "multiple_of": 8,
  "minimum_size": 16,
  "square": false,
  "tiling": "supported"
}
```

These are example values, not universal requirements. Use the model export's actual
contract. `channel_order` may be RGB or BGR; layout NCHW or NHWC. Tiling can be
`supported`, `discouraged`, or `internal`. A static spatial shape is discovered from
the export and padded to; the combined core tile and halo must fit that shape.
Precision must match the export: FP16 models need `--precision fp16`. Multi-file
external-data ONNX, integer/quantized image I/O, different numeric normalization,
multi-output/multi-input networks, and facial alignment need a custom backend.

## Python model plugins

Use Python only when metadata alone cannot describe the model. Installed Python
extensions are opt-in, because entry points execute local code:

```toml
[project.entry-points."rastermoves.models"]
my_plugin = "my_package.plugin:create_model"
```

The factory returns an instance of `rastermoves.plugins.ModelPlugin` or a subclass:

```python
from rastermoves.plugins import ModelPlugin
from rastermoves.specs import ModelSpec

# Import your own validated specification data here.
def create_model():
    return ModelPlugin(ModelSpec.from_dict(MY_SPEC))
```

`MY_SPEC` in the sketch must be supplied by your package. A subclass can override
`load(downloader, *, backend, device, precision, extra_arches, external_backends)`.
It must return `(loaded_model, local_weight_path, selected_resource)`. Custom loaders
are responsible for model-specific validation and must preserve safe download and
checkpoint-handling behavior. The loaded object implements the backend protocol below.

Enable installed extensions explicitly:

```bash
rastermoves --external-plugins models
rastermoves --external-plugins upscale input.png -o output.png -m my-plugin-id
```

External plugins may not shadow an existing model ID or built-in backend name.
Third-party Python packages must be installed by the user; discovery never installs
or runs a downloaded repository's code automatically.

## Backend extensions

Register a constructor in `rastermoves.backends`:

```toml
[project.entry-points."rastermoves.backends"]
myruntime = "my_package.backend:MyBackend"
```

The constructor receives `(path, spec, *, device, precision, extra_arches, ...)`.
Accept extra keyword options for forward compatibility. A model resource selects it
with `"backend": "myruntime"`. Pass `--external-plugins` on the CLI, or
`external_plugins=True` to `Upscaler`.

The loaded-model protocol is defined in `src/rastermoves/backends/base.py`:

- `scale`: positive integer; `input_channels` and `output_channels`: one or three.
- `device`, `precision`: strings for the provenance report; `tiling`: guidance string.
- `predict(image)`: receive contiguous HWC float32 in [0, 1], return HWC floating point
  at the declared native scale. Output must be finite and have the declared channels.
- `clear_cache()`: release temporary device allocations after a failed attempt.
- `close()`: release persistent model resources.

Raise `BackendOOM` only for actual inference-memory exhaustion. Raise `UpscaleError`
for other actionable failures. Do not disguise shape or model incompatibility as OOM;
that would cause pointless smaller-tile retries. Complete prediction buffers are
expected, not encoded image bytes.

The image pipeline owns tiling, alpha handling, final resizing, output writing, and
report generation. Put model-specific preprocessing in your backend or its wrapper.
Any transform must still respect the declared geometry and normalized image contract.

## Extending Spandrel architectures

Spandrel detects architectures from state dictionaries. The adapter performs restricted
loading first, unwraps common checkpoint containers/prefixes, then calls the public
`ModelLoader.load_from_state_dict` method. It never relies on Spandrel's private
checkpoint loader or enables unrestricted pickle loading.

Additional separately licensed architectures can be enabled with the `extra-arches`
installation extra plus `--extra-arches`. Review their licences first. Installing that
extra is not a promise that every OpenModelDB model becomes supported.


## Same-size refinement plugins (0.3.0)

The optional diffusion layer is separate from scale-changing model plugins. Do not
register a multi-component workflow as an ordinary scale=4 checkpoint. The built-in
`RefinerSpec` records a base pipeline and ControlNet with independent pinned commits
and licences; its scope is SD1.5/SDXL same-size img2img. Other original-input restoration
architectures need a future workflow contract rather than pretending to meet this one.

A locally installed extension can expose:

```toml
[project.entry-points."rastermoves.refiners"]
my-tile-refiner = "my_package.refiner:create_refiner"
```

The factory takes no arguments and returns `(spec, backend_factory)` where `spec` is a
`rastermoves.refinement.RefinerSpec` whose ID exactly matches the entry point. The
backend constructor accepts `(spec, *, cache_dir, offline, device, precision, offload,
vae_tiling, scheduler, strict_checksums)`. It must implement:

- `predict(PIL_RGB_tile, *, options: RefineOptions, seed: int) -> (PIL_RGB_tile, dict)`.
  The output must have exactly the input tile dimensions; record real step counts
  under `executed_denoising_steps` rather than estimates.
- `close()` releases persistent runtime/device resources.
- Optional `inventory`, `device`, `precision` attributes become report metadata.

External packages are explicitly installed by the user, not downloaded as model code.
`--external-plugins` is required. Built-in names always resolve to built-ins and cannot
be shadowed. The CLI list/download commands enumerate the two built-in specifications;
custom installed IDs are selected explicitly with `--refiner ID`. Refiner sessions,
like upscaler sessions, are not thread-safe. Inference failure is not silently treated
as no-op success. Native adapter loading uses explicit standard classes, safetensors
and restricted state dictionaries; custom plugins are trusted local Python and are
responsible for preserving those guarantees.

The outer refinement layer owns immutable-baseline tiling, masking, alpha, compositing,
provenance, output collision checks and stage-level resume. Do not generate another
alpha channel or change dimensions inside `predict`. For deterministic experiments,
include third-party code identity in its own metadata and start a new comparison
folder after changing code; package-version fingerprinting cannot hash arbitrary
external implementations. See [REFINEMENT.md](REFINEMENT.md).
