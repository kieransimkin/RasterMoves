# Migrating from UpscaleLab to RasterMoves

RasterMoves 0.1.1 is the renamed package. Image-processing arguments, model IDs,
model manifests, publisher checksums, and inference backends are unchanged.

## Names to change

| Surface | Previous name | New name |
| --- | --- | --- |
| Installable distribution | `upscalelab` | `rastermoves` |
| Command | `upscalelab` | `rastermoves` |
| Module execution | `python -m upscalelab` | `python -m rastermoves` |
| Python API | `from upscalelab import Upscaler` | `from rastermoves import Upscaler` |
| Model plugin entry-point group | `upscalelab.models` | `rastermoves.models` |
| Backend plugin entry-point group | `upscalelab.backends` | `rastermoves.backends` |
| Cache override | `UPSCALELAB_CACHE` | `RASTERMOVES_CACHE` |
| Extra manifest directories | `UPSCALELAB_PLUGIN_PATH` | `RASTERMOVES_PLUGIN_PATH` |
| Offline switch | `UPSCALELAB_OFFLINE` | `RASTERMOVES_OFFLINE` |
| Opt-in live tests | `UPSCALELAB_LIVE` | `RASTERMOVES_LIVE` |
| Platform cache/config application name | `upscalelab` | `rastermoves` |
| Version key in `doctor` JSON | `upscalelab` | `rastermoves` |

`HF_TOKEN` is unchanged. The public class names `Upscaler`, `Registry`, and
`ModelPlugin` are also unchanged; only their package path moves. Output provenance
now additionally includes `software: "RasterMoves"` and `software_version: "0.1.1"`.
`rastermoves --version` prints `RasterMoves 0.1.1`.

Legacy command/import/environment/entry-point aliases are not provided. An already
installed copy of the old distribution remains installed until you remove it, and
its old executable would still run that old version. Existing files are never moved,
deleted, or rewritten automatically.

## Install the renamed package

Extract the new source archive, open a terminal inside its `rastermoves` directory,
and activate the intended virtual environment:

```bash
python -m pip install -e ".[torch,onnx,gdrive]"
rastermoves --version
rastermoves doctor
```

Installing only `.[torch]` or `.[onnx]` is also supported. To remove the old
installation, after updating scripts that use it:

```bash
python -m pip uninstall upscalelab
```

This uninstalls the previous Python distribution; it does not delete its downloaded
model cache. Do not delete the old source directory while other tools still depend
on an editable installation of it.

## Reuse weights without downloading them again

The default cache location now belongs to `rastermoves`. To reuse an earlier cache,
pass its root explicitly; its contents and integrity-verification format are compatible:

```bash
rastermoves --cache-dir "/path/to/existing-cache" upscale input.png -o output.png --offline
```

Use the actual cache path from your previous installation. A custom `--cache-dir` or
`UPSCALELAB_CACHE` setting takes precedence over the old platform default. To print
the old default without importing the old package:

```bash
python -c "from platformdirs import user_cache_path; print(user_cache_path('upscalelab'))"
```

Set `RASTERMOVES_CACHE` to that root to reuse it for subsequent commands:

```bash
# Linux / macOS
export RASTERMOVES_CACHE="/path/to/existing-cache"
rastermoves upscale input.png -o output.png --offline
```

```powershell
# Windows PowerShell
$env:RASTERMOVES_CACHE = 'C:\path\to\existing-cache'
rastermoves upscale input.png -o output.png --offline
```

Alternatively, copy the old cache contents into the new default directory when neither
tool is running. Print the new default with
`python -c "from platformdirs import user_cache_path; print(user_cache_path('rastermoves'))"`.
Copy the full cache, including integrity receipts, `catalog.json`, and
`imported-models` where present. `--offline` succeeds only if the selected model's
metadata and weights are already present and valid.

The default **user model configuration directory** also changes. Custom JSON manifests
can be copied to `platformdirs.user_config_path('rastermoves') / 'models'`, or you can
keep their current directory and supply `--plugin-dir` or `RASTERMOVES_PLUGIN_PATH`.
Model IDs and JSON schema do not change.

## Update external Python plugins

Replace imports of `upscalelab` with `rastermoves`, update the plugin's dependency to
`rastermoves>=0.1.1`, and rename its entry-point group in `pyproject.toml`:

```toml
[project.entry-points."rastermoves.models"]
my_model = "my_plugin:make_plugin"

[project.entry-points."rastermoves.backends"]
my_backend = "my_plugin:Backend"
```

Reinstall the plugin after changing its entry-point metadata. Python plugins still
require explicit `--external-plugins` opt-in. Plain JSON model plugins do not.
