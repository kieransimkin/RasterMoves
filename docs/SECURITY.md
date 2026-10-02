# Security and operational boundaries

Treat model files and third-party Python plugins as untrusted unless their source is
trusted. Integrity verification is not a malware verdict and cannot establish the
trustworthiness of the original creator or a compromised catalogue entry.

## Measures implemented

PyTorch checkpoints use `torch.load(..., weights_only=True, map_location="cpu")` and
require PyTorch 2.6 or newer. There is no unsafe fallback. Safetensors uses the dedicated
tensor loader. Suspected TorchScript archives are refused. Checkpoints requiring custom
Python objects must be converted in a separately trusted environment; the tool will
not allowlist arbitrary objects. The minimum version is not a claim that every newer
version is vulnerability-free: keep PyTorch and all dependencies patched.

Downloaded files are checked against author/catalogue SHA-256 and byte size when
provided. `--strict-checksums` requires a supplied digest. Hashless downloads receive
a local first-seen integrity receipt only; replacing both the file and that receipt
is outside this protection. Cache directories must not be writable by untrusted users.

Downloads are written to partial files under file locks, then atomically published.
Basic HTTP transfers enforce HTTPS at every redirect, timeouts, size caps and retries.
Hugging Face and Google Drive transfers are delegated to their client libraries:
known advertised sizes are checked before download, and actual files are checked
after download. Unknown-size transfers through those clients are not preempted at
the exact byte cap; enforce filesystem quotas when that matters.

The archive catalogue fallback reads JSON members without extracting paths or
executing source. Model IDs cannot be filesystem paths. JSON model manifests are data,
not code. Python entry-point discovery is disabled unless explicitly enabled.

ONNX Runtime receives self-contained model bytes, not a path from which arbitrary
external-data files can be resolved. Multi-file ONNX exports require an explicit
separate implementation. This restriction does not sandbox ONNX Runtime itself.

## Not a sandbox

A malicious or broken model may still exhaust CPU, GPU memory or RAM or trigger a
runtime vulnerability. Model inference is not executed in a process sandbox. Do not
expose this CLI as a public upload-and-run service without process isolation, resource
limits, filesystem restrictions, network restrictions, and separate security review.

The output pixel limit reduces accidental allocations but is not a hard process
memory limit. Compressed images and models can expand significantly. Host output
arrays are materialized at the model's native scale. Tile retries do not recover
from insufficient RAM, huge model load allocations, or all classes of GPU failure.

The application uploads no source images; normal networking retrieves catalogue
metadata and weights. Hosting services see those requests. The Hugging Face client
may have its own network/telemetry behavior controlled by its documented settings.
An opt-in third-party Python plugin can perform arbitrary actions under your account;
only enable installed plugins you trust.

## Local paths and output

In-place source replacement is forbidden. Destination overwrite requires explicit
permission. Outputs are written atomically, but the image and optional JSON report
are not a single two-file transaction; a report write failure can leave a valid image.
EXIF/GPS metadata is not copied. Provenance sidecars contain local input/output paths;
review them before sharing. Cached manifests may contain author-provided URLs and
text. Use appropriate terminal/log handling for third-party catalogue content.
