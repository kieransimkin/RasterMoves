# Timing and CPU/memory resource traces

## Enable recording

Install the small optional tracing dependency once. It is already included in
`rastermoves[all]`, the development extra, and the published CPU container builds.

```bash
# From the source checkout:
python -m pip install -e ".[trace]"

# From a published 0.2.1-or-newer package:
python -m pip install "rastermoves[trace]"
```

No tracing dependency is imported and no sampling thread or trace file is created
when tracing is disabled. Normal plugin signatures and model selection are unchanged.

```bash
# Single image.
rastermoves upscale input.png -o output.png --trace

# A directory of images, one model loaded once for the batch.
rastermoves upscale ./originals -o ./upscaled --recursive --trace

# Every locally registered model, attempted sequentially.
rastermoves upscale input.png --all-models -o comparison --trace

# Also refresh the full OpenModelDB catalogue; this can download many model weights.
rastermoves upscale input.png --all-models --sync-models -o comparison --trace

# An explicit trace filename and a 100 ms periodic sampling interval.
rastermoves upscale input.png -o output.png --trace-file traces/run-01.json --trace-interval 0.1
```

All trace options go **after `upscale`**. `--trace-file` also enables tracing; `--trace`
is not additionally required. The sampling interval defaults to **0.25 seconds** and
must be finite and within **0.05–60 seconds**. `--trace-interval` by itself is an error.
Short intervals provide more samples but increase overhead and trace-file size.

For single-image output, the default is `output.png.trace.json`. For batch and all-model
runs, it is `OUTPUT_DIR/trace.json`. If a default already exists, the recorder creates
`output.png.trace.1.json`, `trace.1.json`, etc. It never replaces an earlier trace.
An explicit path must end in `.json` and must not already exist, even with `--overwrite`.
Image output/provenance/summary collisions are rejected. Output directories are created
as needed; the chosen trace path is printed to stderr at startup and completion.

## What is recorded

The file is a Chrome JSON trace with two top-level parts:

- `traceEvents`: nested stage durations (`X`), timestamped numeric counters (`C`),
  process/thread labels (`M`), and instant event markers (`i`). Event timestamps and
  durations are **microseconds relative to the recording start**.
- `rastermoves`: versioned metadata, interval, completion/error state, overall resource
  summary, and aggregate timings keyed by model and stage. Summary durations are
  **seconds** and memory values are **bytes**. `displayTimeUnit: ms` is a viewer hint,
  not a change to the underlying event timestamp units.

Periodic resource counters:

| Counter | Meaning |
| --- | --- |
| `process.cpu_percent` | CPU-time delta / elapsed wall time × 100. One fully occupied logical CPU is 100%; multiple busy threads can exceed 100%. |
| `process.cpu_seconds` | User + system CPU time consumed by this process since recording began. |
| `process.rss_bytes` | OS-reported resident memory / working set. Includes native runtime memory, not just Python objects. |
| `process.vms_bytes` | psutil's virtual-memory measure; on Windows this maps to its `pagefile` field. Not physical RAM usage. |
| `process.threads` | OS-reported process thread count, including native runtime workers and the sampler. |
| `system.memory_available_bytes` | Memory reported available by psutil for system-level context. |
| `system.memory_percent` | System memory pressure, calculated by psutil from total and available memory. |

CPU percentages are calculated from Python's process CPU timer, not normalized to
host CPU count and not based on psutil's meaningless first percentage reading. The
first sample has no CPU-percentage counter. Overall/per-stage mean percentages use
total CPU time divided by actual wall duration, not an unweighted average of samples.
CPU values are **process-wide**, including all Python/native threads and tracing
itself; they are not specific to the Python thread carrying a stage marker.

Additional memory/thread samples are taken at stage boundaries. Accordingly,
`sample_count` includes boundary observations, while `periodic_sample_count` counts
the initial, periodic and final full observations. A short command still has initial
and final observations, even if it finishes before the periodic timer fires.

Timed stages cover command execution, catalogue sync/search, comparison validation,
model sessions, image processing, preprocessing, weight acquisition/cache checks,
actual download attempts, weight verification, backend construction, input conversion,
inference passes/attempts, postprocessing (including optional alpha inference), output
writing, provenance/checksum work and model cleanup. Missing stages mean that work
was not reached or was not necessary; a warm-cache load need not have a `download`
stage. `oom_retry` markers identify smaller-tile restarts. A custom plugin that bypasses
built-in download/backend code still gets surrounding model/pipeline timings, but needs
its own spans for custom internal stages.

Nested durations are **inclusive**. For example, `load_model` contains `weights` and
`backend_load`; `inference` contains `inference_attempt`; `model` contains almost the
whole attempt. Do **not** add parent and child durations to derive elapsed time.

## View or analyse the trace

Open [Perfetto](https://ui.perfetto.dev/) and use **Open trace file** to load the JSON.
Its documented Chrome JSON support provides duration slices and counter tracks; the
recorder does not require the Perfetto package or SDK at runtime. No upload/network
operation is performed by RasterMoves' recorder itself.

Read an overall summary programmatically:

```python
import json
from pathlib import Path

trace = json.loads(Path("comparison/trace.json").read_text(encoding="utf-8"))
metadata = trace["rastermoves"]
print(metadata["status"], metadata["trace_complete"])
print(json.dumps(metadata["summary"], indent=2))
for stage in metadata["stages"]:
    if stage["stage"] == "inference":
        print(stage["model_id"], stage["wall_seconds"], stage["process_cpu_seconds"])
```

For an all-model run, `summary.json` additionally has `trace_file`, and attempted model
rows have `performance` objects with:

`wall_seconds`, `process_cpu_seconds`, `mean_process_cpu_percent`,
`peak_sampled_rss_bytes`, `peak_sampled_vms_bytes`, `peak_sampled_threads`,
`status`, and `resource_samples_complete`.

These metrics include that attempt's session construction, file processing, output
checksums, cleanup and garbage collection. Catalogue discovery and writing the sweep
summary are outside the per-model scope. Failed/interrupted attempts are measured too.
The top-level trace also has an overall summary and the finer-grained stage metrics.

## Comparisons and resume

Tracing options are deliberately excluded from resume fingerprints, model options,
and image options. You can enable or disable tracing, change its interval or choose a
new trace filename while resuming the same software version/input/settings.

```bash
rastermoves upscale input.png --all-models -o comparison --resume --trace
```

Results reused from an earlier run emit `model_reused` markers and their checksum
verification is timed. They do **not** receive new inference durations or a newly
measured `performance` object. The existing `elapsed_seconds` field, where retained,
is historical. Read the original trace to examine the original inference. Each
invocation has a separate trace; numbered defaults preserve older ones.

Version identity remains part of resume: applying this patch moves to 0.2.1, so a
0.2.0 comparison requires a new results folder or an explicit `--overwrite` run.

`--dry-run --trace` records only catalogue discovery/planning and creates the trace
file/directory, not model images or `summary.json`. It does not download weights or
load a neural backend. `--sync-models` still fetches catalogue metadata.

## Scope, accuracy, overhead and failure behaviour

**This is a lightweight process monitor, not a function-stack profiler, exact heap
profiler, GPU profiler, or controlled model benchmark.** It does not record GPU
utilization/VRAM, child-process CPU/RAM, USS/PSS, network throughput, or separate
system-wide CPU utilization. System memory context is what psutil reports; no special
container/cgroup budget accounting is performed.

Memory peaks are the largest **sampled** values, including boundary observations.
Allocations freed between samples can be missed. All-model values include the Python
interpreter, imported libraries, allocator caches and any memory retained after earlier
models; they are not isolated per-checkpoint allocation deltas. Use separate processes
and controlled input/cache conditions for rigorous independent-model benchmarks.

Wall durations use `time.perf_counter()`; CPU durations use `time.process_time()`.
Recording begins after CLI argument parsing/job planning and psutil import, and
covers catalogue refresh when requested, pipeline work and model cleanup. It does
not measure Python/CLI module-import startup. Time spent waiting for downloads or
GPU work counts as wall time but is not necessarily CPU activity. No extra GPU
synchronization is inserted; custom asynchronous backends are measured as host calls,
not as guaranteed device completion. Sampling can be delayed by OS scheduling or
long native operations holding Python's GIL. The file records actual timestamps.

The sampler is one daemon thread and is stopped/joined on normal completion, Python
exceptions and Ctrl+C. Events are written incrementally rather than accumulated as an
unbounded in-memory list; only active spans and stage aggregates stay in memory. The
stream is **not complete JSON until finalization**. Normal exceptions and Ctrl+C
finalize the trace with failure/interruption status before the existing CLI error/exit
handling proceeds. SIGKILL, native crashes, forced shutdowns and an OS OOM kill cannot
be finalized by Python and can leave incomplete JSON.

Failure to install psutil or open a safe trace path fails before model work. A later
sampling/write error is logged, stops further sampling and is reflected in
`trace_complete`/`trace_error` where finalization is still possible. Such an error does
not replace an existing application exception. If application work otherwise succeeds,
a trace failure makes the command fail rather than silently claiming a good trace.
Disk-full/I/O errors can themselves prevent a valid JSON footer from being saved.

Trace metadata does not enumerate environment variables, full command lines, request
headers or image contents. CLI image spans include **filenames**, and model IDs, timing,
process IDs and platform information are present. Exceptions contribute type names,
not arbitrary exception messages. Custom API metadata/span attributes are recorded as
supplied. Review trace files before sharing them.

## Python API / plugin instrumentation

```python
from rastermoves import Upscaler
from rastermoves.tracing import TraceRecorder, span

with TraceRecorder("my-run.json", interval=0.25):
    with span("model", model_id="4x-realesr-general-x4v3"):
        with Upscaler("4x-realesr-general-x4v3") as up:
            up.upscale_file("input.png", "output.png", report=True)
```

Use `with span("custom_stage"):` or `@traced("custom_stage")` within a trusted plugin
without adding new parameters to its public interface. Calls are no-ops outside an
active recorder. This API follows the existing synchronous session model; recorder
contexts cannot be nested or reused, and context does not automatically propagate to
new Python threads or subprocesses. A process-level sample still includes its native
worker threads. Do not activate two simultaneous recorders in the same operation.

## References

- [psutil 7.2 API](https://psutil.io/7.2/): memory/CPU metric meanings and platform differences.
- [Python time module](https://docs.python.org/3/library/time.html): wall and process CPU timers.
- [Perfetto Chrome JSON support](https://perfetto.dev/docs/getting-started/other-formats#chrome-json-format): duration and counter events.
