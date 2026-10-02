"""Real-process sampling plus deterministic recorder and CLI regression tests."""
from contextlib import nullcontext
from io import StringIO
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from rastermoves import __version__
from rastermoves import tracing
from rastermoves.cli import main
from rastermoves.errors import UpscaleError
from rastermoves.pipeline import Upscaler
from rastermoves.plugins import ModelPlugin
from rastermoves.tracing import TraceRecorder, current_trace, span, traced


def read_trace(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(data["traceEvents"], list)
    assert data["displayTimeUnit"] == "ms"
    assert data["rastermoves"]["software_version"] == __version__
    for item in data["traceEvents"]:
        if "ts" in item:
            assert item["ts"] >= 0 and math.isfinite(item["ts"])
    return data


def slices(data, name=None):
    return [item for item in data["traceEvents"] if item["ph"] == "X" and (name is None or item["name"] == name)]


class RepeatModel:
    def __init__(self, scale=2):
        self.scale = scale
        self.input_channels = self.output_channels = 3
        self.device, self.precision, self.tiling = "cpu", "fp32", "supported"
        self.closed = False

    def predict(self, array):
        return np.repeat(np.repeat(array, self.scale, 0), self.scale, 1)

    def close(self):
        self.closed = True

    def clear_cache(self):
        pass


@pytest.fixture
def fake_models(monkeypatch):
    models = []

    def load(self, downloader, **kwargs):
        model = RepeatModel(self.spec.scale)
        models.append(model)
        return model, None, None

    monkeypatch.setattr(ModelPlugin, "load", load)
    return models


@pytest.fixture
def image_args(tmp_path):
    source = tmp_path / "input.png"
    Image.new("RGB", (6, 5), (25, 50, 100)).save(source)
    return ["--cache-dir", str(tmp_path / "cache"), "upscale", str(source), "--tile", "0"]


def test_real_process_sampling_and_nested_spans(tmp_path):
    path = tmp_path / "trace.json"
    with TraceRecorder(path, interval=0.05) as recorder:
        assert current_trace() is recorder
        with span("model", model_id="demo") as model:
            with span("compute"):
                allocation = bytearray(4 * 1024 * 1024)
                deadline = time.perf_counter() + 0.14
                while time.perf_counter() < deadline:
                    sum(i * i for i in range(1000))
            assert len(allocation) > 0
        assert model.summary["peak_sampled_rss_bytes"] > 0
        # Allow a loaded CI host to schedule at least one periodic sample.
        deadline = time.perf_counter() + 2.0
        while recorder._periodic_count < 2 and time.perf_counter() < deadline:
            time.sleep(0.01)
    assert current_trace() is None
    assert not recorder._thread.is_alive()
    assert not any(t.name == "rastermoves-resource-trace" for t in threading.enumerate())
    data = read_trace(path)
    info = data["rastermoves"]
    assert info["trace_complete"] and info["status"] == "completed"
    assert info["summary"]["wall_seconds"] >= 0.14
    assert info["summary"]["process_cpu_seconds"] > 0
    assert info["summary"]["periodic_sample_count"] >= 3
    assert info["summary"]["peak_sampled_rss_bytes"] > 0
    assert info["summary"]["peak_sampled_threads"] >= 2
    names = {e["name"] for e in data["traceEvents"] if e["ph"] == "C"}
    assert {"process.cpu_percent", "process.cpu_seconds", "process.rss_bytes", "process.vms_bytes",
            "process.threads", "system.memory_available_bytes", "system.memory_percent"} <= names
    compute = slices(data, "compute")[0]
    parent = slices(data, "model")[0]
    assert compute["args"]["model_id"] == "demo"
    assert compute["ts"] >= parent["ts"]
    assert compute["ts"] + compute["dur"] <= parent["ts"] + parent["dur"] + 0.01
    assert len(info["stages"]) == 2


@pytest.mark.parametrize("interval", [0, -1, 0.001, math.nan, math.inf, -math.inf, 61, 1e300])
def test_invalid_intervals_rejected_before_file_creation(tmp_path, interval):
    with pytest.raises(UpscaleError, match="trace-interval"):
        TraceRecorder(tmp_path / "invalid.json", interval=interval)
    assert not list(tmp_path.iterdir())


def test_noop_and_decorator_preserve_return_values_and_signature():
    @traced("function")
    def function(number):
        """Preserved documentation."""
        return number * 2
    assert function(3) == 6
    assert function.__name__ == "function"
    assert function.__wrapped__(4) == 8
    assert function.__doc__ == "Preserved documentation."
    with span("disabled") as measured:
        assert measured is None
    assert current_trace() is None


def test_disabled_cli_does_not_import_psutil_or_start_sampler(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    script = ("import sys, threading; from rastermoves.cli import main; "
              f"assert main(['--cache-dir', {str(tmp_path / 'cache')!r}, 'models', '--json']) == 0; "
              "assert 'psutil' not in sys.modules; "
              "assert not any(t.name == 'rastermoves-resource-trace' for t in threading.enumerate())")
    completed = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("exception", [ValueError("secret-token=hidden"), KeyboardInterrupt(), SystemExit(9)])
def test_exceptions_finalize_trace_without_swallowing_or_leaking_threads(tmp_path, exception):
    recorder = TraceRecorder(tmp_path / "failure.json")
    with pytest.raises(type(exception)):
        with recorder:
            with span("failing"):
                raise exception
    data = read_trace(recorder.path)
    expected = "failed" if isinstance(exception, ValueError) else "interrupted"
    assert data["rastermoves"]["status"] == expected
    assert slices(data, "failing")[0]["args"]["status"] == expected
    assert "hidden" not in recorder.path.read_text()
    assert current_trace() is None and not recorder._thread.is_alive()


def test_explicit_path_never_overwrites_and_default_names_increment(tmp_path):
    original = tmp_path / "trace.json"
    original.write_text("preserve me")
    with pytest.raises(UpscaleError, match="already exists"):
        with TraceRecorder(original):
            pass
    with TraceRecorder(original, auto_suffix=True) as first:
        pass
    with TraceRecorder(original, auto_suffix=True) as second:
        pass
    assert first.path.name == "trace.1.json" and second.path.name == "trace.2.json"
    assert original.read_text() == "preserve me"
    read_trace(first.path)
    read_trace(second.path)


def test_symlink_path_is_not_followed(tmp_path):
    target, link = tmp_path / "target.json", tmp_path / "link.json"
    target.write_text("preserve")
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation unavailable on this host")
    with pytest.raises(UpscaleError):
        with TraceRecorder(link):
            pass
    assert target.read_text() == "preserve"


def test_recorder_rejects_nested_reuse_and_non_json(tmp_path):
    with pytest.raises(UpscaleError, match="extension"):
        TraceRecorder(tmp_path / "out.png")
    r = TraceRecorder(tmp_path / "first.json")
    with r:
        with pytest.raises(UpscaleError, match="nested"):
            with TraceRecorder(tmp_path / "second.json"):
                pass
    with pytest.raises(UpscaleError, match="reused"):
        with r:
            pass
    assert not (tmp_path / "second.json").exists()


def test_cpu_percent_not_machine_normalized_and_first_sample_omitted(tmp_path, monkeypatch):
    recorder = TraceRecorder(tmp_path / "unused.json")
    recorder._file = StringIO()
    recorder._process = SimpleNamespace(
        oneshot=nullcontext, memory_info=lambda: SimpleNamespace(rss=4096, vms=8192), num_threads=lambda: 3)
    recorder._psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=100000, percent=25))
    recorder._start_wall, recorder._start_cpu = 0.0, 5.0
    wall, cpu = [1.0], [5.0]
    monkeypatch.setattr(tracing.time, "perf_counter", lambda: wall[0])
    monkeypatch.setattr(tracing.time, "process_time", lambda: cpu[0])
    recorder._sample()
    assert "process.cpu_percent" not in recorder._file.getvalue()
    # Boundary memory probes must not reset the periodic CPU baseline.
    wall[0], cpu[0] = 2.0, 6.0
    recorder._sample(boundary=True)
    wall[0], cpu[0] = 3.0, 9.0
    recorder._sample()
    records = json.loads("[" + recorder._file.getvalue() + "]")
    percent = [e for e in records if e["name"] == "process.cpu_percent"]
    assert len(percent) == 1 and percent[0]["args"]["value"] == 200.0
    assert recorder._sample_count == 3 and recorder._periodic_count == 2


def test_sampling_failure_is_reported_and_never_masks_work_error(tmp_path, monkeypatch):
    r = TraceRecorder(tmp_path / "sampling-error.json")
    with pytest.raises(ValueError, match="work failure"):
        with r:
            def fail():
                raise PermissionError("private details")
            monkeypatch.setattr(r._process, "oneshot", nullcontext)
            monkeypatch.setattr(r._process, "memory_info", fail)
            r._sample()
            raise ValueError("work failure")
    data = read_trace(r.path)
    assert not data["rastermoves"]["trace_complete"]
    assert "PermissionError" in data["rastermoves"]["trace_error"]
    assert "private details" not in r.path.read_text()
    assert current_trace() is None and not r._thread.is_alive()


def test_trace_fault_makes_successful_work_nonzero(tmp_path, monkeypatch):
    r = TraceRecorder(tmp_path / "fault.json")
    with pytest.raises(UpscaleError, match="sampling failed"):
        with r:
            monkeypatch.setattr(r._process, "oneshot", nullcontext)
            monkeypatch.setattr(r._process, "memory_info", lambda: (_ for _ in ()).throw(OSError("denied")))
            r._sample()
    data = read_trace(r.path)
    assert not data["rastermoves"]["trace_complete"]


def test_trace_stream_does_not_buffer_events_in_memory(tmp_path):
    r = TraceRecorder(tmp_path / "stream.json", interval=60)
    with r:
        for i in range(500):
            r.event("marker", index=i)
        assert r.path.stat().st_size > 10000  # written before finalization
        assert not hasattr(r, "events") and not r._stages
    assert len([e for e in read_trace(r.path)["traceEvents"] if e.get("name") == "marker"]) == 500


def test_cli_default_single_image_trace_and_disabled_result_match(tmp_path, image_args, fake_models, capsys):
    output, plain = tmp_path / "result.png", tmp_path / "plain.png"
    assert main(image_args + ["-o", str(output), "--trace", "--report"]) == 0
    assert main(image_args + ["-o", str(plain)]) == 0
    np.testing.assert_array_equal(np.asarray(Image.open(output)), np.asarray(Image.open(plain)))
    data = read_trace(tmp_path / "result.png.trace.json")
    assert data["rastermoves"]["exit_code"] == 0
    assert {"model", "image", "preprocess", "load_model", "inference", "inference_attempt",
            "postprocess", "write_image", "write_provenance", "model_cleanup"} <= {s["name"] for s in slices(data)}
    assert all(model.closed for model in fake_models)
    assert not (tmp_path / "plain.png.trace.json").exists()
    assert "sampled peak RSS" in capsys.readouterr().err


def test_cli_trace_file_implies_enable_and_nested_parent_created(tmp_path, image_args, fake_models):
    trace_file = tmp_path / "profiling" / "named.json"
    assert main(image_args + ["-o", str(tmp_path / "output.png"), "--trace-file", str(trace_file),
                              "--trace-interval", "0.05"]) == 0
    assert read_trace(trace_file)["rastermoves"]["interval_seconds"] == 0.05


@pytest.mark.parametrize("interval", ["nan", "inf", "0", "-1", "0.01", "100"])
def test_cli_invalid_interval_returns_nonzero_without_model_work(tmp_path, image_args, fake_models, interval):
    assert main(image_args + ["--trace", "--trace-interval", interval]) == 1
    assert fake_models == []


def test_cli_interval_without_trace_is_rejected(image_args, fake_models):
    assert main(image_args + ["--trace-interval", "1"]) == 1
    assert fake_models == []


@pytest.mark.parametrize("filename", ["output.png.json", "input.png", "output.png"])
def test_cli_trace_cannot_clobber_image_or_provenance(tmp_path, image_args, fake_models, filename):
    source_bytes = (tmp_path / "input.png").read_bytes()
    assert main(image_args + ["-o", str(tmp_path / "output.png"), "--trace-file", str(tmp_path / filename)]) == 1
    assert (tmp_path / "input.png").read_bytes() == source_bytes
    assert not fake_models
    assert not (tmp_path / "output.png.json").exists()


def test_cli_missing_psutil_errors_before_model_work(tmp_path, image_args, fake_models, monkeypatch, capsys):
    actual = tracing.importlib.import_module
    def missing(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("missing")
        return actual(name, *args, **kwargs)
    monkeypatch.setattr(tracing.importlib, "import_module", missing)
    assert main(image_args + ["--trace"]) == 1
    assert not fake_models
    assert "rastermoves[trace]" in capsys.readouterr().err
    assert not list(tmp_path.glob("*.trace.json"))


def test_cli_all_models_reports_per_model_and_resume_does_not_retime_reused(tmp_path, image_args, fake_models):
    out = tmp_path / "comparison"
    args = image_args + ["--all-models", "-o", str(out)]
    # First produce results without tracing; toggling trace must not break resume.
    assert main(args) == 0
    old = json.loads((out / "summary.json").read_text())
    assert "trace_file" not in old and all("performance" not in row for row in old["results"])
    assert main(args + ["--resume", "--trace"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["counts"]["reused"] == 8
    assert all("performance" not in row for row in summary["results"])
    data = read_trace(out / "trace.json")
    assert len([e for e in data["traceEvents"] if e["name"] == "model_reused"]) == 8
    assert not slices(data, "inference")
    # A new timed run includes independently measured model metrics.
    assert main(args + ["--overwrite", "--trace"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["counts"]["success"] == 8
    assert summary["trace_file"].endswith("trace.1.json")
    for row in summary["results"]:
        assert row["performance"]["wall_seconds"] > 0
        assert row["performance"]["peak_sampled_rss_bytes"] > 0
        assert row["performance"]["status"] == "success"
    data = read_trace(out / "trace.1.json")
    assert len(slices(data, "model")) == 8
    assert {e["args"]["model_id"] for e in slices(data, "model")} == {r["model_id"] for r in summary["results"]}
    assert all(model.closed for model in fake_models)


def test_cli_all_models_continues_after_failure_and_records_failed_metrics(tmp_path, image_args, monkeypatch):
    def fail(self, downloader, **kwargs):
        if self.spec.id == "4x-Remacri":
            raise UpscaleError("synthetic failure")
        return RepeatModel(self.spec.scale), None, None
    monkeypatch.setattr(ModelPlugin, "load", fail)
    out = tmp_path / "comparison"
    assert main(image_args + ["--all-models", "-o", str(out), "--trace"]) == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["counts"]["failed"] == 1 and summary["counts"]["success"] == 7
    failed = next(r for r in summary["results"] if r["status"] == "failed")
    assert failed["performance"]["status"] == "failed"
    data = read_trace(out / "trace.json")
    assert data["rastermoves"]["exit_code"] == 1
    assert data["rastermoves"]["status"] == "completed_with_errors"


def test_cli_interrupt_preserves_trace_and_partial_sweep_summary(tmp_path, image_args, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(ModelPlugin, "load", interrupt)
    out = tmp_path / "comparison"
    assert main(image_args + ["--all-models", "-o", str(out), "--trace"]) == 130
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "interrupted"
    row = next(row for row in summary["results"] if row["status"] == "interrupted")
    assert row["performance"]["status"] == "interrupted"
    data = read_trace(out / "trace.json")
    assert data["rastermoves"]["status"] == "interrupted" and data["rastermoves"]["exit_code"] == 130
    assert not any(t.name == "rastermoves-resource-trace" for t in threading.enumerate())


def test_cli_batch_trace_includes_each_image(tmp_path, fake_models):
    source, out = tmp_path / "images", tmp_path / "results"
    source.mkdir()
    for name in ("first.png", "second.png"):
        Image.new("RGB", (5, 4)).save(source / name)
    args = ["--cache-dir", str(tmp_path / "cache"), "upscale", str(source), "-o", str(out), "--trace"]
    assert main(args) == 0
    data = read_trace(out / "trace.json")
    assert len(slices(data, "image")) == 2 and len(slices(data, "model")) == 1
    assert len(fake_models) == 1 and fake_models[0].closed


@pytest.mark.parametrize("filename", ["summary.json", "model-4x-Remacri.png.json"])
def test_trace_rejects_sweep_outputs_before_file_creation(tmp_path, image_args, fake_models, filename):
    out = tmp_path / "comparison"
    assert main(image_args + ["--all-models", "-o", str(out), "--trace-file", str(out / filename)]) == 1
    assert not out.exists() and not fake_models


def test_dry_run_trace_records_only_planning(tmp_path, image_args, fake_models, capsys):
    out = tmp_path / "planned"
    assert main(image_args + ["--all-models", "--dry-run", "-o", str(out), "--trace"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert not fake_models and not (out / "summary.json").exists()
    data = read_trace(out / "trace.json")
    assert slices(data, "catalog_search") and not slices(data, "inference")


def test_python_api_context_instruments_without_signature_changes(tmp_path):
    with TraceRecorder(tmp_path / "api.json"):
        with Upscaler(cache_dir=tmp_path / "cache") as up:
            up.loaded = RepeatModel(2)
            result = up.upscale_image(Image.new("RGB", (5, 4)), tile=0)
            assert result.size == (10, 8)
            result.close()
    assert slices(read_trace(tmp_path / "api.json"), "inference")


def test_download_verification_and_backend_loading_have_separate_spans(tmp_path, monkeypatch, weight_data):
    from rastermoves import downloads, plugins
    from rastermoves.downloads import Downloader
    from rastermoves.specs import ModelSpec
    blob, resource = weight_data
    plugin = ModelPlugin(ModelSpec(id="test-weights", name="Test", scale=2, resources=(resource,)))
    # Exercise the real download dispatch/verification/cache code, but not the network.
    monkeypatch.setattr(downloads, "stream_download", lambda url, dest, **kwargs: dest.write_bytes(blob))
    monkeypatch.setattr(plugins, "find_spec", lambda name: True)
    monkeypatch.setattr(plugins, "backend_factories", lambda **kwargs: {"spandrel": lambda *args, **kwargs: RepeatModel(2)})
    with TraceRecorder(tmp_path / "download.json"):
        dl = Downloader(tmp_path / "cache")
        with span("model", model_id=plugin.spec.id):
            model, path, actual = plugin.load(dl)
            assert path.read_bytes() == blob and actual == resource
            model.close()
        # Cached load must have no second download event.
        model, _, _ = plugin.load(dl)
        model.close()
    data = read_trace(tmp_path / "download.json")
    assert len(slices(data, "weights")) == 2
    assert len(slices(data, "download")) == 1
    assert len(slices(data, "verify_weights")) == 2
    assert len(slices(data, "backend_load")) == 2
    assert slices(data, "download")[0]["args"]["model_id"] == "test-weights"
    assert "https://example.org" not in (tmp_path / "download.json").read_text()


def test_oom_retries_are_visible_in_trace(tmp_path):
    from rastermoves.errors import BackendOOM
    from rastermoves.tiling import upscale_array
    class Model(RepeatModel):
        failed = False
        def predict(self, data):
            if not self.failed:
                self.failed = True
                raise BackendOOM("synthetic OOM")
            return super().predict(data)
    with TraceRecorder(tmp_path / "oom.json"):
        output = upscale_array(np.zeros((40, 40, 3), dtype=np.float32), Model(), tile=32, overlap=4, pad=2)
        assert output.shape == (80, 80, 3)
    data = read_trace(tmp_path / "oom.json")
    assert len(slices(data, "inference_attempt")) == 2
    assert slices(data, "inference_attempt")[0]["args"]["error_type"] == "BackendOOM"
    retry = next(e for e in data["traceEvents"] if e["name"] == "oom_retry")
    assert retry["args"]["tile"] == 16


def test_single_model_failure_still_has_trace_and_exit_code(tmp_path, image_args, monkeypatch):
    def fail(*args, **kwargs):
        raise UpscaleError("synthetic load failure")
    monkeypatch.setattr(ModelPlugin, "load", fail)
    assert main(image_args + ["-o", str(tmp_path / "failed.png"), "--trace"]) == 1
    data = read_trace(tmp_path / "failed.png.trace.json")
    assert data["rastermoves"]["status"] == "failed" and data["rastermoves"]["exit_code"] == 1
    assert slices(data, "model")[0]["args"]["error_type"] == "UpscaleError"
    assert not (tmp_path / "failed.png").exists()


def test_pipeline_api_refuses_to_overwrite_active_trace_as_provenance(tmp_path):
    source, output = tmp_path / "input.png", tmp_path / "output.png"
    Image.new("RGB", (4, 3)).save(source)
    recorder = TraceRecorder(tmp_path / "output.png.json")
    with pytest.raises(UpscaleError, match="overlaps"):
        with recorder:
            with Upscaler(cache_dir=tmp_path / "cache") as up:
                up.loaded = RepeatModel()
                up.upscale_file(source, output, report=True)
    assert not output.exists()
    assert read_trace(recorder.path)["rastermoves"]["status"] == "failed"


def test_startup_failure_cleans_context_and_file(tmp_path, monkeypatch):
    def broken_start(self):
        raise RuntimeError("cannot start thread")
    monkeypatch.setattr(threading.Thread, "start", broken_start)
    with pytest.raises(RuntimeError, match="cannot start"):
        with TraceRecorder(tmp_path / "start.json"):
            pass
    assert current_trace() is None
    assert not (tmp_path / "start.json").exists()


def test_write_and_close_errors_do_not_mask_application_exception(tmp_path):
    r = TraceRecorder(tmp_path / "io.json")
    class FailingWriter:
        def __init__(self, file):
            self.file = file
        def write(self, value):
            raise OSError("disk error")
        def flush(self):
            raise OSError("disk error")
        def close(self):
            self.file.close()
            raise OSError("close error")
    with pytest.raises(ValueError, match="original"):
        with r:
            r._file = FailingWriter(r._file)
            r.event("cannot_write")
            raise ValueError("original")
    assert current_trace() is None and not r._thread.is_alive()
