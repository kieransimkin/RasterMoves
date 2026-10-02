"""Opt-in, bounded-memory Chrome JSON resource traces.

The recorder measures this process (including all native/Python threads), not
children or GPU counters. Imports psutil only when recording is requested.
Events stream to disk; the JSON document is completed when the context exits.
"""
from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
import importlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import threading
import time
from typing import Any

from . import __version__
from .errors import UpscaleError

log = logging.getLogger(__name__)
_CURRENT: ContextVar[TraceRecorder | None] = ContextVar("rastermoves_trace", default=None)
DEFAULT_INTERVAL = 0.25
MIN_INTERVAL = 0.05


def current_trace() -> TraceRecorder | None:
    return _CURRENT.get()


def span(name: str, **attributes):
    """A no-op when tracing is disabled; otherwise an inclusive timed stage."""
    recorder = _CURRENT.get()
    return recorder.span(name, **attributes) if recorder else nullcontext(None)


def traced(name: str):
    """Instrument a function without changing its public signature/arguments."""
    def decorate(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            recorder = _CURRENT.get()
            if recorder is None:
                return function(*args, **kwargs)
            with recorder.span(name):
                return function(*args, **kwargs)
        return wrapper
    return decorate


def event(name: str, **attributes) -> None:
    recorder = _CURRENT.get()
    if recorder:
        recorder.event(name, **attributes)


@dataclass
class _Peaks:
    rss: int | None = None
    vms: int | None = None
    threads: int | None = None

    def add(self, sample: dict) -> None:
        for field_name, key in (("rss", "rss_bytes"), ("vms", "vms_bytes"), ("threads", "threads")):
            value = sample.get(key)
            if value is not None:
                old = getattr(self, field_name)
                setattr(self, field_name, value if old is None else max(old, value))

    def as_dict(self) -> dict:
        return {"peak_sampled_rss_bytes": self.rss, "peak_sampled_vms_bytes": self.vms,
                "peak_sampled_threads": self.threads}


@dataclass
class _Span:
    recorder: TraceRecorder
    name: str
    attributes: dict
    status: str = "completed"
    summary: dict = field(default_factory=dict)
    peaks: _Peaks = field(default_factory=_Peaks)

    def __enter__(self):
        r = self.recorder
        with r._lock:
            self.tid = threading.get_native_id()
            stack = r._active.setdefault(self.tid, [])
            inherited = stack[-1].attributes.get("model_id") if stack else None
            if inherited is not None:
                self.attributes.setdefault("model_id", inherited)
            sample = r._sample(boundary=True)
            self.peaks.add(sample)
            self.started, self.cpu_started = time.perf_counter(), time.process_time()
            stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        r = self.recorder
        with r._lock:
            r._sample(boundary=True)
            ended, cpu_ended = time.perf_counter(), time.process_time()
            stack = r._active[self.tid]
            stack.remove(self)
            if not stack:
                del r._active[self.tid]
            if exc_type is not None:
                self.status = "interrupted" if issubclass(exc_type, (KeyboardInterrupt, SystemExit)) else "failed"
                # Do not record arbitrary exception text: it can contain URLs or tokens.
                self.attributes["error_type"] = exc_type.__name__
            wall, cpu = max(0.0, ended - self.started), max(0.0, cpu_ended - self.cpu_started)
            self.summary = {"wall_seconds": wall, "process_cpu_seconds": cpu,
                            "mean_process_cpu_percent": 100 * cpu / wall if wall else 0.0,
                            "status": self.status, "resource_samples_complete": r._fault is None, **self.peaks.as_dict()}
            r._write({"name": self.name, "cat": "rastermoves.stages", "ph": "X",
                      "pid": r.pid, "tid": self.tid, "ts": (self.started - r._start_wall) * 1e6,
                      "dur": wall * 1e6, "args": {**self.attributes, **self.summary}})
            key = (self.attributes.get("model_id"), self.name)
            aggregate = r._stages.setdefault(key, {"model_id": key[0], "stage": self.name, "calls": 0,
                                                   "wall_seconds": 0.0, "process_cpu_seconds": 0.0,
                                                   "peak_sampled_rss_bytes": None})
            aggregate["calls"] += 1
            aggregate["wall_seconds"] += wall
            aggregate["process_cpu_seconds"] += cpu
            if self.peaks.rss is not None:
                aggregate["peak_sampled_rss_bytes"] = max(aggregate["peak_sampled_rss_bytes"] or 0, self.peaks.rss)
        return False


class TraceRecorder:
    """One recorder per synchronous execution context, with one sampler thread.

    ``path`` must end in .json. Existing files/symlinks are never overwritten.
    With ``auto_suffix``, choose trace.1.json, trace.2.json, ... if needed.
    Completed traces also contain a ``rastermoves`` metadata/summary object.
    """
    def __init__(self, path: str | Path, *, interval: float = DEFAULT_INTERVAL,
                 auto_suffix: bool = False, metadata: dict[str, Any] | None = None):
        if not math.isfinite(interval) or not MIN_INTERVAL <= interval <= 60.0:
            raise UpscaleError(f"--trace-interval must be finite and between {MIN_INTERVAL} and 60 seconds.")
        path = Path(path).expanduser().absolute()
        if path.suffix.lower() != ".json":
            raise UpscaleError("--trace-file must have a .json extension.")
        self.path, self.interval, self.auto_suffix = path, interval, auto_suffix
        self.metadata = dict(metadata or {})
        self.pid = os.getpid()
        self.exit_code = None
        self.status = "completed"
        self.summary: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._active: dict[int, list[_Span]] = {}
        self._stages: dict[tuple, dict] = {}
        self._peaks = _Peaks()
        self._thread = None
        self._file = None
        self._token = None
        self._first = True
        self._fault: str | None = None
        self._sample_count = 0
        self._periodic_count = 0
        self._last_cpu_sample = None
        self._used = False

    def protect_paths(self, paths) -> None:
        """Reject a trace destination that would be used as an image/report."""
        resolved = self.path.resolve()
        if any(Path(p).expanduser().resolve() == resolved for p in paths if p is not None):
            raise UpscaleError("Trace path overlaps an input, output, weights, or summary file; choose another --trace-file.")

    def __enter__(self):
        if self._used or _CURRENT.get() is not None:
            raise UpscaleError("Trace recorders cannot be nested or reused.")
        self._used = True
        try:
            self._psutil = importlib.import_module("psutil")
        except ImportError as e:
            raise UpscaleError("Resource tracing needs psutil. Install: python -m pip install 'rastermoves[trace]' "
                               "(from a source checkout: python -m pip install -e '.[trace]').") from e
        self._process = self._psutil.Process(self.pid)
        self._start_wall, self._start_cpu = time.perf_counter(), time.process_time()
        self._started_at = datetime.now(timezone.utc).isoformat()
        # Fail before running a model if process metrics cannot be read at all.
        try:
            self._process.memory_info()
        except Exception as e:
            raise UpscaleError(f"Cannot read process memory for tracing ({type(e).__name__}).") from e
        self.path.parent.mkdir(parents=True, exist_ok=True)
        original = self.path
        for suffix in range(10000):
            if suffix:
                self.path = original.with_name(f"{original.stem}.{suffix}{original.suffix}")
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                break
            except FileExistsError as e:
                if not self.auto_suffix:
                    raise UpscaleError(f"Trace file already exists: {self.path}. Choose a new --trace-file.") from e
        else:
            raise UpscaleError("Too many existing trace files; choose a different path.")
        try:
            self._file = os.fdopen(fd, "w", encoding="utf-8", buffering=1)
            self._file.write('{"traceEvents":[\n')
            self._write({"name": "process_name", "ph": "M", "pid": self.pid, "tid": 0,
                         "args": {"name": "RasterMoves"}})
            self._write({"name": "thread_name", "ph": "M", "pid": self.pid,
                         "tid": threading.get_native_id(), "args": {"name": "Pipeline"}})
            self._sample()
            if self._fault:
                raise UpscaleError(self._fault)
            self._token = _CURRENT.set(self)
            self._thread = threading.Thread(target=self._monitor, name="rastermoves-resource-trace", daemon=True)
            self._thread.start()
        except BaseException:
            if self._token is not None:
                _CURRENT.reset(self._token)
                self._token = None
            if self._file:
                self._file.close()
            # No processing took place; don't leave an invalid startup trace.
            self.path.unlink(missing_ok=True)
            raise
        return self

    def span(self, name: str, **attributes) -> _Span:
        return _Span(self, name, attributes)

    def event(self, name: str, **attributes) -> None:
        with self._lock:
            self._write({"name": name, "ph": "i", "s": "t", "cat": "rastermoves.events",
                         "pid": self.pid, "tid": threading.get_native_id(),
                         "ts": (time.perf_counter() - self._start_wall) * 1e6, "args": attributes})

    def _fail(self, operation: str, exception: Exception) -> None:
        if self._fault is None:
            self._fault = f"Resource trace {operation} failed ({type(exception).__name__})."
            log.warning("%s Partial trace: %s", self._fault, self.path)
        self._stop.set()

    def _write(self, record: dict) -> None:
        if self._file is None or self._fault:
            return
        try:
            text = json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            self._file.write(("" if self._first else ",") + text + "\n")
            self._first = False
        except Exception as e:
            self._fail("write", e)

    def _sample(self, *, boundary: bool = False) -> dict:
        # Caller holds the lock, or the sampling thread has not started yet.
        if self._fault:
            return {}
        try:
            with self._process.oneshot():
                memory = self._process.memory_info()
                metrics = {"rss_bytes": int(memory.rss), "vms_bytes": int(memory.vms),
                           "threads": int(self._process.num_threads())}
            wall, cpu = time.perf_counter(), time.process_time()
            self._peaks.add(metrics)
            for stack in self._active.values():
                for item in stack:
                    item.peaks.add(metrics)
            counters = {f"process.{key}": value for key, value in metrics.items()}
            if not boundary:
                virtual = self._psutil.virtual_memory()
                counters.update({"system.memory_available_bytes": int(virtual.available),
                                 "system.memory_percent": float(virtual.percent),
                                 "process.cpu_seconds": max(0.0, cpu - self._start_cpu)})
                if self._last_cpu_sample is not None:
                    previous_wall, previous_cpu = self._last_cpu_sample
                    elapsed = wall - previous_wall
                    if elapsed > 0:
                        counters["process.cpu_percent"] = max(0.0, cpu - previous_cpu) / elapsed * 100
                self._last_cpu_sample = (wall, cpu)
                self._periodic_count += 1
            self._sample_count += 1
            for name, value in counters.items():
                self._write({"name": name, "cat": "rastermoves.resources", "ph": "C",
                             "pid": self.pid, "tid": 0, "ts": (wall - self._start_wall) * 1e6,
                             "args": {"value": value}})
            return metrics
        except Exception as e:
            self._fail("sampling", e)
            return {}

    def _monitor(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                self._sample()

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread:
            self._thread.join()
        try:
            with self._lock:
                self._sample()
                ended, cpu_ended = time.perf_counter(), time.process_time()
                if exc_type:
                    self.status = "interrupted" if issubclass(exc_type, (KeyboardInterrupt, SystemExit)) else "failed"
                if self._fault and exc_type is None:
                    self.status = "trace_failed"
                    if self.exit_code in (None, 0):
                        self.exit_code = 1
                wall, cpu = max(0.0, ended - self._start_wall), max(0.0, cpu_ended - self._start_cpu)
                self.summary = {"wall_seconds": wall, "process_cpu_seconds": cpu,
                                "mean_process_cpu_percent": 100 * cpu / wall if wall else 0.0,
                                "sample_count": self._sample_count,
                                "periodic_sample_count": self._periodic_count, **self._peaks.as_dict()}
                details = {"schema_version": 1, "software": "RasterMoves", "software_version": __version__,
                           "started_at": self._started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
                           "pid": self.pid, "platform": platform.system(), "python_version": platform.python_version(),
                           "psutil_version": self._psutil.__version__, "logical_cpu_count": os.cpu_count(),
                           "interval_seconds": self.interval, "status": self.status, "exit_code": self.exit_code,
                           "error_type": exc_type.__name__ if exc_type else None,
                           "trace_complete": self._fault is None, "trace_error": self._fault,
                           "scope": "current process, all threads; excludes child processes and GPU counters",
                           "cpu_percent_basis": "100% = one logical CPU; not normalized to machine capacity",
                           "memory_peak_basis": "periodic and stage-boundary samples; not an allocation high-water mark",
                           "timing_basis": "host wall time (perf_counter); inclusive nested stages; no added GPU synchronization",
                           "metadata": self.metadata, "summary": self.summary,
                           "stages": list(self._stages.values())}
                try:
                    self._file.write('],"displayTimeUnit":"ms","rastermoves":' +
                                     json.dumps(details, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "}\n")
                    self._file.flush()
                except Exception as e:
                    self._fail("finalization", e)
        finally:
            try:
                if self._file:
                    try:
                        self._file.close()
                    except Exception as e:
                        self._fail("close", e)
            finally:
                if self._token is not None:
                    _CURRENT.reset(self._token)
                    self._token = None
        if self._fault and exc_type is None:
            raise UpscaleError(self._fault + f" See {self.path} (may be incomplete).")
        return False
