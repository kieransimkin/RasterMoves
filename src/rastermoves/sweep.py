"""Sequential, resumable comparison of one input against a registry snapshot.

The registry is passive metadata: every selected model is attempted, not presumed
compatible. A failure belongs to that model; it must not prevent later attempts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable

from filelock import FileLock, Timeout
from PIL import Image

from . import __version__
from .downloads import sha256_file
from .errors import UpscaleError
from .network import atomic_json
from .pipeline import Upscaler, _prepare, target_size
from .specs import ModelSpec
from .registry import Registry
from .tracing import current_trace, event, span, traced

SUMMARY = "summary.json"
FORMATS = {"png", "webp", "jpg", "tiff"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def output_name(spec: ModelSpec, format: str = "png") -> str:
    # IDs are validated by ModelSpec. A prefix also avoids Windows reserved names
    # (CON, NUL, COM1...) without changing the model ID used for loading.
    return f"model-{spec.id}.{format}"


@traced("comparison_validate")
def _validate(source: Path, output_dir: Path, format: str, options: dict) -> None:
    if not source.is_file():
        raise UpscaleError("--all-models requires one input image file, not a directory.")
    if output_dir.exists() and not output_dir.is_dir():
        raise UpscaleError("--all-models output must be a named directory, not a file.")
    if format not in FORMATS:
        raise UpscaleError(f"Unsupported comparison output format: {format}")
    # Validate/decode once before potentially downloading many large checkpoints.
    with Image.open(source) as original:
        color, alpha, _ = _prepare(original)
        try:
            if format == "jpg" and alpha is not None:
                raise UpscaleError("JPEG cannot preserve alpha. Select PNG/WebP/TIFF.")
            target_size(*color.size, 1, scale=options.get("scale"),
                        target_width=options.get("width"), target_height=options.get("height"),
                        long_edge=options.get("long_edge"))
        finally:
            color.close()
            if alpha is not None:
                alpha.close()
    import math
    mp = options.get("max_output_mp", 64)
    if not math.isfinite(mp) or mp <= 0:
        raise UpscaleError("max_output_mp must be finite and positive.")
    tile, overlap, pad = (options.get(k, d) for k, d in (("tile", 256), ("overlap", 32), ("tile_pad", 16)))
    if tile < 0 or overlap < 0 or pad < 0 or (tile and overlap >= tile):
        raise UpscaleError("Use tile >= 0, tile-pad >= 0, and 0 <= overlap < tile (unless tile=0).")


@traced("resume_verify")
def _verified(row: dict, destination: Path, spec_hash: str) -> bool:
    if row.get("status") not in {"success", "reused"} or row.get("spec_sha256") != spec_hash:
        return False
    sidecar = destination.with_suffix(destination.suffix + ".json")
    try:
        if row.get("refinement_artifacts"):
            from .refinement.api import paths_for
            paths = paths_for(destination)
            for key in ("baseline", "baseline_report"):
                item = row["refinement_artifacts"].get(key, {})
                if not paths[key].is_file() or sha256_file(paths[key]) != item.get("sha256"):
                    return False
        return (destination.is_file() and sidecar.is_file()
                and sha256_file(destination) == row.get("output_sha256")
                and sha256_file(sidecar) == row.get("report_sha256"))
    except OSError:
        return False


@traced("all_models")
def run_all_models(
    source: str | Path,
    output_dir: str | Path,
    specs: Iterable[ModelSpec],
    *,
    session_options: dict[str, Any] | None = None,
    image_options: dict[str, Any] | None = None,
    format: str = "png",
    registry: Registry | None = None,
    overwrite: bool = False,
    resume: bool = False,
    progress: Callable[[int, int, str, str], None] | None = None,
    refinement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attempt every model, writing images, provenance sidecars and an atomic summary.

    Resume only reuses results matching the input bytes, processing options,
    software version, model manifest, output checksum and sidecar checksum.
    It retries failed/interrupted results and accepts newly discovered models.
    Registry/checkpoint availability is not a promise of inference compatibility.
    """
    source, output_dir = Path(source).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    sessions, options = dict(session_options or {}), dict(image_options or {})
    selected = sorted(specs, key=lambda s: s.id.casefold())
    if not selected:
        raise UpscaleError("No model plugins selected.")
    if len({s.id.casefold() for s in selected}) != len(selected):
        raise UpscaleError("Duplicate model IDs in comparison.")
    if resume and overwrite:
        raise UpscaleError("Choose --resume or --overwrite, not both.")
    _validate(source, output_dir, format, options)
    if refinement:
        from .refinement.api import refinement_identity, paths_for, validate_paths
        refinement_identity(refinement)  # Fail early for invalid settings/masks.
        for spec in selected:
            validate_paths(source, paths_for(output_dir / output_name(spec, format)),
                           mask=Path(refinement["protect_mask"]) if refinement.get("protect_mask") else None)
    # Never overwrite an input, even if it happens to use a planned output name.
    planned = [output_dir / output_name(spec, format) for spec in selected]
    if any(source == p.resolve() or source == p.with_suffix(p.suffix + ".json").resolve() for p in planned):
        raise UpscaleError("Input overlaps a comparison output; use a different output directory.")
    if source == (output_dir / SUMMARY).resolve():
        raise UpscaleError("Input overlaps the comparison summary; use a different output directory.")
    trace = current_trace()
    if trace:
        trace.protect_paths([source, output_dir / SUMMARY, *planned,
                             *[p.with_suffix(p.suffix + ".json") for p in planned]])
    output_dir.mkdir(parents=True, exist_ok=True)
    # One writer per comparison folder. Do not hold a device while waiting for it.
    try:
        with FileLock(str(output_dir / ".rastermoves-sweep.lock"), timeout=0):
            return _run(source, output_dir, selected, sessions, options, format, overwrite, resume, progress, registry, refinement)
    except Timeout as e:
        raise UpscaleError(f"Another comparison is using {output_dir}.") from e


def _run(source, directory, selected, sessions, options, format, overwrite, resume, progress, registry, refinement=None):
    summary_path = directory / SUMMARY
    # Cache location/network permissions do not affect output identity. Include
    # custom plugin locations so changing installed overrides invalidates resume.
    identity_sessions = {k: str(v) if isinstance(v, Path) else v for k, v in sessions.items()
                         if k not in {"cache_dir", "offline"}}
    identity_sessions["model_dirs"] = [str(Path(p).expanduser().resolve()) for p in sessions.get("model_dirs", [])]
    config = {"software_version": __version__, "input_sha256": sha256_file(source), "format": format,
              "image_options": options, "session_options": identity_sessions}
    if refinement:
        from .refinement.api import refinement_identity
        config["refinement"] = refinement_identity(refinement)
    fingerprint = _digest(config)
    previous = {}
    if resume:
        if not summary_path.is_file():
            raise UpscaleError("--resume needs an existing summary.json in the output directory.")
        try:
            old = json.loads(summary_path.read_text(encoding="utf-8"))
            if (old["schema_version"] != 1 or old["software"] != "RasterMoves"
                    or old["fingerprint"] != fingerprint):
                raise UpscaleError("Cannot resume: input, version or processing options changed. Use a new folder or --overwrite.")
            if not isinstance(old["results"], list):
                raise ValueError("results must be a list")
            previous = {r["model_id"]: r for r in old["results"]}
        except (KeyError, TypeError, ValueError) as e:
            raise UpscaleError("Invalid comparison summary; use a new folder or --overwrite.") from e
    elif summary_path.exists() and not overwrite:
        raise UpscaleError("Comparison summary already exists. Use --resume, --overwrite, or a new folder.")

    rows = []
    for spec in selected:
        name = output_name(spec, format)
        dest = directory / name
        sidecar = dest.with_suffix(dest.suffix + ".json")
        prior = previous.get(spec.id, {})
        tracked = prior.get("output") == name
        if not overwrite and not (resume and tracked) and (dest.exists() or sidecar.exists()):
            raise UpscaleError(f"Untracked/existing comparison output: {dest}. Use a new folder or --overwrite.")
        extra_paths = []
        if refinement:
            from .refinement.api import paths_for
            extra_paths = [paths_for(dest)[k] for k in ("baseline", "baseline_report")]
            if not overwrite and not (resume and tracked) and any(p.exists() for p in extra_paths):
                raise UpscaleError(f"Existing refinement baseline for {dest}; use --resume or --overwrite.")
        for path in (dest, sidecar, *extra_paths):
            # Existing directories/symlinks are never candidates for replacement.
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise UpscaleError(f"Comparison output must be a regular file, not a directory or symlink: {path}")
        rows.append({"model_id": spec.id, "model_name": spec.name, "model_license": spec.license,
                     "model_page": spec.source_page, "native_scale": spec.scale,
                     "spec_sha256": _digest(spec.to_dict()), "output": name, "status": "pending"})
    summary = {"schema_version": 1, "software": "RasterMoves", "software_version": __version__,
               "started_at": _now(), "input": str(source), "output_directory": str(directory),
               "fingerprint": fingerprint, "config": config, "status": "running", "results": rows}

    trace = current_trace()
    if trace:
        summary["trace_file"] = str(trace.path)

    def save():
        summary["updated_at"] = _now()
        summary["counts"] = {k: sum(row["status"] == k for row in rows)
                             for k in ("success", "reused", "failed", "interrupted", "running", "pending")}
        atomic_json(summary_path, summary)

    save()
    try:
        for index, (spec, row) in enumerate(zip(selected, rows), 1):
            destination = directory / row["output"]
            prior = previous.get(spec.id, {})
            if resume and prior.get("output") == row["output"] and _verified(prior, destination, row["spec_sha256"]):
                row.update({k: prior[k] for k in ("output_sha256", "report_sha256", "elapsed_seconds") if k in prior})
                if prior.get("refinement_artifacts"):
                    row["refinement_artifacts"] = prior["refinement_artifacts"]
                row["status"] = "reused"
                event("model_reused", model_id=spec.id)
                save()
                if progress:
                    progress(index, len(rows), spec.id, "reused")
                continue
            row["status"] = "running"
            save()
            if progress:
                progress(index, len(rows), spec.id, "running")
            measured = None
            try:
                with span("model", model_id=spec.id) as measured:
                    started, up = time.perf_counter(), None
                    try:
                        if refinement:
                            from .refinement.api import WorkflowUpscaler
                            up = WorkflowUpscaler(spec.id, registry=registry, refinement=refinement, **sessions)
                            sidecar = destination.with_suffix(destination.suffix + ".json")
                            tracked = prior.get("output") == row["output"]
                            same_spec = prior.get("spec_sha256") == row["spec_sha256"]
                            intact_report = (sidecar.is_file() and
                                (prior.get("status") not in {"success", "reused"}
                                 or sha256_file(sidecar) == prior.get("report_sha256")))
                            workflow_resume = bool(resume and tracked and same_spec and intact_report)
                            up.upscale_file(source, destination,
                                            overwrite=overwrite or (resume and tracked and not workflow_resume),
                                            resume=workflow_resume, report=True, **options)
                            row["refinement_artifacts"] = up.artifacts()
                        else:
                            up = Upscaler(spec.id, registry=registry, **sessions)
                            up.upscale_file(source, destination, overwrite=overwrite or resume, report=True, **options)
                        row["output_sha256"] = sha256_file(destination)
                        row["report_sha256"] = sha256_file(destination.with_suffix(destination.suffix + ".json"))
                        row["status"] = "success"
                    except (KeyboardInterrupt, SystemExit):
                        row["status"] = "interrupted"
                        raise
                    except Exception as e:
                        # Plugin implementations may raise non-UpscaleError exceptions.
                        # Isolate them here; do not swallow SystemExit/KeyboardInterrupt.
                        row.update(status="failed", error=f"{type(e).__name__}: {e}")
                    finally:
                        if up is not None:
                            try:
                                up.close()
                            except Exception as e:
                                row["cleanup_error"] = f"{type(e).__name__}: {e}"
                                if row["status"] != "interrupted":
                                    row["status"] = "failed"
                            up = None
                        gc.collect()
                        row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                        if measured is not None:
                            measured.status = row["status"]
            finally:
                if measured is not None:
                    row["performance"] = measured.summary
                save()
            if progress:
                progress(index, len(rows), spec.id, row["status"])
    except BaseException:
        summary["status"] = "interrupted"
        save()
        raise
    summary["status"] = "completed_with_errors" if any(r["status"] == "failed" for r in rows) else "completed"
    summary["finished_at"] = _now()
    save()
    return summary
