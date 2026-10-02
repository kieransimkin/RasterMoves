"""Offline refinement tests: no pretrained models or GPU are used."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from rastermoves.cli import main, parser
from rastermoves.errors import UpscaleError
from rastermoves.refinement import Refiner, RefineOptions
from rastermoves.refinement.api import paths_for, run_workflow
from rastermoves.refinement.specs import BUILTINS, Component, get_spec
from rastermoves.refinement.tiles import positions, protection_mask, refine_tiles, tile_seed


class FakeBackend:
    constructions = 0
    calls = []
    closed = 0
    device = "synthetic-cpu"
    precision = "fp32"
    inventory = []

    def __init__(self, spec=None, **kwargs):
        type(self).constructions += 1

    def predict(self, image, *, options, seed):
        type(self).calls.append((image.size, seed))
        pixels = np.asarray(image).astype(np.int16)
        return Image.fromarray(np.clip(pixels + 32, 0, 255).astype(np.uint8)), {"executed_denoising_steps": int(options.steps * options.strength)}

    def close(self):
        type(self).closed += 1


@pytest.fixture
def fake(monkeypatch):
    from rastermoves.refinement import diffusers_backend
    FakeBackend.constructions, FakeBackend.calls, FakeBackend.closed = 0, [], 0
    monkeypatch.setattr(diffusers_backend, "DiffusersTileBackend", FakeBackend)
    return FakeBackend


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "input.png"
    rng = np.random.default_rng(4)
    Image.fromarray(rng.integers(0, 200, (79, 91, 3), dtype=np.uint8)).save(path)
    return path


@pytest.mark.parametrize("field,value", [
    ("strength", -0.1), ("strength", 1.1), ("strength", float("nan")), ("blend", 2),
    ("steps", 0), ("steps", 1001), ("steps", True), ("steps", 2.3), ("seed", -1),
    ("seed", 2**63), ("seed", 1.2), ("seed", True), ("tile", 60), ("tile", 257),
    ("overlap", -1), ("overlap", 512), ("guidance", float("inf")), ("guidance", -1),
    ("control_scale", -1), ("mask_feather", -1), ("max_output_mp", 0), ("prompt", 3),
])
def test_option_validation(field, value):
    with pytest.raises(UpscaleError):
        replace(RefineOptions(), **{field: value}).validate(BUILTINS["sd15-tile"])


def test_too_few_denoising_steps():
    with pytest.raises(UpscaleError, match="at least one"):
        Refiner(options=RefineOptions(strength=0.01, steps=10))


@pytest.mark.parametrize("kwargs", [{"device": "bad"}, {"precision": "int8"}, {"scheduler": "fake"},
                                     {"device": "cpu", "offload": "model"}, {"device": "cpu", "precision": "fp16"}])
def test_runtime_validation(kwargs):
    with pytest.raises(UpscaleError):
        Refiner(**kwargs)


@pytest.mark.parametrize("length,tile,overlap", [(1, 64, 0), (64, 64, 32), (65, 64, 63),
                                                (201, 64, 17), (200, 96, 0), (209, 128, 64)])
def test_tile_coverage_identity(length, tile, overlap):
    a = np.random.default_rng(length).integers(0, 256, (length, length + 3, 3), dtype=np.uint8)
    calls = []
    def predict(image, **kwargs):
        calls.append(image.size)
        return image.copy(), {"executed_denoising_steps": 1}
    image = Image.fromarray(a)
    options = RefineOptions(tile=tile, overlap=overlap)
    output, stats = refine_tiles(image, predict, options, tile=tile)
    np.testing.assert_array_equal(output, a)
    assert all(w % 8 == h % 8 == 0 and w >= 64 and h >= 64 for w, h in calls)
    assert stats["executed_denoising_steps"] == len(calls)
    assert positions(length, tile, overlap)[0] == 0


def test_tile_inputs_frozen_and_seed_independent():
    original = Image.new("RGB", (150, 131), (80, 90, 100))
    seen = []
    def predict(image, *, options, seed):
        assert np.all(np.asarray(image) == (80, 90, 100))
        seen.append(seed)
        return Image.new("RGB", image.size, (200, 190, 180)), {"executed_denoising_steps": 1}
    output, _ = refine_tiles(original, predict, RefineOptions(tile=96, overlap=32), tile=96)
    assert len(seen) == len(set(seen))
    assert tile_seed(0, 0, 0) == tile_seed(0, 0, 0)
    assert tile_seed(1, 0, 0) != tile_seed(0, 0, 0)
    assert np.all(np.asarray(output) == (200, 190, 180))


@pytest.mark.parametrize("kind", ["size", "mode", "not-image"])
def test_invalid_backend_output(kind):
    def predict(image, **kwargs):
        return ({"size": Image.new("RGB", (1, 2)), "mode": image.convert("L"), "not-image": None}[kind], {})
    with pytest.raises(UpscaleError, match="geometry/mode"):
        refine_tiles(Image.new("RGB", (70, 81)), predict, RefineOptions(), tile=512)


def test_white_protection_survives_outward_feather():
    a = np.zeros((80, 70), dtype=np.uint8)
    a[20:40, 30:50] = 255
    protected = protection_mask(Image.fromarray(a), (70, 80), 5)
    assert np.all(protected[20:40, 30:50] == 1)
    assert np.any(protected[:20] > 0)


@pytest.mark.parametrize("mode", ["RGBA", "I", "F", "LA"])
def test_reject_ambiguous_masks(mode):
    with pytest.raises(UpscaleError):
        protection_mask(Image.new(mode, (10, 10)), (10, 10), 0)


def test_mask_size_rejected():
    with pytest.raises(UpscaleError, match="dimensions"):
        protection_mask(Image.new("L", (12, 13)), (13, 12), 0)


def test_protected_pixels_alpha_and_mode(fake):
    src = Image.new("RGBA", (85, 71), (64, 96, 128, 180))
    mask_array = np.zeros((71, 85), np.uint8)
    mask_array[:, :24] = 255
    with Refiner(options=RefineOptions(tile=64, overlap=16, mask_feather=3)) as refiner:
        out = refiner.refine_image(src, protect_mask=Image.fromarray(mask_array))
        a = np.asarray(out)
        assert out.mode == "RGBA" and out.size == src.size
        assert np.all(a[:, :24] == np.asarray(src)[:, :24])
        assert np.all(a[:, :, 3] == 180)
        assert np.all(a[:, 60:, :3] == (96, 128, 160))
        assert refiner.last_report["metrics"]["executed_denoising_steps"] > 0
    assert fake.closed == 1


@pytest.mark.parametrize("reason", ["zero-strength", "zero-blend", "protected", "transparent"])
def test_noop_never_loads_runtime(fake, reason):
    options = RefineOptions(strength=0) if reason == "zero-strength" else (
        RefineOptions(blend=0) if reason == "zero-blend" else RefineOptions())
    src = Image.new("RGBA", (81, 93), (50, 60, 70, 0 if reason == "transparent" else 255))
    mask = Image.new("L", src.size, 255) if reason == "protected" else None
    with Refiner(options=options) as refiner:
        out = refiner.refine_image(src, protect_mask=mask)
        np.testing.assert_array_equal(src, out)
        assert not refiner.last_report["generative"]
    assert fake.constructions == 0


def test_global_colour_match_and_blend(fake):
    with Refiner(options=RefineOptions(color_match=True)) as refiner:
        src = Image.new("RGB", (30, 31), (32, 48, 64))
        np.testing.assert_array_equal(src, refiner.refine_image(src))
    with Refiner(options=RefineOptions(blend=0.5)) as refiner:
        assert np.all(np.asarray(refiner.refine_image(src)) == (48, 64, 80))


def test_workflow_roundtrip_and_resume(source, tmp_path, fake):
    dest = tmp_path / "out.png"
    with Refiner() as refiner:
        refiner.refine_file(source, dest)
        initial_calls = len(fake.calls)
        initial_report = paths_for(dest)["report"].read_bytes()
        refiner.refine_file(source, dest, resume=True)
        assert refiner.reused
        assert len(fake.calls) == initial_calls
        assert paths_for(dest)["report"].read_bytes() == initial_report
    paths = paths_for(dest)
    assert all(p.exists() for p in paths.values())
    report = json.loads(paths["report"].read_text())
    assert report["status"] == "success"
    assert report["refinement"]["kind"] == "same-size-generative-refinement"
    assert report["artifacts"]["baseline"]["sha256"] == hashlib.sha256(paths["baseline"].read_bytes()).hexdigest()
    with Image.open(source) as a, Image.open(paths["baseline"]) as b:
        np.testing.assert_array_equal(a, b)


def test_resume_changed_settings_rejected(source, tmp_path, fake):
    dest = tmp_path / "out.png"
    with Refiner() as r:
        r.refine_file(source, dest)
    with Refiner(options=RefineOptions(seed=92)) as r, pytest.raises(UpscaleError, match="Cannot resume"):
        r.refine_file(source, dest, resume=True)


def test_resume_changed_mask_rejected(source, tmp_path, fake):
    mask = tmp_path / "mask.png"
    Image.new("L", (91, 79), 0).save(mask)
    dest = tmp_path / "out.png"
    with Refiner() as r:
        r.refine_file(source, dest, protect_mask=mask)
        Image.new("L", (91, 79), 200).save(mask)
        with pytest.raises(UpscaleError, match="Cannot resume"):
            r.refine_file(source, dest, resume=True, protect_mask=mask)


def test_tampered_output_rerun_keeps_baseline(source, tmp_path, fake):
    dest = tmp_path / "out.png"
    with Refiner() as r:
        r.refine_file(source, dest)
        baseline = paths_for(dest)["baseline"]
        mtime = baseline.stat().st_mtime_ns
        dest.write_bytes(b"corrupted")
        r.refine_file(source, dest, resume=True)
        assert not r.reused
        assert baseline.stat().st_mtime_ns == mtime
        with Image.open(dest) as image:
            assert image.size == (91, 79)


def test_failure_preserves_baseline_and_resume(source, tmp_path, monkeypatch, fake):
    dest = tmp_path / "out.png"
    orig = fake.predict
    monkeypatch.setattr(fake, "predict", lambda *_a, **_k: (_ for _ in ()).throw(UpscaleError("synthetic failure")))
    with Refiner() as r, pytest.raises(UpscaleError):
        r.refine_file(source, dest)
    paths = paths_for(dest)
    assert not dest.exists() and paths["baseline"].exists()
    assert json.loads(paths["report"].read_text())["status"] == "failed"
    mtime = paths["baseline"].stat().st_mtime_ns
    monkeypatch.setattr(fake, "predict", orig)
    with Refiner() as r:
        r.refine_file(source, dest, resume=True)
    assert dest.exists() and paths["baseline"].stat().st_mtime_ns == mtime


def test_interrupt_report_and_no_final(source, tmp_path, fake, monkeypatch):
    monkeypatch.setattr(fake, "predict", lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt()))
    dest = tmp_path / "out.png"
    with Refiner() as r, pytest.raises(KeyboardInterrupt):
        r.refine_file(source, dest)
    assert json.loads(paths_for(dest)["report"].read_text())["status"] == "interrupted"
    assert not dest.exists()


@pytest.mark.parametrize("which", ["output", "report", "baseline", "baseline_report"])
def test_existing_files_not_clobbered(source, tmp_path, fake, which):
    dest = tmp_path / "out.png"
    paths_for(dest)[which].write_bytes(b"precious")
    with Refiner() as r, pytest.raises(UpscaleError, match="already exists"):
        r.refine_file(source, dest)
    assert paths_for(dest)[which].read_bytes() == b"precious"
    assert fake.constructions == 0


def test_input_output_overlap(source, fake):
    with Refiner() as r, pytest.raises(UpscaleError, match="overlaps"):
        r.refine_file(source, source, overwrite=True)


def test_mask_output_overlap(source, tmp_path, fake):
    dest = tmp_path / "out.png"
    Image.new("L", (91, 79), 0).save(dest)
    with Refiner() as r, pytest.raises(UpscaleError, match="overlaps"):
        r.refine_file(source, dest, protect_mask=dest, overwrite=True)


def test_no_jpeg(source, tmp_path, fake):
    with Refiner() as r, pytest.raises(UpscaleError, match="lossless"):
        r.refine_file(source, tmp_path / "out.jpg")


def test_symlink_output_rejected(source, tmp_path, fake):
    dest = tmp_path / "out.png"
    try:
        dest.symlink_to(source)
    except OSError:
        pytest.skip("Symlink privilege unavailable")
    with Refiner() as r, pytest.raises(UpscaleError):
        r.refine_file(source, dest, overwrite=True)


def test_output_limit_before_loading(fake):
    with Refiner(options=RefineOptions(max_output_mp=0.001)) as r, pytest.raises(UpscaleError, match="exceeds"):
        r.refine_image(Image.new("RGB", (80, 90)))
    assert fake.constructions == 0


def test_refine_cli_trace(source, tmp_path, fake):
    dest = tmp_path / "out.png"
    assert main(["refine", str(source), "-o", str(dest), "--trace"]) == 0
    trace = json.loads(Path(str(dest) + ".trace.json").read_text())
    names = {e.get("name") for e in trace["traceEvents"]}
    assert {"refine_image", "refine_tile", "refine_composite", "workflow_baseline"} <= names
    assert trace["rastermoves"]["trace_complete"]


def test_dryrun_has_pinned_components_no_files(source, tmp_path, fake, capsys):
    dest = tmp_path / "fresh" / "out.png"
    assert main(["refine", str(source), "-o", str(dest), "--dry-run", "--offline"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["refinement"]["spec"]["components"]) == 2
    assert not dest.parent.exists()
    assert fake.constructions == 0


def test_listing_no_runtime(capsys):
    assert main(["refiners", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["id"] for r in rows} == {"sd15-tile", "sdxl-tile"}


@pytest.mark.parametrize("flag", [["--seed", "1"], ["--refine-tile", "128"], ["--refine-color-match"], ["--refine-strength", "0"]])
def test_refine_flags_require_optin(source, flag):
    assert main(["upscale", str(source), *flag]) == 1


def test_unknown_plugin_no_auto_execution():
    with pytest.raises(UpscaleError, match="Unknown refiner"):
        get_spec("not-installed")


@pytest.mark.parametrize("revision", ["main", "v1", "abc123", "A" * 40, "../bad"])
def test_immutable_revision_required(revision):
    with pytest.raises(UpscaleError, match="immutable"):
        Component("base", "owner/repo", revision, "unknown")


def test_import_stays_lightweight():
    code = "import sys; import rastermoves.refinement; assert not any(n in sys.modules for n in ('torch','diffusers','transformers','accelerate'))"
    subprocess.run([sys.executable, "-c", code], check=True, env={**__import__('os').environ, "PYTHONPATH": "src"})


def test_refine_parser_defaults():
    a = parser().parse_args(["refine", "in.png"])
    assert a.refiner == "sd15-tile"
    a = parser().parse_args(["upscale", "in.png"])
    assert a.refiner is None


def test_recursive_batch_names_and_resume(source, tmp_path, fake):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "nested").mkdir()
    Image.open(source).save(inputs / "nested" / "one.png")
    Image.open(source).save(inputs / "one.webp")
    out = tmp_path / "batch"
    args = ["refine", str(inputs), "-o", str(out), "--recursive"]
    assert main(args) == 0
    assert (out / "nested" / "one.png_refined.png").is_file()
    assert (out / "one.webp_refined.png").is_file()
    calls = len(fake.calls)
    assert main(args + ["--resume"]) == 0
    assert len(fake.calls) == calls
