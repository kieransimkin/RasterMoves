import json
from pathlib import Path

from filelock import FileLock
import pytest
from PIL import Image

from rastermoves import sweep
from rastermoves.cli import main
from rastermoves.errors import UpscaleError
from rastermoves.network import atomic_json
from rastermoves.specs import ModelSpec


@pytest.fixture
def setup_sweep(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    Image.new("RGB", (7, 5), (20, 70, 150)).save(source)
    output = tmp_path / "comparison"
    state = {"created": [], "closed": [], "errors": {}, "close_errors": {}, "active": 0, "peak": 0}

    class FakeUpscaler:
        def __init__(self, model, **kwargs):
            state["created"].append(model)
            self.model = model
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])

        def upscale_file(self, source, destination, **kwargs):
            if self.model in state["errors"]:
                raise state["errors"][self.model]
            with Image.open(source) as image:
                result = image.resize((image.width * 2, image.height * 2))
                result.save(destination)
            atomic_json(destination.with_suffix(destination.suffix + ".json"), {"model": self.model})

        def close(self):
            state["active"] -= 1
            state["closed"].append(self.model)
            if self.model in state["close_errors"]:
                raise state["close_errors"][self.model]

    monkeypatch.setattr(sweep, "Upscaler", FakeUpscaler)
    models = [ModelSpec(id=i, name=i, scale=2) for i in ("c-model", "a-model", "b-model")]
    return source, output, models, state


def test_all_sorted_independent_outputs_and_cleanup(setup_sweep):
    source, output, models, state = setup_sweep
    result = sweep.run_all_models(source, output, models)
    assert state["created"] == state["closed"] == ["a-model", "b-model", "c-model"]
    assert state["active"] == 0 and state["peak"] == 1
    assert result["status"] == "completed"
    assert result["counts"]["success"] == 3
    for row in result["results"]:
        with Image.open(output / row["output"]) as image:
            assert image.size == (14, 10)
        assert row["output_sha256"] and row["report_sha256"]
    assert json.loads((output / sweep.SUMMARY).read_text()) == result


@pytest.mark.parametrize("error", [UpscaleError("no backend"), RuntimeError("bad architecture"), MemoryError("OOM")])
def test_failure_does_not_prevent_other_models(setup_sweep, error):
    source, output, models, state = setup_sweep
    state["errors"]["b-model"] = error
    result = sweep.run_all_models(source, output, models)
    assert result["counts"]["success"] == 2 and result["counts"]["failed"] == 1
    assert state["created"] == state["closed"] == ["a-model", "b-model", "c-model"]
    assert type(error).__name__ in result["results"][1]["error"]


def test_resume_reuses_successes_retries_failures(setup_sweep):
    source, output, models, state = setup_sweep
    state["errors"]["b-model"] = UpscaleError("offline missing weights")
    sweep.run_all_models(source, output, models)
    state["errors"].clear()
    state["created"].clear()
    result = sweep.run_all_models(source, output, models, resume=True)
    assert state["created"] == ["b-model"]
    assert result["counts"]["reused"] == 2 and result["counts"]["success"] == 1


@pytest.mark.parametrize("change", ["image", "sidecar", "missing", "manifest"])
def test_resume_rebuilds_changed_result(setup_sweep, change):
    source, output, models, state = setup_sweep
    sweep.run_all_models(source, output, models)
    target = output / "model-a-model.png"
    if change == "image":
        target.write_bytes(b"corrupt")
    elif change == "sidecar":
        target.with_suffix(".png.json").write_text("{}")
    elif change == "missing":
        target.unlink()
    else:
        models[1] = ModelSpec(id="a-model", name="new checkpoint metadata", scale=2)
    state["created"].clear()
    result = sweep.run_all_models(source, output, models, resume=True)
    assert state["created"] == ["a-model"] and result["counts"]["reused"] == 2


@pytest.mark.parametrize("change", ["input", "options", "version"])
def test_resume_rejects_incompatible_run(setup_sweep, monkeypatch, change):
    source, output, models, state = setup_sweep
    sweep.run_all_models(source, output, models)
    kwargs = {}
    if change == "input":
        Image.new("RGB", (7, 5), (0, 0, 0)).save(source)
    elif change == "version":
        monkeypatch.setattr(sweep, "__version__", "9.0.0")
    else:
        kwargs["image_options"] = {"width": 50}
    state["created"].clear()
    with pytest.raises(UpscaleError, match="Cannot resume"):
        sweep.run_all_models(source, output, models, resume=True, **kwargs)
    assert state["created"] == []


def test_resume_accepts_new_models(setup_sweep):
    source, output, models, state = setup_sweep
    sweep.run_all_models(source, output, models)
    state["created"].clear()
    result = sweep.run_all_models(source, output, models + [ModelSpec(id="new", name="New", scale=2)], resume=True)
    assert state["created"] == ["new"] and result["counts"]["reused"] == 3


def test_interrupt_saves_and_closes_then_resume(setup_sweep):
    source, output, models, state = setup_sweep
    state["errors"]["b-model"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        sweep.run_all_models(source, output, models)
    result = json.loads((output / sweep.SUMMARY).read_text())
    assert result["status"] == "interrupted"
    assert [r["status"] for r in result["results"]] == ["success", "interrupted", "pending"]
    assert state["closed"] == ["a-model", "b-model"]
    state["errors"].clear()
    state["created"].clear()
    sweep.run_all_models(source, output, models, resume=True)
    assert state["created"] == ["b-model", "c-model"]


def test_cleanup_failure_is_recorded_and_next_model_attempted(setup_sweep):
    source, output, models, state = setup_sweep
    state["close_errors"]["a-model"] = RuntimeError("cleanup")
    result = sweep.run_all_models(source, output, models)
    assert result["counts"]["failed"] == 1 and len(state["created"]) == 3
    assert "cleanup" in result["results"][0]["cleanup_error"]


def test_existing_outputs_need_explicit_permission(setup_sweep):
    source, output, models, state = setup_sweep
    output.mkdir()
    (output / "model-a-model.png").write_bytes(b"mine")
    with pytest.raises(UpscaleError, match="existing comparison output"):
        sweep.run_all_models(source, output, models)
    assert not state["created"]
    assert (output / "model-a-model.png").read_bytes() == b"mine"
    assert sweep.run_all_models(source, output, models, overwrite=True)["counts"]["success"] == 3
    with pytest.raises(UpscaleError, match="summary already exists"):
        sweep.run_all_models(source, output, models)


@pytest.mark.parametrize("contents", ['[]', '{}', '{invalid', '{"results": 5}'])
def test_bad_resume_summary(setup_sweep, contents):
    source, output, models, _ = setup_sweep
    output.mkdir()
    (output / sweep.SUMMARY).write_text(contents)
    with pytest.raises(UpscaleError):
        sweep.run_all_models(source, output, models, resume=True)


def test_folder_lock_and_no_accidental_input_replacement(setup_sweep):
    source, output, models, state = setup_sweep
    output.mkdir()
    with FileLock(str(output / ".rastermoves-sweep.lock")):
        with pytest.raises(UpscaleError, match="Another comparison"):
            sweep.run_all_models(source, output, models)
    source = output / "model-a-model.png"
    Image.new("RGB", (2, 2)).save(source)
    with pytest.raises(UpscaleError, match="Input overlaps"):
        sweep.run_all_models(source, output, models, overwrite=True)
    assert state["created"] == []


def test_windows_reserved_model_names_get_safe_prefix():
    assert sweep.output_name(ModelSpec(id="CON", name="CON", scale=2)) == "model-CON.png"


@pytest.mark.parametrize("options", [{"width": 0}, {"tile": -1}, {"tile": 16, "overlap": 32},
                                      {"tile_pad": -1}, {"max_output_mp": float("inf")}])
def test_invalid_processing_options_fail_before_loading(setup_sweep, options):
    source, output, models, state = setup_sweep
    with pytest.raises(UpscaleError):
        sweep.run_all_models(source, output, models, image_options=options)
    assert state["created"] == []


def test_cli_dry_run_and_named_output_requirement(tmp_path, capsys):
    source = tmp_path / "input.png"
    Image.new("RGB", (3, 3)).save(source)
    base = ["--cache-dir", str(tmp_path / "cache"), "upscale", str(source), "--all-models"]
    assert main(base) == 1
    assert "named output folder" in capsys.readouterr().err
    folder = tmp_path / "outputs"
    assert main(base + ["-o", str(folder), "--dry-run", "--offline"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model_count"] == 8 and result["dry_run"] and not folder.exists()


@pytest.mark.parametrize("flag", ["--sync-models", "--dry-run", "--resume"])
def test_all_only_flags_rejected_in_single_mode(tmp_path, capsys, flag):
    assert main(["upscale", str(tmp_path / "input.png"), flag]) == 1
    assert "require --all-models" in capsys.readouterr().err


def test_all_cannot_select_single_model():
    with pytest.raises(SystemExit) as e:
        main(["upscale", "input.png", "--all-models", "-m", "model"])
    assert e.value.code == 2


def test_sync_called_before_listing_and_offline_rejected(tmp_path, monkeypatch, capsys):
    from rastermoves import cli
    calls = []
    monkeypatch.setattr(cli, "sync_catalog", lambda root: calls.append(root))
    source = tmp_path / "input.png"
    Image.new("RGB", (2, 2)).save(source)
    args = ["--cache-dir", str(tmp_path / "cache"), "upscale", str(source), "-o", str(tmp_path / "out"),
            "--all-models", "--sync-models", "--dry-run"]
    assert main(args) == 0 and len(calls) == 1
    assert main(args + ["--offline"]) == 1 and len(calls) == 1


def test_cli_actual_pipeline_all_eight_models(tmp_path, monkeypatch, capsys, repeat_model):
    # Actual CLI -> sweep -> Upscaler -> tiler -> image/provenance outputs. Only
    # pretrained-network loading is substituted by a deterministic test network.
    from rastermoves.plugins import ModelPlugin
    def load(plugin, *args, **kwargs):
        model = type(repeat_model)()
        model.scale = plugin.spec.scale
        return model, None, None
    monkeypatch.setattr(ModelPlugin, "load", load)
    source, output = tmp_path / "input.png", tmp_path / "outputs"
    Image.new("RGBA", (5, 3), (11, 55, 99, 100)).save(source)
    args = ["--cache-dir", str(tmp_path / "cache"), "upscale", str(source), "-o", str(output),
            "--all-models", "--offline", "--tile", "0"]
    assert main(args) == 0
    report = json.loads((output / sweep.SUMMARY).read_text())
    assert report["counts"]["success"] == 8
    for row in report["results"]:
        with Image.open(output / row["output"]) as image:
            assert image.size == (5 * row["native_scale"], 3 * row["native_scale"])
            assert image.mode == "RGBA"
    assert main(args + ["--resume"]) == 0
    assert json.loads((output / sweep.SUMMARY).read_text())["counts"]["reused"] == 8


def test_cli_failures_return_nonzero_and_interrupt_returns_130(tmp_path, monkeypatch):
    from rastermoves.plugins import ModelPlugin
    source, output = tmp_path / "input.png", tmp_path / "outputs"
    Image.new("RGB", (3, 2)).save(source)
    args = ["--cache-dir", str(tmp_path / "cache"), "upscale", str(source), "-o", str(output), "--all-models"]
    def fail(*args, **kwargs):
        raise UpscaleError("no runtime")
    monkeypatch.setattr(ModelPlugin, "load", fail)
    assert main(args) == 1
    assert json.loads((output / sweep.SUMMARY).read_text())["counts"]["failed"] == 8
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(ModelPlugin, "load", interrupt)
    assert main(args + ["--overwrite"]) == 130
    assert json.loads((output / sweep.SUMMARY).read_text())["status"] == "interrupted"


def test_cli_reuses_registry_snapshot_instead_of_parsing_catalogue_per_model(tmp_path, monkeypatch):
    from rastermoves import pipeline
    from rastermoves.plugins import ModelPlugin
    source = tmp_path / "input.png"
    Image.new("RGB", (2, 2)).save(source)
    def reject_reparse(*args, **kwargs):
        raise AssertionError("Registry should not be rebuilt for each checkpoint")
    def unavailable(*args, **kwargs):
        raise UpscaleError("intentional missing runtime")
    monkeypatch.setattr(pipeline, "Registry", reject_reparse)
    monkeypatch.setattr(ModelPlugin, "load", unavailable)
    output = tmp_path / "out"
    assert main(["--cache-dir", str(tmp_path / "cache"), "upscale", str(source), "--all-models", "-o", str(output)]) == 1
    summary = json.loads((output / sweep.SUMMARY).read_text())
    assert summary["counts"]["failed"] == 8
    assert all("intentional missing runtime" in r["error"] for r in summary["results"])
