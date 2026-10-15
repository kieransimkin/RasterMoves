from pathlib import Path

from rastermoves.gui import comparison_entries, unique_output_dir


def test_comparison_entries_include_original_baseline_and_refined(tmp_path):
    source = tmp_path / "input.png"
    source.write_bytes(b"original")
    output = tmp_path / "comparison"
    output.mkdir()
    (output / "model-demo.baseline.png").write_bytes(b"baseline")
    (output / "model-demo.png").write_bytes(b"final")
    summary = {
        "results": [{
            "model_id": "demo",
            "status": "success",
            "output": "model-demo.png",
            "refinement_artifacts": {"baseline": {"file": "model-demo.baseline.png"}},
        }]
    }
    assert comparison_entries(summary, output, source) == [
        ("Original", source),
        ("demo — upscale baseline", output / "model-demo.baseline.png"),
        ("demo — post-diffusion", output / "model-demo.png"),
    ]


def test_unique_output_dir_never_reuses_existing_folder(tmp_path):
    source = tmp_path / "photo.png"
    source.write_bytes(b"x")
    first = tmp_path / "photo-rastermoves"
    first.mkdir()
    second = tmp_path / "photo-rastermoves-2"
    second.mkdir()
    assert unique_output_dir(source) == tmp_path / "photo-rastermoves-3"
