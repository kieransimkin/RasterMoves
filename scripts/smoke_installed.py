"""Run outside the checkout against an installed wheel, including packaged data."""
import argparse
from importlib import metadata, resources
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--runtimes", action="store_true")
    parser.add_argument("--trace", action="store_true", help="Also smoke-test the optional resource tracer")
    args = parser.parse_args()
    import rastermoves
    assert metadata.version("rastermoves") == rastermoves.__version__ == args.version
    assert resources.files("rastermoves").joinpath("sweep.py").is_file()
    assert resources.files("rastermoves").joinpath("tracing.py").is_file()
    assert resources.files("rastermoves").joinpath("py.typed").is_file()
    assert resources.files("rastermoves").joinpath("refinement", "diffusers_backend.py").is_file()
    subprocess.run([sys.executable, "-m", "rastermoves", "--version"], check=True)
    subprocess.run([sys.executable, "-m", "rastermoves", "upscale", "--help"], check=True, stdout=subprocess.DEVNULL)
    with tempfile.TemporaryDirectory() as temp:
        from PIL import Image
        directory = Path(temp)
        source = directory / "input.png"
        Image.new("RGB", (4, 3)).save(source)
        result = subprocess.run([sys.executable, "-m", "rastermoves", "--cache-dir", str(directory / "cache"),
                                 "upscale", str(source), "--all-models", "--dry-run", "--offline",
                                 "-o", str(directory / "outputs")], check=True, text=True, capture_output=True)
        assert json.loads(result.stdout)["model_count"] == 8
        refined = directory / "refined.png"
        command = [sys.executable, "-m", "rastermoves", "refine", str(source), "-o", str(refined),
                   "--refine-strength", "0", "--offline"]
        if args.trace or args.runtimes:
            command.append("--trace")
        subprocess.run(command, check=True, capture_output=True, text=True)
        assert Image.open(refined).size == (4, 3)
        baseline = directory / "refined.baseline.png"
        assert baseline.is_file()
        report_path = directory / "refined.png.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["status"] == "success" and not report["refinement"]["generative"]
        assert report["refinement"]["metrics"]["executed_denoising_steps"] == 0
        before = report_path.read_bytes()
        subprocess.run(command + ["--resume"], check=True, capture_output=True, text=True)
        assert before == report_path.read_bytes()
        if args.trace or args.runtimes:
            trace_file = directory / "installed.trace.json"
            subprocess.run([sys.executable, "-m", "rastermoves", "--cache-dir", str(directory / "cache"),
                            "upscale", str(source), "--all-models", "--dry-run", "--offline",
                            "-o", str(directory / "outputs"), "--trace-file", str(trace_file)],
                           check=True, capture_output=True, text=True)
            trace = json.loads(trace_file.read_text(encoding="utf-8"))
            assert trace["rastermoves"]["trace_complete"]
            assert trace["rastermoves"]["summary"]["peak_sampled_rss_bytes"] > 0
            assert any(event.get("ph") == "X" for event in trace["traceEvents"])
    if args.runtimes:
        import torch
        import spandrel
        import onnxruntime
        assert torch.ones(2).sum().item() == 2
        assert callable(spandrel.ModelLoader)
        assert "CPUExecutionProvider" in onnxruntime.get_available_providers()
    print("Installed-package smoke checks passed.")


if __name__ == "__main__":
    main()
