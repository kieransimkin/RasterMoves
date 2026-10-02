"""Optional real-model integration tests. Never imply these ran when skipped."""
import os
from PIL import Image
import pytest
from rastermoves.pipeline import Upscaler

pytestmark = [pytest.mark.live, pytest.mark.skipif(os.environ.get("RASTERMOVES_LIVE") != "1",
                                                  reason="Set RASTERMOVES_LIVE=1 for real model downloads/inference")]


@pytest.mark.parametrize("model,backend", [
    ("4x-realesr-animevideo-v3", "spandrel"),
    ("4x-SPANkendata", "onnx"),
])
def test_real_model(tmp_path, model, backend):
    with Upscaler(model,backend=backend,device="cpu",cache_dir=tmp_path / "cache",strict_checksums=True) as up:
        result=up.upscale_image(Image.new("RGB",(32,32),(80,120,160)),tile=0)
        assert result.size==(128,128)
        assert up.last_report["publisher_verified"]
        assert up.last_report["native_scale"]==4
