import hashlib
from pathlib import Path
import numpy as np
import pytest

from rastermoves.specs import ModelSpec, Resource


@pytest.fixture
def raw_model():
    return {
        "name": "Example", "architecture": "esrgan", "scale": 2, "author": "test-author",
        "license": "CC0-1.0", "tags": ["photo"], "inputChannels": 3, "outputChannels": 3,
        "resources": [{"platform": "pytorch", "type": "pth", "urls": ["https://example.org/model.pth"],
                       "size": 16, "sha256": "a" * 64}],
    }


@pytest.fixture
def weight_data():
    blob = b"synthetic-test-weights-not-a-real-model" * 32
    resource = Resource(format="pth", urls=("https://example.org/a.pth",), size=len(blob),
                        sha256=hashlib.sha256(blob).hexdigest())
    return blob, resource


class RepeatModel:
    scale = 2
    input_channels = 3
    output_channels = 3
    tiling = "supported"
    device = "cpu"
    precision = "fp32"
    def predict(self, image):
        return image.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
    def clear_cache(self):
        pass
    def close(self):
        pass


@pytest.fixture
def repeat_model():
    return RepeatModel()
