"""Adapter contract tests with stub runtimes; not pretrained-model validation."""
from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rastermoves.backends.onnx_backend import OnnxModel
from rastermoves.backends.spandrel_backend import SpandrelModel, normalize_state_dict
from rastermoves.errors import UpscaleError
from rastermoves.specs import ModelSpec


@pytest.mark.parametrize("wrapper", ["params_ema", "state_dict", "model_state_dict", "params", "model", "net"])
def test_normalize_checkpoint(wrapper):
    value = object()
    result = normalize_state_dict({wrapper: {"module.netG.weight": value}, "epoch": 5})
    assert result == {"weight": value}


def test_normalize_ema_priority_and_empty_rejection():
    assert normalize_state_dict({"params": {"w": 1}, "params_ema": {"w": 2}}) == {"w": 2}
    with pytest.raises(UpscaleError, match="nonempty"):
        normalize_state_dict({})


@pytest.fixture
def stub_onnx(tmp_path, monkeypatch):
    options = {"layout": "NCHW", "shape": [1, 3, None, None], "dtype": "tensor(float)",
               "providers": ["CPUExecutionProvider"], "output_shape": [1, 3, None, None]}
    captured = {}

    class Session:
        def __init__(self, data, providers):
            assert data == b"synthetic-onnx-stub"
            captured["requested_providers"] = providers
        def get_providers(self):
            return options["providers"]
        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=options["shape"], type=options["dtype"])]
        def get_outputs(self):
            return [SimpleNamespace(name="output", shape=options["output_shape"])]
        def run(self, names, feed):
            x = feed["input"]
            captured["input"] = x
            axes = (2, 3) if options["layout"] == "NCHW" else (1, 2)
            return [x.repeat(2, axes[0]).repeat(2, axes[1])]

    module = ModuleType("onnxruntime")
    module.get_available_providers = lambda: options["providers"]
    module.InferenceSession = Session
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    path = tmp_path / "m.onnx"
    path.write_bytes(b"synthetic-onnx-stub")
    return path, options, captured


@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_onnx_dynamic_layout_padding_and_bgr(stub_onnx, layout):
    path, options, captured = stub_onnx
    options["layout"] = layout
    options["shape"] = options["output_shape"] = [1, 3, None, None] if layout == "NCHW" else [1, None, None, 3]
    spec = ModelSpec("stub", "Stub", 2, options={"layout": layout, "multiple_of": 8, "channel_order": "BGR"})
    model = OnnxModel(path, spec, device="cpu")
    image = np.random.default_rng(2).random((5, 11, 3), dtype=np.float32)
    result = model.predict(image)
    np.testing.assert_array_equal(result, image.repeat(2, 0).repeat(2, 1))
    assert captured["input"].shape == ((1, 3, 8, 16) if layout == "NCHW" else (1, 8, 16, 3))
    model.close()
    assert model.session is None


def test_onnx_static_padding_and_oversized_rejection(stub_onnx):
    path, options, captured = stub_onnx
    options["shape"] = [1, 3, 16, 16]
    model = OnnxModel(path, ModelSpec("stub", "Stub", 2), device="cpu")
    result = model.predict(np.ones((7, 10, 3), np.float32))
    assert result.shape == (14, 20, 3)
    assert captured["input"].shape == (1, 3, 16, 16)
    with pytest.raises(UpscaleError, match="fixed shape"):
        model.predict(np.ones((17, 10, 3), np.float32))


def test_onnx_precision_and_missing_cuda(stub_onnx):
    path, options, _ = stub_onnx
    spec = ModelSpec("stub", "Stub", 2)
    options["dtype"] = "tensor(float16)"
    with pytest.raises(UpscaleError, match="fp16"):
        OnnxModel(path, spec, precision="fp32")
    model = OnnxModel(path, spec, precision="fp16")
    assert model.predict(np.ones((3, 3, 3), np.float32)).dtype == np.float32
    with pytest.raises(UpscaleError, match="no CUDA"):
        OnnxModel(path, spec, device="cuda", precision="fp16")


def test_onnx_bad_metadata_rejected(stub_onnx):
    path, options, _ = stub_onnx
    options["shape"] = [1, 4, None, None]
    with pytest.raises(UpscaleError, match="channel"):
        OnnxModel(path, ModelSpec("stub", "Stub", 2))


def test_spandrel_bridge_with_real_torch_and_stub_detection(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    captured = {}

    class Descriptor:
        scale = 2
        input_channels = output_channels = 3
        supports_half = True
        purpose = "SR"
        tiling = SimpleNamespace(name="SUPPORTED")
        architecture = SimpleNamespace(id="TestOnly")
        def to(self, device=None, dtype=None):
            self.device, self.dtype = torch.device(device), dtype or torch.float32
            return self
        def eval(self): return self
        def __call__(self, image):
            assert image.dtype == self.dtype and image.device == self.device
            return image.repeat_interleave(2, 2).repeat_interleave(2, 3)

    class Loader:
        def __init__(self, device): assert device == "cpu"
        def load_from_state_dict(self, state):
            captured["state"] = state
            return Descriptor()

    module = ModuleType("spandrel")
    module.ImageModelDescriptor = Descriptor
    module.ModelLoader = Loader
    monkeypatch.setitem(sys.modules, "spandrel", module)
    path = tmp_path / "tiny.pth"
    torch.save({"params_ema": {"module.weight": torch.ones(1)}}, path)
    spec = ModelSpec("stub", "Stub", 2)
    loaded = SpandrelModel(path, spec, device="cpu")
    assert list(captured["state"]) == ["weight"]
    image = np.random.default_rng(4).random((5, 7, 3), dtype=np.float32)
    np.testing.assert_array_equal(loaded.predict(image), image.repeat(2, 0).repeat(2, 1))
    with pytest.raises(UpscaleError, match="disagree"):
        SpandrelModel(path, replace(spec, scale=4), device="cpu")
    with pytest.raises(UpscaleError, match="FP16"):
        SpandrelModel(path, spec, device="cpu", precision="fp16")
    loaded.close()
