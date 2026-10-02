from __future__ import annotations

import gc
from pathlib import Path
import numpy as np

from ..errors import BackendOOM, UnsupportedModelError, UpscaleError


class OnnxModel:
    """Single-file, single-image ONNX models; normalized float NCHW/NHWC tensors."""
    def __init__(self, path: Path, spec, *, device="auto", precision="fp32", **_):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise UpscaleError("Install the onnx extra (CPU), or onnx-gpu for CUDA. Do not install both runtime packages.") from e
        available = ort.get_available_providers()
        if device == "auto":
            device = "cuda" if "CUDAExecutionProvider" in available else "cpu"
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda" or device.startswith("cuda:"):
            if "CUDAExecutionProvider" not in available:
                raise UpscaleError("The installed ONNX Runtime has no CUDA provider. Install onnxruntime-gpu or use cpu.")
            index = int(device.split(":")[1]) if ":" in device else 0
            providers = [("CUDAExecutionProvider", {"device_id": index}), "CPUExecutionProvider"]
        else:
            raise UpscaleError("The ONNX backend currently accepts auto, cpu, cuda, or cuda:N.")
        try:
            # Bytes disable external-data path resolution; multi-file ONNX needs a custom backend.
            self.session = ort.InferenceSession(path.read_bytes(), providers=providers)
        except Exception as e:
            raise UnsupportedModelError(f"ONNX load failed ({type(e).__name__}). Only self-contained ONNX files are supported.") from e
        active = self.session.get_providers()
        if device.startswith("cuda") and "CUDAExecutionProvider" not in active:
            raise UpscaleError("ONNX CUDA initialization failed and fell back to CPU; check CUDA/cuDNN libraries.")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or len(inputs[0].shape) != 4 or len(outputs[0].shape) != 4:
            raise UnsupportedModelError("ONNX model must expose one rank-four image input and one rank-four image output.")
        self.input, self.output = inputs[0], outputs[0]
        types = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
        if self.input.type not in types:
            raise UnsupportedModelError("Only normalized float32/float16 ONNX image inputs are supported.")
        self.dtype = types[self.input.type]
        actual = "fp16" if self.dtype == np.float16 else "fp32"
        if precision != actual:
            raise UpscaleError(f"ONNX precision is fixed by the export ({actual}); select --precision {actual}.")
        self.layout = spec.options.get("layout", "NCHW")
        if self.layout not in {"NCHW", "NHWC"}:
            raise UnsupportedModelError("ONNX layout must be NCHW or NHWC.")
        self.input_channels, self.output_channels, self.scale = spec.input_channels, spec.output_channels, spec.scale
        if self.input_channels not in (1, 3) or self.output_channels not in (1, 3):
            raise UnsupportedModelError("Only RGB/grayscale ONNX models are supported.")
        channel_axis = 1 if self.layout == "NCHW" else 3
        for info, expected in ((self.input, self.input_channels), (self.output, self.output_channels)):
            if isinstance(info.shape[0], int) and info.shape[0] != 1:
                raise UnsupportedModelError("ONNX batch size must be one or dynamic.")
            if isinstance(info.shape[channel_axis], int) and info.shape[channel_axis] != expected:
                raise UnsupportedModelError("ONNX channel shape disagrees with the model plugin.")
        dims = self.input.shape[2:4] if self.layout == "NCHW" else self.input.shape[1:3]
        self.fixed = tuple(v if isinstance(v, int) and v > 0 else None for v in dims)
        self.multiple = int(spec.options.get("multiple_of", 1))
        self.minimum = int(spec.options.get("minimum_size", 1))
        self.square = bool(spec.options.get("square", False))
        self.bgr = spec.options.get("channel_order", "RGB") == "BGR"
        if self.multiple < 1 or self.minimum < 1:
            raise UpscaleError("ONNX padding requirements must be positive.")
        self.tiling = spec.options.get("tiling", "supported")
        self.device, self.precision = device, actual

    def predict(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        sizes = [max(self.minimum, ((v + self.multiple - 1) // self.multiple) * self.multiple) for v in (h, w)]
        if self.square:
            sizes = [max(sizes)] * 2
        sizes = [fixed if fixed is not None else v for fixed, v in zip(self.fixed, sizes)]
        if sizes[0] < h or sizes[1] < w:
            raise UpscaleError(f"ONNX input exceeds fixed shape {self.fixed}; lower --tile and --tile-pad, or use a dynamic export.")
        x = np.pad(image, ((0, sizes[0] - h), (0, sizes[1] - w), (0, 0)), mode="edge")
        if self.bgr and self.input_channels == 3:
            x = x[..., ::-1]
        if self.layout == "NCHW":
            x = x.transpose(2, 0, 1)
        try:
            out = self.session.run([self.output.name], {
                self.input.name: np.ascontiguousarray(x[None], dtype=self.dtype)
            })[0]
        except Exception as e:
            if any(k in str(e).lower() for k in ("out of memory", "cuda_error_out_of_memory")):
                raise BackendOOM("ONNX device ran out of memory.") from None
            raise UpscaleError(f"ONNX inference failed ({type(e).__name__}); check shape/padding options in the model plugin.") from e
        if not isinstance(out, np.ndarray) or out.ndim != 4 or out.shape[0] != 1:
            raise UnsupportedModelError("Unexpected ONNX image output.")
        out = out[0]
        if self.layout == "NCHW":
            out = out.transpose(1, 2, 0)
        expected = (sizes[0] * self.scale, sizes[1] * self.scale, self.output_channels)
        if out.shape != expected:
            raise UnsupportedModelError(f"ONNX output shape {out.shape} does not match {expected}; check native scale/layout.")
        out = out[:h * self.scale, :w * self.scale]
        if self.bgr and self.output_channels == 3:
            out = out[..., ::-1]
        return np.ascontiguousarray(out, dtype=np.float32)

    def clear_cache(self):
        gc.collect()

    def close(self):
        self.session = None
        self.clear_cache()
