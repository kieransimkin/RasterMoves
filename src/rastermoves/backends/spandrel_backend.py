from __future__ import annotations

from collections.abc import Mapping
import gc
from pathlib import Path
import zipfile

import numpy as np

from ..errors import BackendOOM, UnsupportedModelError, UpscaleError


def safe_state_dict(path: Path):
    import torch
    major, minor = map(int, torch.__version__.split("+")[0].split(".")[:2])
    if (major, minor) < (2, 6):
        raise UpscaleError("PyTorch >=2.6 is required for restricted checkpoint loading. Upgrade torch/torchvision.")
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                if any("/code/" in n or n.endswith("/constants.pkl") for n in z.namelist()):
                    raise UnsupportedModelError("TorchScript archives are not accepted; use state-dict weights or safetensors.")
        # Never fall back to weights_only=False or torch.hub/trust_remote_code.
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise UnsupportedModelError("Expected a tensor state dictionary, not a serialized Python model.")
    return state


def normalize_state_dict(state):
    """Unwrap common training checkpoints without invoking private Spandrel APIs.

    Spandrel's public load_from_state_dict expects the actual parameter mapping,
    unlike its load_from_file convenience method. Only passive mappings are read.
    """
    wrappers = ("model_state_dict", "state_dict", "params_ema", "params-ema", "params", "model", "net")
    for _ in range(8):
        child = next((state[k] for k in wrappers if k in state and isinstance(state[k], Mapping)), None)
        if child is None and len(state) == 1:
            value = next(iter(state.values()))
            child = value if isinstance(value, Mapping) else None
        if child is None:
            break
        state = child
    if not state or not all(isinstance(k, str) for k in state):
        raise UnsupportedModelError("Expected a nonempty state dictionary with string parameter names.")
    for _ in range(8):
        prefix = next((p for p in ("module.", "netG.") if all(k.startswith(p) for k in state)), None)
        if prefix is None:
            break
        state = {k[len(prefix):]: v for k, v in state.items()}
    return state


_EXTRA_ARCHES_INSTALLED = False


def resolve_device(requested: str):
    import torch
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if requested not in {"cpu", "mps", "cuda"} and not requested.startswith("cuda:"):
        raise UpscaleError("PyTorch device must be auto, cpu, mps, cuda, or cuda:N.")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as e:
        raise UpscaleError("Invalid PyTorch device; use cpu, mps, cuda, or cuda:N.") from e
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise UpscaleError("CUDA is unavailable. Install the matching PyTorch GPU build or use --device cpu.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise UpscaleError("Requested CUDA device index does not exist.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise UpscaleError("Apple MPS is unavailable on this machine/PyTorch build.")
    return device


class SpandrelModel:
    def __init__(self, path: Path, spec, *, device="auto", precision="fp32", extra_arches=False,
                 validate_metadata=True):
        try:
            import torch
            from spandrel import ImageModelDescriptor, ModelLoader
        except (ImportError, RuntimeError, OSError) as e:
            raise UpscaleError("PyTorch backend unavailable. Install the torch extra and matching torch/torchvision builds.") from e
        global _EXTRA_ARCHES_INSTALLED
        if extra_arches and not _EXTRA_ARCHES_INSTALLED:
            try:
                import spandrel_extra_arches
                spandrel_extra_arches.install()
                _EXTRA_ARCHES_INSTALLED = True
            except ImportError as e:
                raise UpscaleError("Install the extra-arches extra from this project and review the architecture licences.") from e
        target = resolve_device(device)
        try:
            state = normalize_state_dict(safe_state_dict(path))
            descriptor = ModelLoader(device="cpu").load_from_state_dict(state)
            if not isinstance(descriptor, ImageModelDescriptor):
                raise UnsupportedModelError("This is not an image-to-image model; masked/inpainting models need another plugin.")
            if descriptor.purpose == "FaceSR":
                raise UnsupportedModelError("Face restoration needs face detection/alignment; this is not a general image upscaler.")
            if descriptor.input_channels not in (1, 3) or descriptor.output_channels not in (1, 3):
                raise UnsupportedModelError("This image pipeline supports only RGB and grayscale neural models.")
            if validate_metadata and (descriptor.scale != spec.scale or
                    descriptor.input_channels != spec.input_channels or
                    descriptor.output_channels != spec.output_channels):
                raise UnsupportedModelError("Detected scale/channels disagree with the manifest. Fix the plugin or use a standalone local model.")
            if precision not in {"fp32", "fp16"}:
                raise UpscaleError("Precision must be fp32 or fp16.")
            if precision == "fp16" and (target.type != "cuda" or not descriptor.supports_half):
                raise UpscaleError("FP16 requires CUDA and a model advertising half-precision support; use fp32.")
            descriptor.to(device=target, dtype=torch.float16 if precision == "fp16" else torch.float32)
            descriptor.eval()
        except UpscaleError:
            raise
        except Exception as e:
            raise UnsupportedModelError(f"Unable to safely load this checkpoint ({type(e).__name__}). "
                                        "Check the architecture, dependency versions and weight format; unsafe pickle loading is disabled.") from e
        self.descriptor = descriptor
        self.scale = int(descriptor.scale)
        self.input_channels = descriptor.input_channels
        self.output_channels = descriptor.output_channels
        self.tiling = descriptor.tiling.name.lower()
        self.device, self.precision = str(target), precision
        self.architecture = str(descriptor.architecture.id)

    def predict(self, image: np.ndarray) -> np.ndarray:
        import torch
        tensor = output = None
        try:
            tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
            tensor = tensor.to(device=self.descriptor.device, dtype=self.descriptor.dtype)
            with torch.inference_mode():
                # The descriptor pads to its size/window requirements and crops output back.
                output = self.descriptor(tensor)
            return output.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
        except RuntimeError as e:
            if isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower():
                raise BackendOOM("Device ran out of memory during inference.") from None
            raise UpscaleError(f"PyTorch inference failed: {e}") from e
        finally:
            del tensor, output

    def clear_cache(self):
        import torch
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def close(self):
        self.descriptor = None
        self.clear_cache()
