"""Native, pinned SD1.5/SDXL ControlNet Tile img2img. Heavy imports are lazy."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from importlib.metadata import PackageNotFoundError, version
import gc
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import re
import sys
import time

from ..downloads import Downloader, sha256_file
from ..errors import UpscaleError
from ..tracing import event, span
from .specs import Component, RefineOptions, RefinerSpec

log = logging.getLogger(__name__)
BIN_SHA256 = "eb05b4c3665bd76dad70a90652014a9b3aab391abd8a5bb484e860330f9492fb"


def runtime_versions() -> dict:
    result = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "huggingface-hub", "Pillow", "numpy"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def component_files(component: Component, family: str) -> list[str]:
    if component.role == "controlnet":
        ext = "bin" if component.format == "restricted-bin" else "safetensors"
        return ["config.json", f"diffusion_pytorch_model.{ext}"]
    files = ["model_index.json", "scheduler/scheduler_config.json",
             "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
             "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
    for suffix in (["", "_2"] if family == "sdxl" else [""]):
        files += [f"text_encoder{suffix}/config.json", f"text_encoder{suffix}/model.safetensors"]
        files += [f"tokenizer{suffix}/{f}" for f in
                  ("merges.txt", "vocab.json", "tokenizer_config.json", "special_tokens_map.json")]
    if family == "sd15":
        files += ["feature_extractor/preprocessor_config.json", "safety_checker/config.json",
                  "safety_checker/model.safetensors"]
    return sorted(files)


def _verify_hub_file(path: Path, component: Component, filename: str) -> dict:
    digest = sha256_file(path)
    remote_name = path.resolve().name
    verified = False
    if re.fullmatch(r"[a-f0-9]{64}", remote_name):
        if digest != remote_name:
            raise UpscaleError(f"Corrupt cached component: {component.role}/{filename}. Remove it and download again.")
        verified = True
    elif re.fullmatch(r"[a-f0-9]{40}", remote_name):
        data = path.read_bytes()
        git_digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if git_digest != remote_name:
            raise UpscaleError(f"Corrupt cached configuration: {component.role}/{filename}.")
        verified = True
    if (component.repo_id == "lllyasviel/control_v11f1e_sd15_tile"
            and component.revision == "3f877705c37010b7221c3d10743307d6b5b6efac"
            and filename == "diffusion_pytorch_model.bin"):
        if digest != BIN_SHA256:
            raise UpscaleError("SD1.5 Tile checkpoint does not match its publisher SHA-256.")
        verified = True
    return {"filename": filename, "sha256": digest, "size": path.stat().st_size,
            "content_address_verified": verified}


def _read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise UpscaleError(f"Expected an object in {path.name}.")
    return data


def validate_bundle(spec: RefinerSpec, roots: dict[str, Path]) -> None:
    """Reject mismatched model families and custom/dynamic component class loaders."""
    base, control = roots["base"], roots["controlnet"]
    index = _read(base / "model_index.json")
    expected = "StableDiffusionPipeline" if spec.family == "sd15" else "StableDiffusionXLPipeline"
    if index.get("_class_name") != expected:
        raise UpscaleError(f"Base pipeline must be {expected} for {spec.id}.")
    allowed = {
        "vae": ("diffusers", "AutoencoderKL"), "unet": ("diffusers", "UNet2DConditionModel"),
        "text_encoder": ("transformers", "CLIPTextModel"),
        "text_encoder_2": ("transformers", "CLIPTextModelWithProjection"),
        "tokenizer": ("transformers", "CLIPTokenizer"), "tokenizer_2": ("transformers", "CLIPTokenizer"),
        "feature_extractor": ("transformers", "CLIPImageProcessor"),
        "safety_checker": ("stable_diffusion", "StableDiffusionSafetyChecker"),
    }
    schedulers = {"DDIMScheduler", "PNDMScheduler", "EulerDiscreteScheduler", "DPMSolverMultistepScheduler"}
    for name, value in index.items():
        if isinstance(value, list):
            if name == "scheduler" and len(value) == 2 and value[0] == "diffusers" and value[1] in schedulers:
                continue
            if name not in allowed or tuple(value) != allowed[name]:
                raise UpscaleError(f"Unapproved pipeline component loader {name}; remote/custom code is not supported.")
    required = ["vae", "unet", "text_encoder", "tokenizer"]
    required += ["text_encoder_2", "tokenizer_2"] if spec.family == "sdxl" else ["feature_extractor", "safety_checker"]
    for name in required:
        value = index.get(name)
        if not isinstance(value, list) or tuple(value) != allowed[name]:
            raise UpscaleError(f"Required pipeline component {name} is missing or incompatible.")
    if not isinstance(index.get("scheduler"), list):
        raise UpscaleError("Required scheduler configuration is missing.")
    dimension = 768 if spec.family == "sd15" else 2048
    for label, config in (("UNet", _read(base / "unet/config.json")),
                          ("ControlNet", _read(control / "config.json"))):
        if config.get("cross_attention_dim") != dimension or config.get("in_channels") != 4:
            raise UpscaleError(f"{label} is incompatible with {spec.family}; mixed families are not supported.")
    if _read(control / "config.json").get("_class_name") != "ControlNetModel":
        raise UpscaleError("Expected a standard ControlNetModel, not a custom architecture.")


def download_bundle(spec: RefinerSpec, *, cache_dir=None, offline=False, strict_checksums=False) -> tuple[dict, list]:
    """Fetch only required component files, never full repos or executable code.

    Every request uses an immutable revision. HF_TOKEN authentication stays inside
    huggingface_hub. After download, all pipeline loading is forced local-only.
    """
    from huggingface_hub import hf_hub_download
    dl = Downloader(cache_dir, offline=offline)
    cache = dl.root / "diffusion-hub"
    roots, records = {}, {"base": {}, "controlnet": {}}

    def fetch(component, name):
        try:
            path = Path(hf_hub_download(component.repo_id, name, revision=component.revision,
                                       cache_dir=str(cache), local_files_only=dl.offline))
        except Exception as e:
            raise UpscaleError(f"Cannot obtain {component.role} component {name} ({type(e).__name__}). "
                               "Check connectivity, cached files, disk space, and HF_TOKEN/access terms. "
                               "Offline mode requires this exact pinned revision.") from e
        root = path.parents[len(PurePosixPath(name).parts) - 1]
        if component.role in roots and roots[component.role] != root:
            raise UpscaleError("Component files resolved to different snapshots.")
        roots[component.role] = root
        with span("refine_verify", component=component.role, filename=name):
            record = _verify_hub_file(path, component, name)
            if strict_checksums and not record["content_address_verified"]:
                raise UpscaleError(f"No verifiable content address for {component.role}/{name}.")
            records[component.role][name] = record

    with span("refine_download"):
        # Validate the architecture and loader metadata BEFORE any multi-GB weights.
        fetch(spec.base, "model_index.json")
        fetch(spec.base, "unet/config.json")
        fetch(spec.controlnet, "config.json")
        validate_bundle(spec, roots)
        for component in (spec.base, spec.controlnet):
            for name in component_files(component, spec.family):
                if name not in records[component.role]:
                    fetch(component, name)
    inventory = [{**component.to_dict(), "files": list(records[component.role].values())}
                 for component in (spec.base, spec.controlnet)]
    return roots, inventory


def select_device(torch, device: str, precision: str) -> tuple[str, str]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", device):
        raise UpscaleError("Refinement device must be auto, cpu, mps, cuda or cuda:N.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise UpscaleError("CUDA was requested for refinement but is unavailable.")
    if device.startswith("cuda"):
        index = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise UpscaleError("Requested CUDA refinement device does not exist.")
        device = f"cuda:{index}"
    if device == "mps" and not torch.backends.mps.is_available():
        raise UpscaleError("MPS was requested for refinement but is unavailable.")
    if precision == "auto":
        precision = "fp16" if device.startswith("cuda") else "fp32"
    if precision not in {"fp32", "fp16"} or (device == "cpu" and precision == "fp16"):
        raise UpscaleError("Use fp32 on CPU; fp16 is an explicit accelerator option.")
    return device, precision


def restricted_state_dict(path: Path, torch) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise UpscaleError("Restricted ControlNet checkpoint loading failed; unsafe pickle fallback is disabled.") from e
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v)
                                                          for k, v in state.items()):
        raise UpscaleError("ControlNet checkpoint must contain only named tensors.")
    return state


class DiffusersTileBackend:
    def __init__(self, spec: RefinerSpec, *, cache_dir=None, offline=False, device="auto", precision="auto",
                 offload="none", vae_tiling=True, scheduler="ddim", strict_checksums=False):
        try:
            import torch
            from diffusers import (ControlNetModel, DDIMScheduler, EulerDiscreteScheduler,
                                   StableDiffusionControlNetImg2ImgPipeline,
                                   StableDiffusionXLControlNetImg2ImgPipeline)
        except Exception as e:
            raise UpscaleError("Diffusion refinement requires a working optional runtime: "
                               "pip install 'rastermoves[refine]' (or pip install -e '.[refine]'). "
                               f"Import failed: {type(e).__name__}.") from e
        self.torch, self.pipe = torch, None
        self.device, self.precision = select_device(torch, device, precision)
        self.spec = spec
        if offload not in {"none", "model", "sequential"} or (offload != "none" and not self.device.startswith("cuda")):
            raise UpscaleError("Model/sequential CPU offload is supported only with CUDA; use offload=none otherwise.")
        if scheduler not in {"ddim", "euler"}:
            raise UpscaleError("Refinement scheduler must be ddim or euler.")
        dtype = torch.float16 if self.precision == "fp16" else torch.float32
        self.offload, self.scheduler = offload, scheduler
        self.roots, self.inventory = download_bundle(spec, cache_dir=cache_dir, offline=offline,
                                                     strict_checksums=strict_checksums)
        with span("refine_load", refiner=spec.id, device=self.device, precision=self.precision):
            try:
                if spec.controlnet.format == "restricted-bin":
                    cn = ControlNetModel.from_config(_read(self.roots["controlnet"] / "config.json"))
                    state = restricted_state_dict(self.roots["controlnet"] / "diffusion_pytorch_model.bin", torch)
                    cn.load_state_dict(state, strict=True)
                    del state
                    cn.to(dtype=dtype)
                else:
                    cn = ControlNetModel.from_pretrained(str(self.roots["controlnet"]), torch_dtype=dtype,
                                                        use_safetensors=True, local_files_only=True)
                cls = StableDiffusionControlNetImg2ImgPipeline if spec.family == "sd15" else StableDiffusionXLControlNetImg2ImgPipeline
                self.pipe = cls.from_pretrained(str(self.roots["base"]), controlnet=cn, torch_dtype=dtype,
                                                use_safetensors=True, local_files_only=True)
                del cn
                scheduler_cls = DDIMScheduler if scheduler == "ddim" else EulerDiscreteScheduler
                self.pipe.scheduler = scheduler_cls.from_config(self.pipe.scheduler.config)
                if vae_tiling:
                    self.pipe.enable_vae_tiling()
                if offload == "model":
                    self.pipe.enable_model_cpu_offload(gpu_id=int(self.device.split(":")[1]))
                elif offload == "sequential":
                    self.pipe.enable_sequential_cpu_offload(gpu_id=int(self.device.split(":")[1]))
                else:
                    self.pipe.to(self.device)
                self.pipe.set_progress_bar_config(disable=True)
            except Exception as e:
                self.close()
                raise UpscaleError(f"Refinement model loading failed ({type(e).__name__}). Check runtime versions, "
                                   "RAM/VRAM and component compatibility; no unsafe loader fallback was attempted.") from e

    def _sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        elif self.device == "mps":
            self.torch.mps.synchronize()

    @contextmanager
    def _timed_method(self, obj, name, label, timings, before=None):
        original = getattr(obj, name)
        def measured(*args, **kwargs):
            if before is not None:
                before()
            self._sync()
            started = time.perf_counter()
            try:
                with span(label):
                    value = original(*args, **kwargs)
                    self._sync()
                    return value
            finally:
                timings[label + "_seconds"] = timings.get(label + "_seconds", 0) + time.perf_counter() - started
        setattr(obj, name, measured)
        try:
            yield
        finally:
            setattr(obj, name, original)

    @contextmanager
    def _denoising_phase(self, timings):
        """Span from the first ControlNet call to VAE decode; no per-step GPU sync."""
        model = getattr(self.pipe, "controlnet", None)
        original = getattr(model, "forward", None)
        active, started = None, None
        def finish(exc_info=(None, None, None)):
            nonlocal active
            if active is not None:
                measured, active = active, None
                try:
                    if exc_info[0] is None:
                        self._sync()
                finally:
                    measured.__exit__(*exc_info)
                    timings["refine_denoising_seconds"] = time.perf_counter() - started
        if original is not None:
            def forward(*args, **kwargs):
                nonlocal active, started
                if started is None:
                    self._sync()
                    started = time.perf_counter()
                    active = span("refine_denoising")
                    active.__enter__()
                return original(*args, **kwargs)
            model.forward = forward
        try:
            yield finish
        except BaseException:
            finish(sys.exc_info())
            raise
        finally:
            finish()
            if original is not None:
                model.forward = original

    def predict(self, image, *, options: RefineOptions, seed: int):
        pipe, torch = self.pipe, self.torch
        self._sync()
        cuda = self.device.startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        started, cpu_started = time.perf_counter(), time.process_time()
        steps = 0
        timings = {}
        def callback(_pipe, _step, _timestep, values):
            nonlocal steps
            steps += 1
            # No forced per-step synchronization: avoids distorting sampling performance.
            event("refine_step", step=steps)
            return values
        try:
            with ExitStack() as stack:
                stack.enter_context(span("refine_img2img"))
                # Public VAE methods are temporarily wrapped only on this private session.
                finish_denoising = stack.enter_context(self._denoising_phase(timings))
                stack.enter_context(self._timed_method(pipe.vae, "encode", "refine_vae_encode", timings))
                stack.enter_context(self._timed_method(pipe.vae, "decode", "refine_vae_decode", timings,
                                                     before=finish_denoising))
                stack.enter_context(torch.inference_mode())
                result = pipe(prompt=options.prompt, negative_prompt=options.negative_prompt,
                              image=image, control_image=image, width=image.width, height=image.height,
                              strength=options.strength, num_inference_steps=options.steps,
                              guidance_scale=options.guidance, controlnet_conditioning_scale=options.control_scale,
                              generator=torch.Generator(device="cpu").manual_seed(seed),
                              callback_on_step_end=callback, output_type="pil")
            self._sync()
        except Exception as e:
            if isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower():
                raise UpscaleError("Refinement exhausted memory. Reduce --refine-tile / use --refine-offload model. "
                                   "No automatic tile/seed changes were made; rerun with explicit settings.") from e
            raise UpscaleError(f"Diffusion tile failed ({type(e).__name__}): {e}") from e
        if any(getattr(result, "nsfw_content_detected", None) or []):
            raise UpscaleError("The base pipeline safety checker flagged a tile; no refined output was saved.")
        if steps < 1:
            raise UpscaleError("The diffusion runtime executed no denoising steps.")
        stats = {"configured_steps": options.steps, "executed_denoising_steps": steps,
                 "wall_seconds": time.perf_counter() - started, "process_cpu_seconds": time.process_time() - cpu_started}
        stats.update(timings)
        if cuda:
            stats.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                         cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
            event("refine_cuda_memory", **{k: v for k, v in stats.items() if k.startswith("cuda_")})
        return result.images[0].convert("RGB"), stats

    def close(self):
        with span("refine_cleanup"):
            self.pipe = None
            gc.collect()
            if self.device.startswith("cuda"):
                with self.torch.cuda.device(self.device):
                    self.torch.cuda.empty_cache()
            elif self.device == "mps":
                self.torch.mps.empty_cache()
