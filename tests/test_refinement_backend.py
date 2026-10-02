"""ControlNet/Diffusers adapter tests with fake pipeline objects and real CPU Torch."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from rastermoves.errors import UpscaleError
from rastermoves.refinement import diffusers_backend as db
from rastermoves.refinement.specs import BUILTINS, RefineOptions


def test_allowlisted_download_files():
    for spec in BUILTINS.values():
        for component in (spec.base, spec.controlnet):
            files = db.component_files(component, spec.family)
            assert all(not f.endswith((".py", ".ckpt", ".onnx")) for f in files)
            assert all("fp16" not in f for f in files)
            if component.role == "base":
                assert "vae/diffusion_pytorch_model.safetensors" in files
                assert not any(f.endswith(".bin") for f in files)
    assert "text_encoder_2/model.safetensors" in db.component_files(BUILTINS['sdxl-tile'].base, 'sdxl')


def make_bundle(tmp_path, family):
    spec = BUILTINS[family + "-tile"]
    base, control = tmp_path / "base", tmp_path / "controlnet"
    (base / "unet").mkdir(parents=True)
    control.mkdir()
    index = {"_class_name": "StableDiffusionPipeline" if family == "sd15" else "StableDiffusionXLPipeline",
             "scheduler": ["diffusers", "EulerDiscreteScheduler"],
             "text_encoder": ["transformers", "CLIPTextModel"],
             "tokenizer": ["transformers", "CLIPTokenizer"],
             "unet": ["diffusers", "UNet2DConditionModel"], "vae": ["diffusers", "AutoencoderKL"]}
    if family == "sd15":
        index["safety_checker"] = ["stable_diffusion", "StableDiffusionSafetyChecker"]
        index["feature_extractor"] = ["transformers", "CLIPImageProcessor"]
    else:
        index["text_encoder_2"] = ["transformers", "CLIPTextModelWithProjection"]
        index["tokenizer_2"] = ["transformers", "CLIPTokenizer"]
    (base / "model_index.json").write_text(json.dumps(index))
    config = {"_class_name": "ControlNetModel", "in_channels": 4,
              "cross_attention_dim": 768 if family == "sd15" else 2048}
    (control / "config.json").write_text(json.dumps(config))
    (base / "unet/config.json").write_text(json.dumps(config))
    return spec, {"base": base, "controlnet": control}


@pytest.mark.parametrize("family", ["sd15", "sdxl"])
def test_valid_bundle(tmp_path, family):
    spec, roots = make_bundle(tmp_path, family)
    db.validate_bundle(spec, roots)


@pytest.mark.parametrize("change", ["family", "unet", "control", "custom-loader", "no-checker"])
def test_bad_bundle_rejected(tmp_path, change):
    spec, roots = make_bundle(tmp_path, "sd15")
    path = roots['base'] / 'model_index.json'
    if change in {"unet", "control"}:
        path = roots['base'] / 'unet/config.json' if change == 'unet' else roots['controlnet'] / 'config.json'
    data = json.loads(path.read_text())
    if change == "family": data["_class_name"] = "StableDiffusionXLPipeline"
    if change in {"unet", "control"}: data["cross_attention_dim"] = 2048
    if change == "custom-loader": data["text_encoder"] = ["evil_module", "RunCode"]
    if change == "no-checker": data.pop("safety_checker")
    path.write_text(json.dumps(data))
    with pytest.raises(UpscaleError): db.validate_bundle(spec, roots)


def test_content_addressed_sha256_verification(tmp_path):
    component = BUILTINS['sdxl-tile'].controlnet
    payload = b"example weights"
    path = tmp_path / hashlib.sha256(payload).hexdigest()
    path.write_bytes(payload)
    assert db._verify_hub_file(path, component, 'diffusion_pytorch_model.safetensors')["content_address_verified"]
    path.write_bytes(b"corrupt")
    with pytest.raises(UpscaleError, match="Corrupt"):
        db._verify_hub_file(path, component, 'diffusion_pytorch_model.safetensors')


def test_git_blob_verification(tmp_path):
    b = b'{"example": true}'
    name = hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
    path = tmp_path / name
    path.write_bytes(b)
    assert db._verify_hub_file(path, BUILTINS['sdxl-tile'].base, 'config.json')["content_address_verified"]


def test_sd15_known_checksum_enforced(tmp_path):
    path = tmp_path / 'diffusion_pytorch_model.bin'
    path.write_bytes(b'fake checkpoint')
    with pytest.raises(UpscaleError, match="publisher SHA"):
        db._verify_hub_file(path, BUILTINS['sd15-tile'].controlnet, path.name)


def test_download_pin_offline_cache_and_no_runtime(tmp_path, monkeypatch):
    spec, roots = make_bundle(tmp_path, 'sdxl')
    calls = []
    def download(repo, name, *, revision, cache_dir, local_files_only):
        calls.append((repo, name, revision, local_files_only))
        root = roots['base' if repo == spec.base.repo_id else 'controlnet']
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists(): p.write_bytes(b'placeholder')
        return str(p)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    got, inventory = db.download_bundle(spec, cache_dir=tmp_path/'cache', offline=True)
    assert got == roots and len(inventory) == 2
    assert all(len(revision) == 40 and offline for _, _, revision, offline in calls)
    assert [name for _, name, _, _ in calls[:3]] == ["model_index.json", "unet/config.json", "config.json"]
    with pytest.raises(UpscaleError, match="content address"):
        db.download_bundle(spec, cache_dir=tmp_path/'cache', offline=True, strict_checksums=True)


def test_failed_download_has_actionable_message(tmp_path, monkeypatch):
    import huggingface_hub
    def fail(*a, **k): raise OSError('secret-token-example')
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fail)
    with pytest.raises(UpscaleError) as error:
        db.download_bundle(BUILTINS['sdxl-tile'], cache_dir=tmp_path)
    assert 'HF_TOKEN' in str(error.value)
    assert 'secret-token-example' not in str(error.value)


def test_restricted_torch_load(tmp_path):
    torch = pytest.importorskip('torch')
    path = tmp_path / 'state.bin'
    torch.save({'weight': torch.tensor([2.0])}, path)
    assert db.restricted_state_dict(path, torch)['weight'].item() == 2
    torch.save({'weight': 'not a tensor'}, path)
    with pytest.raises(UpscaleError, match="only named tensors"):
        db.restricted_state_dict(path, torch)
    torch.save({'bad': set([1, 2])}, path)
    with pytest.raises(UpscaleError): db.restricted_state_dict(path, torch)


def test_restricted_loader_never_enables_pickle_fallback(tmp_path):
    calls = []
    def load(*a, **k):
        calls.append(k)
        raise RuntimeError('pickle rejected')
    with pytest.raises(UpscaleError, match='fallback is disabled'):
        db.restricted_state_dict(tmp_path/'unused', SimpleNamespace(load=load))
    assert len(calls) == 1 and calls[0]['weights_only'] is True


@pytest.fixture
def adapter_env(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    state = {'construct': [], 'calls': [], 'flagged': False, 'fail': None, 'step_count': 3}
    class ControlNet:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            state['construct'].append(('control', path, kwargs)); return cls()
        @classmethod
        def from_config(cls, config): return cls()
        def load_state_dict(self, state_dict, strict): assert strict
        def to(self, **kwargs): return self
    class Scheduler:
        config = {'fake': 'scheduler'}
        @classmethod
        def from_config(cls, config): return cls()
    class VAE:
        def encode(self, x): return x
        def decode(self, x): return x
    class Pipe:
        def __init__(self): self.vae = VAE(); self.scheduler = Scheduler()
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            state['construct'].append(('base', path, kwargs)); return cls()
        def to(self, device): state['device'] = device
        def enable_vae_tiling(self): state['vae_tiling'] = True
        def set_progress_bar_config(self, **kwargs): pass
        def __call__(self, **kwargs):
            state['calls'].append(kwargs)
            if state['fail']: raise state['fail']
            self.vae.encode(kwargs['image'])
            for n in range(state['step_count']): kwargs['callback_on_step_end'](self, n, 0, {'latents': None})
            self.vae.decode(kwargs['image'])
            return SimpleNamespace(images=[kwargs['image'].copy()], nsfw_content_detected=[state['flagged']])
    fake = SimpleNamespace(ControlNetModel=ControlNet, DDIMScheduler=Scheduler, EulerDiscreteScheduler=Scheduler,
                           StableDiffusionControlNetImg2ImgPipeline=Pipe, StableDiffusionXLControlNetImg2ImgPipeline=Pipe)
    monkeypatch.setitem(sys.modules, 'diffusers', fake)
    spec, roots = make_bundle(tmp_path, 'sdxl')
    monkeypatch.setattr(db, 'download_bundle', lambda *a, **k: (roots, [{'test_only': True}]))
    return spec, state


def test_adapter_conditioning_seed_and_actual_steps(adapter_env):
    spec, state = adapter_env
    backend = db.DiffusersTileBackend(spec, device='cpu')
    source = Image.new('RGB', (88, 80))
    options = RefineOptions(steps=40, strength=0.22, prompt='stones', negative_prompt='letters')
    output, stats = backend.predict(source, options=options, seed=123)
    call = state['calls'][0]
    assert call['image'] is call['control_image'] is source
    assert call['generator'].initial_seed() == 123
    assert call['width'] == 88 and call['height'] == 80
    assert call['strength'] == 0.22 and call['prompt'] == 'stones'
    assert stats['configured_steps'] == 40 and stats['executed_denoising_steps'] == 3
    assert stats['wall_seconds'] >= 0 and output.size == source.size
    assert all(k['local_files_only'] and k['use_safetensors'] for _, _, k in state['construct'])
    assert state['vae_tiling']
    backend.close()
    assert backend.pipe is None


def test_adapter_safety_flags_not_saved(adapter_env):
    spec, state = adapter_env
    state['flagged'] = True
    backend = db.DiffusersTileBackend(spec, device='cpu')
    with pytest.raises(UpscaleError, match='safety checker'):
        backend.predict(Image.new('RGB', (64, 64)), options=RefineOptions(), seed=0)
    backend.close()


def test_adapter_no_steps_not_silent_success(adapter_env):
    spec, state = adapter_env
    state['step_count'] = 0
    backend = db.DiffusersTileBackend(spec, device='cpu')
    with pytest.raises(UpscaleError, match='no denoising'):
        backend.predict(Image.new('RGB', (64, 64)), options=RefineOptions(), seed=0)
    backend.close()


def test_oom_no_hidden_tile_retries(adapter_env):
    spec, state = adapter_env
    state['fail'] = RuntimeError('CUDA out of memory')
    backend = db.DiffusersTileBackend(spec, device='cpu')
    with pytest.raises(UpscaleError, match='No automatic tile/seed changes'):
        backend.predict(Image.new('RGB', (64, 64)), options=RefineOptions(), seed=0)
    assert len(state['calls']) == 1
    backend.close()


def test_cpu_device_and_precision():
    torch = pytest.importorskip('torch')
    assert db.select_device(torch, 'cpu', 'auto') == ('cpu', 'fp32')
    with pytest.raises(UpscaleError): db.select_device(torch, 'cpu', 'fp16')


def test_rejects_family_before_weight_downloads(tmp_path, monkeypatch):
    import huggingface_hub
    spec, roots = make_bundle(tmp_path, 'sdxl')
    p = roots['controlnet'] / 'config.json'
    data = json.loads(p.read_text()); data['cross_attention_dim'] = 768
    p.write_text(json.dumps(data))
    calls = []
    def fetch(repo, name, **kwargs):
        calls.append(name)
        return str(roots['base' if repo == spec.base.repo_id else 'controlnet'] / name)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fetch)
    with pytest.raises(UpscaleError, match="incompatible"):
        db.download_bundle(spec, cache_dir=tmp_path/'cache')
    assert calls == ['model_index.json', 'unet/config.json', 'config.json']


def test_denoising_error_trace_and_method_cleanup(adapter_env, tmp_path):
    from rastermoves.tracing import TraceRecorder
    spec, state = adapter_env
    backend = db.DiffusersTileBackend(spec, device='cpu')
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic denoising failure')
    model = SimpleNamespace(forward=fail)
    backend.pipe.controlnet = model
    class Invoke:
        def __init__(self, wrapped):
            self.vae, self.controlnet = wrapped.vae, wrapped.controlnet
        def __call__(self, **kwargs):
            self.vae.encode(kwargs['image'])
            self.controlnet.forward()
    backend.pipe = Invoke(backend.pipe)
    trace = tmp_path/'failure.json'
    with TraceRecorder(trace):
        with pytest.raises(UpscaleError):
            backend.predict(Image.new('RGB',(64,64)), options=RefineOptions(), seed=0)
    stages = {e['name']:e for e in json.loads(trace.read_text())['traceEvents'] if e.get('ph')=='X'}
    assert stages['refine_denoising']['args']['status']=='failed'
    assert stages['refine_img2img']['args']['status']=='failed'
    assert stages['refine_denoising']['ts'] >= stages['refine_img2img']['ts']
    assert model.forward is fail
    backend.close()
