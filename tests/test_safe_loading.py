from pathlib import Path
import zipfile
import pytest

from rastermoves.backends.spandrel_backend import safe_state_dict
from rastermoves.errors import UnsupportedModelError


def test_real_torch_weights_only_loading(tmp_path):
    torch=pytest.importorskip("torch")
    p=tmp_path / "model.pth"
    state={"params_ema":{"weight":torch.ones((3,3,3,3))}}
    torch.save(state,p)
    actual=safe_state_dict(p)
    assert torch.equal(actual["params_ema"]["weight"],state["params_ema"]["weight"])


def _forbidden_marker(path):
    Path(path).write_text("executed")
    return {}


class Unsafe:
    def __init__(self,path): self.path=str(path)
    def __reduce__(self): return _forbidden_marker,(self.path,)


def test_unsafe_pickle_never_executed(tmp_path):
    torch=pytest.importorskip("torch")
    p=tmp_path / "unsafe.pth"
    marker=tmp_path / "should-not-exist"
    torch.save(Unsafe(marker),p)
    with pytest.raises(Exception):
        safe_state_dict(p)
    assert not marker.exists()


def test_torchscript_archive_rejected(tmp_path):
    pytest.importorskip("torch")
    p=tmp_path / "script.pt"
    with zipfile.ZipFile(p,"w") as z:
        z.writestr("archive/code/__torch__.py","malicious code must never execute")
        z.writestr("archive/constants.pkl",b"not real pickle")
    with pytest.raises(UnsupportedModelError,match="TorchScript"):
        safe_state_dict(p)


def test_safetensors_loading(tmp_path):
    torch=pytest.importorskip("torch")
    safetensors=pytest.importorskip("safetensors.torch")
    p=tmp_path / "model.safetensors"
    safetensors.save_file({"weight":torch.ones((2,3))},str(p))
    assert torch.equal(safe_state_dict(p)["weight"],torch.ones((2,3)))
