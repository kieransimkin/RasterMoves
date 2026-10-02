import numpy as np
import pytest

from rastermoves.errors import BackendOOM, UpscaleError
from rastermoves.tiling import upscale_array, positions


@pytest.mark.parametrize("shape,tile,overlap", [
    ((1,1,3),16,4), ((7,9,3),16,4), ((31,47,3),16,4),
    ((64,63,3),32,12), ((100,117,3),33,16), ((19,20,3),0,0),
    ((29,31,3),17,0), ((25,25,3),12,11),
])
def test_tiled_matches_whole_image(repeat_model, shape, tile, overlap):
    x = np.random.default_rng(42).random(shape, dtype=np.float32)
    expected = repeat_model.predict(x)
    actual = upscale_array(x, repeat_model, tile=tile, overlap=overlap, pad=3)
    np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_context_halo_removes_convolution_seams(repeat_model):
    def blur(x):
        p = np.pad(x, ((1,1),(1,1),(0,0)), mode="edge")
        h,w,_ = x.shape
        y = sum(p[dy:dy+h, dx:dx+w] for dy in range(3) for dx in range(3)) / 9
        return y.repeat(2,0).repeat(2,1)
    repeat_model.predict = blur
    x = np.random.default_rng(3).random((79,93,3), dtype=np.float32)
    actual = upscale_array(x, repeat_model, tile=19, overlap=5, pad=2)
    np.testing.assert_allclose(actual, blur(x), atol=3e-7)


def test_device_oom_restarts_smaller(repeat_model):
    original = repeat_model.predict
    calls = []
    def limited(x):
        calls.append(x.shape)
        if max(x.shape[:2]) > 32:
            raise BackendOOM()
        return original(x)
    repeat_model.predict = limited
    x = np.full((81,77,3),0.25,dtype=np.float32)
    y = upscale_array(x, repeat_model, tile=64, overlap=16, pad=8)
    assert any(max(shape[:2]) > 32 for shape in calls)
    np.testing.assert_allclose(y,0.25,atol=1e-7)


def test_global_context_is_not_silently_tiled(repeat_model):
    repeat_model.tiling = "discouraged"
    shapes = []
    original = repeat_model.predict
    def call(x):
        shapes.append(x.shape)
        return original(x)
    repeat_model.predict = call
    upscale_array(np.zeros((50,60,3),np.float32),repeat_model,tile=16,overlap=4)
    assert shapes == [(50,60,3)]


@pytest.mark.parametrize("kind", ["nan", "shape", "runtime"])
def test_bad_backend_output_fails(repeat_model, kind):
    def bad(x):
        if kind == "runtime":
            raise RuntimeError("not OOM")
        return np.full((4,4,3),np.nan if kind=="nan" else 0,np.float32)
    repeat_model.predict = bad
    with pytest.raises((UpscaleError, RuntimeError)):
        upscale_array(np.zeros((4,5,3),np.float32),repeat_model,tile=0)


@pytest.mark.parametrize("kwargs", [{"tile":-1}, {"tile":8,"overlap":8}, {"pad":-1}, {"max_output_pixels":1}])
def test_invalid_tiling_options(repeat_model,kwargs):
    with pytest.raises(UpscaleError):
        upscale_array(np.zeros((4,5,3),np.float32),repeat_model,**kwargs)


def test_progress(repeat_model):
    progress = []
    upscale_array(np.zeros((30,40,3),np.float32),repeat_model,tile=16,overlap=4,
                  progress=lambda n,total:progress.append((n,total)))
    assert progress[-1][0] == progress[-1][1]
    assert [p[0] for p in progress] == list(range(1,len(progress)+1))


def test_actual_torch_convolution_matches_tiled(repeat_model):
    torch=pytest.importorskip("torch")
    layer=torch.nn.Conv2d(3,3,3,padding=1,padding_mode="replicate",bias=False)
    with torch.no_grad():
        layer.weight.fill_(1/27)
    def predict(x):
        tensor=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1)))[None]
        with torch.inference_mode():
            out=layer(tensor).repeat_interleave(2,2).repeat_interleave(2,3)
        return out[0].permute(1,2,0).numpy()
    repeat_model.predict=predict
    x=np.random.default_rng(2).random((29,33,3),dtype=np.float32)
    np.testing.assert_allclose(upscale_array(x,repeat_model,tile=12,overlap=4,pad=2),predict(x),atol=3e-7)


@pytest.mark.parametrize("option", [{"overlap": 1.5}, {"pad": 1.5}, {"tile": True}])
def test_tile_options_require_integers(repeat_model, option):
    with pytest.raises(UpscaleError, match="integers"):
        upscale_array(np.zeros((4, 4, 3), np.float32), repeat_model, **option)
