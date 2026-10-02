import json
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image
import pytest

from rastermoves.cli import main, batch_jobs
from rastermoves.errors import UpscaleError
from rastermoves.pipeline import Upscaler, target_size


@pytest.fixture
def upscaler(tmp_path, repeat_model):
    up=Upscaler(cache_dir=tmp_path / "cache")
    up.loaded=repeat_model
    yield up
    up.close()


def test_rgba_preserved_and_target_size(upscaler):
    a=np.zeros((13,17,4),np.uint8)
    a[...,0]=180
    a[...,3]=np.arange(17)*15
    original=Image.fromarray(a)
    result=upscaler.upscale_image(original,width=51,tile=8,overlap=2,tile_pad=2)
    assert result.mode=="RGBA" and result.size==(51,39)
    np.testing.assert_array_equal(np.asarray(result.getchannel("A")),
                                  np.asarray(original.getchannel("A").resize((51,39),Image.Resampling.LANCZOS)))


def test_alpha_can_use_neural_backend(upscaler):
    original=Image.new("RGBA",(8,7),(100,150,200,128))
    result=upscaler.upscale_image(original,alpha="model",tile=0)
    assert result.mode=="RGBA" and result.size==(16,14)
    assert result.getchannel("A").getextrema()==(128,128)


def test_palette_transparency_preserved(upscaler):
    image=Image.new("P",(8,9),0)
    image.putpalette([255,0,0]+[0]*765)
    image.info["transparency"]=0
    result=upscaler.upscale_image(image,tile=0)
    assert result.mode=="RGBA"
    assert result.getchannel("A").getextrema()==(0,0)


def test_file_report_and_no_overwrite(tmp_path, upscaler):
    source=tmp_path / "in.png"
    output=tmp_path / "out.png"
    Image.new("RGB",(7,5),(20,40,60)).save(source)
    upscaler.upscale_file(source,output,tile=0,report=True)
    assert Image.open(output).size==(14,10)
    report=json.loads(output.with_suffix(".png.json").read_text())
    assert report["native_scale"]==2 and report["output_size"]==[14,10]
    with pytest.raises(UpscaleError,match="already exists"):
        upscaler.upscale_file(source,output)
    with pytest.raises(UpscaleError,match="different files"):
        upscaler.upscale_file(source,source,overwrite=True)


def test_alpha_to_jpeg_rejected_before_model_load(tmp_path,upscaler):
    source=tmp_path / "in.png"
    Image.new("RGBA",(5,5)).save(source)
    with pytest.raises(UpscaleError,match="JPEG cannot preserve alpha"):
        upscaler.upscale_file(source,tmp_path / "out.jpg")


def test_exif_orientation_applied(upscaler):
    image=Image.new("RGB",(7,5))
    exif=image.getexif()
    exif[274]=6
    image.info["exif"]=exif.tobytes()
    result=upscaler.upscale_image(image,tile=0)
    assert result.size==(10,14)
    assert not result.info.get("exif")


def test_high_bit_depth_and_animation_rejected(upscaler):
    with pytest.raises(UpscaleError,match="High-bit-depth"):
        upscaler.upscale_image(Image.fromarray(np.zeros((5,5),np.uint16)))
    buffer=BytesIO()
    Image.new("RGB",(5,5),"red").save(buffer,format="GIF",save_all=True,
                                      append_images=[Image.new("RGB",(5,5),"blue")])
    buffer.seek(0)
    with Image.open(buffer) as animated, pytest.raises(UpscaleError,match="Animated"):
        upscaler.upscale_image(animated)


def test_size_validation(upscaler):
    assert target_size(100,50,4,long_edge=300)==(300,150)
    assert target_size(100,50,4,target_height=200)==(400,200)
    assert target_size(100,50,4,scale=1.5)==(150,75)
    for kwargs in ({"scale":0},{"scale":float("nan")},{"target_width":1,"target_height":1}):
        with pytest.raises(UpscaleError):
            target_size(100,50,4,**kwargs)
    with pytest.raises(UpscaleError,match="exceeds"):
        upscaler.upscale_image(Image.new("RGB",(10,10)),width=100,max_output_mp=0.001)


def test_batch_collision_resistance_and_output_exclusion(tmp_path):
    source=tmp_path / "images"
    source.mkdir()
    (source / "photo.jpg").touch()
    (source / "photo.png").touch()
    output=source / "results"
    output.mkdir()
    (output / "previous.png").touch()
    jobs=batch_jobs(source,output,recursive=True)
    assert len(jobs)==2 and len({out for _,out in jobs})==2
    assert {out.name for _,out in jobs}=={"photo.jpg_upscaled.png","photo.png_upscaled.png"}
    with pytest.raises(UpscaleError):
        batch_jobs(source,source)


def test_cli_models_json_and_info_offline(tmp_path,capsys):
    flags=["--cache-dir",str(tmp_path)]
    assert main(flags+["models","--architecture","hat","--json"])==0
    items=json.loads(capsys.readouterr().out)
    assert len(items)==1 and items[0]["id"]=="4x-LexicaHAT"
    assert main(flags+["info","4x-UltraSharpV2","--offline"])==0
    assert json.loads(capsys.readouterr().out)["license"]=="CC-BY-NC-SA-4.0"


def test_cli_error_exit(tmp_path,capsys):
    assert main(["--cache-dir",str(tmp_path),"download","not-in-catalog","--offline"])==1
    assert "ERROR" in capsys.readouterr().err


def test_standalone_onnx_requires_scale(tmp_path):
    path=tmp_path / "model.onnx"
    path.write_bytes(b"test")
    with pytest.raises(UpscaleError,match="native-scale"):
        Upscaler(model_file=path,cache_dir=tmp_path / "cache")


def test_rejects_16bit_rgb_png_before_pillow_precision_loss(tmp_path):
    import struct
    import zlib
    from rastermoves.pipeline import _prepare
    def chunk(name, data):
        return struct.pack("!I", len(data)) + name + data + struct.pack("!I", zlib.crc32(name + data))
    header = struct.pack("!IIBBBBB", 1, 1, 16, 2, 0, 0, 0)
    data = b"\x00" + struct.pack("!HHH", 65535, 32000, 12345)
    p = tmp_path / "rgb16.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(data)) + chunk(b"IEND", b""))
    with Image.open(p) as image:
        assert image.mode == "RGB"  # Mode alone is not enough to detect 16-bit source precision.
        with pytest.raises(UpscaleError, match="High-bit-depth"):
            _prepare(image)
