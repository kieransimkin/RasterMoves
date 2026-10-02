"""Real Diffusers CPU API smoke tests with tiny RANDOM weights, no model downloads.

Runs when the optional runtime is installed. CI sets REQUIRE_DIFFUSERS to turn a
missing/broken installation into failure rather than a skip. This is not a quality
benchmark and does not validate the published full-size checkpoints.
"""
import json
import os

import pytest
from PIL import Image

from rastermoves.refinement.diffusers_backend import DiffusersTileBackend
from rastermoves.refinement.specs import BUILTINS, RefineOptions


@pytest.mark.parametrize('family',['sd15','sdxl'])
def test_tiny_real_diffusers(tmp_path, family):
    try:
        import torch
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        from diffusers import (AutoencoderKL, ControlNetModel, DDIMScheduler, UNet2DConditionModel,
                               StableDiffusionControlNetImg2ImgPipeline, StableDiffusionXLControlNetImg2ImgPipeline)
    except ImportError:
        if os.environ.get('RASTERMOVES_REQUIRE_DIFFUSERS') == '1': raise
        pytest.skip('Optional Diffusers runtime not installed')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    vocab=tmp_path/'vocab.json';vocab.write_text(json.dumps({'<|startoftext|>':0,'<|endoftext|>':1}))
    merges=tmp_path/'merges.txt';merges.write_text('#version: 0.2\n')
    tokenizer=CLIPTokenizer(vocab_file=str(vocab),merges_file=str(merges),model_max_length=16)
    text_config=CLIPTextConfig(vocab_size=2,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                              num_attention_heads=2,max_position_embeddings=16,bos_token_id=0,eos_token_id=1,
                              pad_token_id=1,projection_dim=16)
    extra=dict(addition_embed_type='text_time',addition_time_embed_dim=8,
               projection_class_embeddings_input_dim=64) if family=='sdxl' else {}
    unet=UNet2DConditionModel(sample_size=32,in_channels=4,out_channels=4,layers_per_block=1,
                              block_out_channels=(16,32),norm_num_groups=8,
                              down_block_types=('DownBlock2D','CrossAttnDownBlock2D'),
                              up_block_types=('CrossAttnUpBlock2D','UpBlock2D'),
                              cross_attention_dim=32 if family=='sdxl' else 16,attention_head_dim=2,**extra)
    vae=AutoencoderKL(in_channels=3,out_channels=3,latent_channels=4,layers_per_block=1,
                      block_out_channels=(16,32),norm_num_groups=8,sample_size=64,
                      down_block_types=('DownEncoderBlock2D','DownEncoderBlock2D'),
                      up_block_types=('UpDecoderBlock2D','UpDecoderBlock2D'))
    cn=ControlNetModel.from_unet(unet,conditioning_embedding_out_channels=(8,16))
    components=dict(vae=vae,unet=unet,controlnet=cn,text_encoder=CLIPTextModel(text_config),
                    tokenizer=tokenizer,scheduler=DDIMScheduler(clip_sample=False))
    if family=='sdxl':
        pipe=StableDiffusionXLControlNetImg2ImgPipeline(**components,text_encoder_2=CLIPTextModelWithProjection(text_config),
                tokenizer_2=tokenizer,add_watermarker=False)
    else:
        # No safety model exists for these random tiny test weights. Production
        # loading retains and checks the actual SD1.5 safety checker.
        pipe=StableDiffusionControlNetImg2ImgPipeline(**components,safety_checker=None,
                feature_extractor=None,requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    backend=DiffusersTileBackend.__new__(DiffusersTileBackend)
    backend.pipe,backend.torch,backend.device,backend.precision=pipe,torch,'cpu','fp32'
    backend.spec=BUILTINS[family+'-tile']
    try:
        output,stats=backend.predict(Image.new('RGB',(64,64),(80,100,160)),
                options=RefineOptions(steps=4,strength=0.5,guidance=1.0),seed=99)
        assert output.mode=='RGB' and output.size==(64,64)
        assert stats['executed_denoising_steps']==2
        assert stats['refine_vae_encode_seconds']>=0
        assert stats['refine_vae_decode_seconds']>=0
        assert stats['refine_denoising_seconds']>=0
    finally:
        backend.close()
