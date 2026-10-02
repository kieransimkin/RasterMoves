"""Explicit opt-in checks of the pinned full pretrained component bundles."""
import os

import pytest
from PIL import Image

from rastermoves.refinement import Refiner, RefineOptions


@pytest.mark.live
@pytest.mark.parametrize('refiner_id',['sd15-tile','sdxl-tile'])
def test_pinned_pretrained_refiner(refiner_id):
    if os.environ.get('RASTERMOVES_REFINE_LIVE')!='1':
        pytest.skip('Set RASTERMOVES_REFINE_LIVE=1 to allow large pretrained downloads/inference')
    with Refiner(refiner_id,device=os.environ.get('RASTERMOVES_REFINE_DEVICE','auto'),
                 options=RefineOptions(tile=64,overlap=16,strength=0.5,steps=2,guidance=1.0)) as r:
        output=r.refine_image(Image.new('RGB',(64,64),(75,110,160)))
        assert output.size==(64,64) and output.mode=='RGB'
        assert r.last_report['components']
        assert r.last_report['metrics']['executed_denoising_steps']>=1
