"""Chaining/sweep regression tests with synthetic upscaler and refiner backends."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from rastermoves import sweep
from rastermoves.cli import main
from rastermoves.errors import UpscaleError
from rastermoves.network import atomic_json
from rastermoves.refinement import api, diffusers_backend
from rastermoves.refinement.api import Refiner, paths_for, run_workflow
from rastermoves.specs import ModelSpec


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    source = tmp_path/'input.png'
    Image.new('RGB',(17,13),(20,70,150)).save(source)
    state = {'up': [], 'refine': 0, 'active_up': False, 'active_refine': False, 'fail_up': set(), 'fail_refine': False}
    class Up:
        def __init__(self, model=None, **kwargs):
            assert not state['active_refine']
            self.spec = ModelSpec(id=model or 'a-model', name=model or 'a-model', scale=2)
            self.backend, self.device, self.precision = 'synthetic', 'cpu', 'fp32'
            self.local_file = None
        def upscale_file(self, source, destination, **kwargs):
            assert not state['active_refine']
            state['up'].append(self.spec.id)
            state['active_up'] = True
            if self.spec.id in state['fail_up']: raise UpscaleError('simulated upscale failure')
            with Image.open(source) as image:
                image.resize((image.width*2,image.height*2)).save(destination)
            atomic_json(Path(str(destination)+'.json'), {'upscaler':'synthetic'})
        def close(self): state['active_up'] = False
    class Backend:
        device, precision, inventory = 'synthetic-cpu', 'fp32', []
        def __init__(self,*a,**k):
            assert not state['active_up']
            state['active_refine'] = True
        def predict(self, image, **kwargs):
            assert not state['active_up']
            state['refine'] += 1
            if state['fail_refine']: raise UpscaleError('simulated diffusion failure')
            return image.copy(), {'executed_denoising_steps': 1}
        def close(self): state['active_refine'] = False
    from rastermoves.refinement import cli
    monkeypatch.setattr(api,'Upscaler',Up)
    monkeypatch.setattr(cli,'Upscaler',Up)
    monkeypatch.setattr(diffusers_backend,'DiffusersTileBackend',Backend)
    return source, state, Up


def test_chained_cli_releases_models(workflow,tmp_path):
    source,state,_ = workflow
    assert main(['upscale',str(source),'-o',str(tmp_path/'out.png'),'--refiner','sd15-tile']) == 0
    assert len(state['up']) == state['refine'] == 1
    assert not state['active_up'] and not state['active_refine']
    with Image.open(tmp_path/'out.png') as image: assert image.size == (34,26)


def test_chained_resume_skips_upscale_and_diffusion(workflow,tmp_path):
    source,state,Up = workflow
    out = tmp_path/'out.png'
    with Refiner() as r:
        run_workflow(source,out,r,upscaler=Up())
    state['up'].clear(); state['refine']=0
    with Refiner() as r:
        run_workflow(source,out,r,upscaler=Up(),resume=True)
    assert not state['up'] and state['refine']==0


def test_chained_failure_resume_reuses_baseline(workflow,tmp_path):
    source,state,Up = workflow
    out=tmp_path/'out.png'
    state['fail_refine']=True
    with Refiner() as r, pytest.raises(UpscaleError):
        run_workflow(source,out,r,upscaler=Up())
    assert paths_for(out)['baseline'].exists()
    state['fail_refine']=False
    with Refiner() as r:
        run_workflow(source,out,r,upscaler=Up(),resume=True)
    assert len(state['up'])==1


def test_refined_sweep_failure_isolation_and_resume(workflow,tmp_path):
    source,state,_=workflow
    specs=[ModelSpec(id=x,name=x,scale=2) for x in ['a-model','b-model','c-model']]
    config={'refiner':'sd15-tile'}
    state['fail_up'].add('b-model')
    summary=sweep.run_all_models(source,tmp_path/'compare',specs,refinement=config)
    assert summary['counts']['success']==2 and summary['counts']['failed']==1
    for row in summary['results']:
        if row['status']=='success': assert 'baseline' in row['refinement_artifacts']
    state['fail_up'].clear();state['up'].clear();state['refine']=0
    summary=sweep.run_all_models(source,tmp_path/'compare',specs,refinement=config,resume=True)
    assert summary['counts']['reused']==2 and summary['counts']['success']==1
    assert state['up']==['b-model'] and state['refine']==1
    summary=sweep.run_all_models(source,tmp_path/'compare',specs,refinement=config,resume=True)
    assert summary['counts']['reused']==3
    assert all('baseline' in r['refinement_artifacts'] for r in summary['results'])


@pytest.mark.parametrize('change',['baseline','report','spec','output'])
def test_sweep_tamper_rerun(workflow,tmp_path,change):
    source,state,_=workflow
    spec=ModelSpec(id='a-model',name='a-model',scale=2)
    config={'refiner':'sd15-tile'}
    directory=tmp_path/'compare'
    sweep.run_all_models(source,directory,[spec],refinement=config)
    paths=paths_for(directory/'model-a-model.png')
    if change=='spec': spec=ModelSpec(id='a-model',name='new model',scale=2)
    else: paths[change].write_bytes(b'corrupted')
    result=sweep.run_all_models(source,directory,[spec],refinement=config,resume=True)
    # A fake Up returns the same fixed manifest, unlike the real registry; the outer
    # sweep still detects the changed supplied spec and rebuilds its output.
    assert result['counts']['success']==1


def test_sweep_rejects_changed_refiner(workflow,tmp_path):
    source,state,_=workflow
    specs=[ModelSpec(id='a-model',name='a-model',scale=2)]
    directory=tmp_path/'compare'
    sweep.run_all_models(source,directory,specs,refinement={'refiner':'sd15-tile'})
    with pytest.raises(UpscaleError,match='Cannot resume'):
        sweep.run_all_models(source,directory,specs,refinement={'refiner':'sdxl-tile'},resume=True)


def test_chained_batch_no_simultaneous_models(workflow,tmp_path):
    source,state,_=workflow
    inputs=tmp_path/'inputs';inputs.mkdir()
    Image.open(source).save(inputs/'a.png');Image.open(source).save(inputs/'b.png')
    assert main(['upscale',str(inputs),'-o',str(tmp_path/'outputs'),'--refiner','sd15-tile'])==0
    assert len(state['up'])==state['refine']==2


@pytest.mark.parametrize('contents',['[]','{}','null','5','{broken'])
def test_bad_resume_report(workflow,tmp_path,contents):
    source,_,_=workflow
    out=tmp_path/'out.png';Path(str(out)+'.json').write_text(contents)
    with Refiner() as r,pytest.raises(UpscaleError):r.refine_file(source,out,resume=True)
