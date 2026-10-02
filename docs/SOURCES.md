# Primary references

Inspected on 2026-10-02 while implementing this release. Links identify documentation
and metadata, not evidence that every remote download or model was executed.

## Catalogue schema and model metadata

- OpenModelDB: https://openmodeldb.info/
- Official source and licence: https://github.com/OpenModelDB/open-model-database
- Export implementation: https://github.com/OpenModelDB/open-model-database/blob/main/scripts/generate-api.ts
- Model JSON records: https://github.com/OpenModelDB/open-model-database/tree/main/data/models
- Export URL used by this implementation: https://openmodeldb.info/api/v1/models.json
- Official repository fallback: https://codeload.github.com/OpenModelDB/open-model-database/tar.gz/refs/heads/main

Each bundled model manifest has its specific `source_page`, licence, author and exact
resource URLs. README.md links all eight records. The importer extracts selected
metadata, omits sample images, and maps resource platforms into backend names.
Descriptions in the starter manifests are abbreviated factual summaries, not complete
copies of the authors' descriptions.

## Inference and safe loading

- Spandrel: https://github.com/chaiNNer-org/spandrel
- Spandrel release metadata: https://pypi.org/project/spandrel/
- Spandrel ModelLoader implementation inspected:
  https://github.com/chaiNNer-org/spandrel/blob/b0c3eb2d044112a5302200757e84f839b971393b/libs/spandrel/spandrel/__helpers/loader.py
- Spandrel checkpoint normalization reference:
  https://github.com/chaiNNer-org/spandrel/blob/b0c3eb2d044112a5302200757e84f839b971393b/libs/spandrel/spandrel/__helpers/canonicalize.py
- PyTorch installer: https://pytorch.org/get-started/locally/
- PyTorch serialization: https://docs.pytorch.org/docs/stable/notes/serialization.html
- PyTorch torch.load API: https://docs.pytorch.org/docs/stable/generated/torch.load.html
- ONNX Runtime Python API: https://onnxruntime.ai/docs/api/python/api_summary.html
- ONNX Runtime CUDA requirements: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- Safetensors: https://huggingface.co/docs/safetensors/

## Model transport

- Hugging Face download and cache API: https://huggingface.co/docs/huggingface_hub/guides/download
- Hugging Face environment variables: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
- gdown: https://github.com/wkentaro/gdown

## Implementation/licence distinction

Source code was written for this project. No pretrained weights or third-party source
packages are embedded. Dependency package licences and model-weight licences remain
separate. OpenModelDB metadata attribution and GPL source terms are retained in
THIRD_PARTY_NOTICES.md. The project is distributed under GPL-3.0-only.


## Resource tracing (0.2.1)

- Current psutil release API, including portable RSS/VMS and memory context:
  https://psutil.io/7.2/
- Python monotonic wall and process CPU timers:
  https://docs.python.org/3/library/time.html#time.perf_counter
  https://docs.python.org/3/library/time.html#time.process_time
- Perfetto's support for Chrome JSON duration, counter and metadata events:
  https://perfetto.dev/docs/getting-started/other-formats#chrome-json-format

Sources verified on 2026-10-02. The psutil extra is capped below 8 pending review of
its breaking API changes; the sampler uses the documented 5.9.8–7.x methods. Recorder
CPU percentages are calculated from process-time deltas, not psutil's first-call
CPU-percentage value. No GPU utilization or allocation-profiler data is inferred.


## Generative refinement implementation references (checked 2026-10-02)

These primary sources informed the API adapter and pinned workflow metadata. This
list is not evidence of successful pretrained execution in the patch environment.

- Diffusers 0.35.2 SD1.5 ControlNet img2img source/signature: https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/controlnet/pipeline_controlnet_img2img.py
- SDXL counterpart: https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py
- Img2img strength/steps: https://huggingface.co/docs/diffusers/en/using-diffusers/img2img
- Memory/offload/VAE tiling: https://huggingface.co/docs/diffusers/en/optimization/memory
- Callback API: https://huggingface.co/docs/diffusers/en/using-diffusers/callback
- SD1.5 base: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
- Original SD1.5 Tile: https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/tree/3f877705c37010b7221c3d10743307d6b5b6efac
- SD1.5 Tile file hash: https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/blob/3f877705c37010b7221c3d10743307d6b5b6efac/diffusion_pytorch_model.bin
- SDXL base: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b
- Xinsir Tile: https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0/tree/1ae8d9529efe58f7362a987363ff86a7904dc84f
- Hub caching/downloads: https://huggingface.co/docs/huggingface_hub/en/guides/download
- Restricted loading: https://docs.pytorch.org/docs/stable/generated/torch.load.html
- CUDA allocator measurements: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html

Tile assembly in this package is independent frozen-image crop/blend code. No code
from the non-commercial AUTOMATIC1111 tiled-diffusion extension, no guided-filter
repository Python, and no ComfyUI custom-node implementation was imported.
