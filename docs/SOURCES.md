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
