# Third-party notices

## OpenModelDB

Bundled model metadata and the catalogue integration are based on the community's
OpenModelDB project: https://github.com/OpenModelDB/open-model-database

Copyright belongs to the OpenModelDB contributors and the respective model creators.
The OpenModelDB repository is published under the GNU General Public License v3.0.
A copy of GPLv3 is included as LICENSE. This source distribution uses GPL-3.0-only.

Changes relative to the original model records: selected metadata is converted into
RasterMoves's plugin schema; descriptions are abbreviated; training/example-image
fields are omitted; resources are mapped to runtime backend names. Original authors,
model licence identifiers, source pages, file sizes and supplied SHA-256 hashes are
preserved. No claim is made to authorship of the models or their metadata.

## Model weights

No weights are distributed here. Models downloaded at runtime are separate works
under their creators' terms. Their licences are visible in each manifest and through
`rastermoves info`. The project's GPL licence does not override model terms, including
non-commercial restrictions or attribution/share-alike requirements.

## Runtime dependencies

PyTorch, torchvision, Spandrel, ONNX Runtime, safetensors, Hugging Face Hub, gdown,
Pillow, NumPy, requests, platformdirs and filelock are installed separately and retain
their upstream licences. Optional spandrel-extra-arches can add architectures with
additional or different terms. Consult the licences in the installed packages.
