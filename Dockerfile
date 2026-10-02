# Build a wheel first: python -m build. Release CI copies its tested wheel here.
FROM python:3.12-slim-bookworm
ARG VERSION=dev
ARG REVISION=unknown
ARG SOURCE_URL=https://github.com/kieransimkin/RasterMoves
LABEL org.opencontainers.image.title="RasterMoves" \
      org.opencontainers.image.description="Modular CPU image upscaling for DanceFlow" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.licenses="GPL-3.0-only"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    RASTERMOVES_CACHE=/cache \
    HF_HOME=/cache/huggingface \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY dist/*.whl /tmp/wheels/
# Explicit CPU wheels prevent accidentally shipping multi-GB CUDA dependencies.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6' 'torchvision>=0.21' \
    && for wheel in /tmp/wheels/*.whl; do python -m pip install "${wheel}[all]"; done \
    && python -m pip check \
    && rm -rf /tmp/wheels \
    && mkdir -p /cache /work \
    && chown -R 10001:10001 /cache /work
USER 10001:10001
WORKDIR /work
ENTRYPOINT ["rastermoves"]
CMD ["--help"]
