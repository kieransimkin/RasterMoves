from __future__ import annotations
from typing import Protocol
import numpy as np


class LoadedModel(Protocol):
    """Backend contract: HWC float32 RGB/gray arrays in [0,1], no batch axis."""
    scale: int
    input_channels: int
    output_channels: int
    tiling: str  # supported | discouraged | internal
    device: str
    precision: str

    def predict(self, image: np.ndarray) -> np.ndarray: ...
    def clear_cache(self) -> None: ...
    def close(self) -> None: ...
