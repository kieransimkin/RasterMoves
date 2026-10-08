"""RasterMoves: modular image upscaling and enhancement for DanceFlow."""
__version__ = "0.3.2"

# Keep importing the package lightweight; neural frameworks load only on use.
def __getattr__(name):
    if name == "Upscaler":
        from .pipeline import Upscaler
        return Upscaler
    if name == "Registry":
        from .registry import Registry
        return Registry
    raise AttributeError(name)

__all__ = ["Upscaler", "Registry", "__version__"]
