class UpscaleError(Exception):
    """An actionable, user-facing error."""

class DownloadError(UpscaleError):
    pass

class IntegrityError(DownloadError):
    pass

class UnsupportedModelError(UpscaleError):
    pass

class BackendOOM(UpscaleError):
    """Device memory exhaustion; the tiler may retry with smaller tiles."""
