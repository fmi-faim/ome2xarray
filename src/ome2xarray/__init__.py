"""Top-level package for ome2xarray."""

__all__ = ["companion", "CompanionFile", "sanitize_pixels"]

from . import companion
from .companion import CompanionFile, sanitize_pixels
