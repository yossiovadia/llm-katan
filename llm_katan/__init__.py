"""
LLM Katan - One tiny model, every LLM API.

A lightweight LLM serving package using FastAPI and HuggingFace transformers,
designed for testing and development with real tiny models.
Katan means "small" in Hebrew.
"""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("llm-katan")
except PackageNotFoundError:
    __version__ = "0.2.0"

__author__ = "Yossi Ovadia"
__email__ = "yovadia@redhat.com"

from .cli import main
from .model import ModelBackend
from .server import create_app

__all__ = ["create_app", "ModelBackend", "main"]
