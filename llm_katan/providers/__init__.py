"""
Provider registry for llm-katan.

Each provider implements its own API format (OpenAI, Anthropic, Bedrock, etc.)
backed by the same model inference engine.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

from .base import Provider

_REGISTRY: dict[str, type[Provider]] = {}


def register_provider(name: str, cls: type[Provider]):
    _REGISTRY[name] = cls


def get_provider(name: str) -> type[Provider]:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown provider: {name!r}. Available: {available}")
    return _REGISTRY[name]


def available_providers() -> list[str]:
    return sorted(_REGISTRY.keys())


# Auto-register built-in providers on import
from . import openai as _openai  # noqa: F401, E402
