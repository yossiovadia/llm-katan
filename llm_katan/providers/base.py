"""
Base provider class for llm-katan.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

from abc import ABC, abstractmethod

from fastapi import FastAPI

from llm_katan.model import ModelBackend


class Provider(ABC):
    """Base class for API format providers."""

    name: str  # e.g., "openai", "anthropic"
    auth_header: str | None = None  # e.g., "Authorization", "x-api-key"

    def __init__(self, backend: ModelBackend):
        self.backend = backend

    @abstractmethod
    def register_routes(self, app: FastAPI) -> None:
        """Register provider-specific routes on the FastAPI app."""

    def check_auth(self, headers: dict) -> str | None:
        """Check auth header exists. Returns error message if missing, None if OK."""
        if self.auth_header is None:
            return None
        for key in headers:
            if key.lower() == self.auth_header.lower():
                return None
        return f"missing {self.auth_header} header"
