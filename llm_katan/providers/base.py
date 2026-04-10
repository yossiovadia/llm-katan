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

    def __init__(self, backend: ModelBackend, expected_key: str | None = None):
        self.backend = backend
        self.expected_key = expected_key  # None = don't validate value

    @abstractmethod
    def register_routes(self, app: FastAPI) -> None:
        """Register provider-specific routes on the FastAPI app."""

    def extract_key_value(self, headers: dict) -> str | None:
        """Extract the API key value from the request headers.

        Subclasses can override for provider-specific extraction
        (e.g., stripping 'Bearer ' prefix).
        Returns the key value or None if header is missing.
        """
        if self.auth_header is None:
            return None
        for key, value in headers.items():
            if key.lower() == self.auth_header.lower():
                return value
        return None

    def check_auth(self, headers: dict) -> str | None:
        """Check auth header exists and optionally validate the key value.

        Returns error message if auth fails, None if OK.
        When validate_keys is enabled, the error message includes the expected key.
        """
        if self.auth_header is None:
            return None

        key_value = self.extract_key_value(headers)
        if key_value is None:
            return f"missing {self.auth_header} header"

        # If key validation is enabled, check the value
        if self.expected_key is not None:
            actual = self._normalize_key(key_value)
            if actual != self.expected_key:
                return (
                    f"invalid API key for {self.name}: "
                    f"got '{actual}', expected '{self.expected_key}'"
                )

        return None

    def _normalize_key(self, raw_value: str) -> str:
        """Normalize the raw header value to just the key.

        Override in subclasses for provider-specific formats.
        Default: return as-is.
        """
        return raw_value
