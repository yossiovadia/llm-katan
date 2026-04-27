"""
Base provider class for llm-katan.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import uuid
from abc import ABC, abstractmethod

from fastapi import FastAPI

from llm_katan.model import ModelBackend


def content_to_text(content) -> str:
    """Normalize message content (string, array of blocks, or None) to plain text."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        mime = url.split(";")[0].split(":", 1)[1] if ";" in url else "image"
                        parts.append(f"[image:{mime}]")
                    else:
                        parts.append("[image:url]")
                elif block.get("type") == "image":
                    source = block.get("source", {})
                    parts.append(f"[image:{source.get('media_type', 'unknown')}]")
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool_use:{block.get('name', '?')}]")
                elif block.get("type") == "tool_result":
                    parts.append(block.get("content", str(block.get("output", ""))))
                elif "text" in block:
                    parts.append(block["text"])
                elif "functionCall" in block:
                    parts.append(f"[functionCall:{block['functionCall'].get('name', '?')}]")
                elif "inlineData" in block:
                    parts.append(f"[image:{block['inlineData'].get('mimeType', 'unknown')}]")
        return " ".join(parts) if parts else ""
    return str(content)


def generate_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def generate_dummy_args(parameters: dict | None) -> dict:
    """Generate minimal valid arguments from a JSON schema."""
    if not parameters or not isinstance(parameters, dict):
        return {}
    props = parameters.get("properties", {})
    result = {}
    for name, schema in props.items():
        typ = schema.get("type", "string")
        if "enum" in schema:
            result[name] = schema["enum"][0]
        elif typ == "string":
            result[name] = f"test_{name}"
        elif typ == "integer":
            result[name] = 1
        elif typ == "number":
            result[name] = 1.0
        elif typ == "boolean":
            result[name] = True
        elif typ == "array":
            result[name] = []
        elif typ == "object":
            result[name] = {}
    return result


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
