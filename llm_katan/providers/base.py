"""
Base provider class for llm-katan.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from fastapi import FastAPI

from llm_katan.model import ModelBackend

if TYPE_CHECKING:
    from llm_katan.conversations import ConversationStore

logger = logging.getLogger(__name__)


class Provider(ABC):
    """Base class for API format providers."""

    name: str  # e.g., "openai", "anthropic"
    auth_header: str | None = None  # e.g., "Authorization", "x-api-key"

    def __init__(
        self,
        backend: ModelBackend,
        expected_key: str | None = None,
        conversations: ConversationStore | None = None,
    ):
        self.backend = backend
        self.expected_key = expected_key  # None = don't validate value
        self.conversations = conversations

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

    # ── Think-block helpers ──────────────────────────────────────
    _THINK_RE = re.compile(r"<think>(.*?)</think>\s*", flags=re.DOTALL)
    _THINK_OPEN_RE = re.compile(r"<think>\s*", flags=re.DOTALL)

    @staticmethod
    def strip_think(text: str) -> str:
        """Remove <think> blocks entirely, return only the answer."""
        # Closed block: <think>...</think>
        cleaned = Provider._THINK_RE.sub("", text)
        if cleaned != text and cleaned.strip():
            return cleaned.strip()
        # Unclosed block: strip <think> tag and all content (model never reached the answer)
        if text.strip().startswith("<think>"):
            after_tag = Provider._THINK_OPEN_RE.sub("", text, count=1).strip()
            return after_tag if after_tag else text.strip()
        return text.strip()

    @staticmethod
    def split_think(text: str) -> tuple[str | None, str]:
        """Split into (thinking_text, answer_text). thinking is None if absent."""
        m = Provider._THINK_RE.search(text)
        if m:
            thinking = m.group(1).strip() or None
            answer = Provider._THINK_RE.sub("", text).strip()
            return thinking, answer
        # Unclosed <think> — treat entire content as thinking, use it as answer too
        if text.strip().startswith("<think>"):
            raw = Provider._THINK_OPEN_RE.sub("", text, count=1).strip()
            return raw or None, raw or text.strip()
        return None, text.strip()

    # ── Multi turn conversation helpers ─────────────────────────
    @property
    def conversations_enabled(self) -> bool:
        return self.conversations is not None

    async def resolve_messages(
        self,
        conv_id: str | None,
        new_messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> tuple[list[dict[str, str]], str | None]:
        """
        Merge conversation history with new messages. Returns (merged_messages, conversation_id)
        """
        if not self.conversations_enabled:
            return new_messages, None

        if conv_id:
            prior = await self.conversations.get_messages(conv_id)
            if prior is not None:
                merged = prior + new_messages
                return merged, conv_id
            logger.warning(
                "%s | conversation %s not found, starting new", self.name, conv_id,
            )

        conv_id = await self.conversations.create(
            provider=self.name,
            system_prompt=system_prompt,
        )
        if system_prompt:
            return [{"role": "system", "content": system_prompt}] + new_messages, conv_id
        return new_messages, conv_id

    async def store_turn(
        self,
        conv_id: str | None,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """Store a user+assistant turn in the conversation."""
        if not self.conversations_enabled or conv_id is None:
            return
        await self.conversations.append(conv_id, "user", user_content)
        await self.conversations.append(conv_id, "assistant", assistant_content)
