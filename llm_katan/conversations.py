"""
Conversation store for multi-turn support.

In-memory store that tracks conversation history across requests,
allowing providers to automatically prepend prior turns when calling
the backend.

added by : Ariel Harush
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Conversation:
    """A single conversation with its message history."""

    id: str
    provider: str
    messages: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.last_active = time.time()


class ConversationStore:
    """in-memory conversation store with time to live (ttl) and least revently updated(lru) eviction:
    ttl: time to live in seconds
    max_conversations: maximum number of conversations to store
    """
    def __init__(self, ttl_seconds: int = 3600, max_conversations: int = 1000):
        self._conversations: dict[str, Conversation] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl_seconds
        self._max = max_conversations

    async def create(
        self,
        provider: str,
        system_prompt: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Create a new conversation and return its id."""
        conv_id = f"conv_{uuid.uuid4().hex[:24]}"
        conv = Conversation(
            id=conv_id,
            provider=provider,
            metadata=metadata or {},
        )
        if system_prompt:
            conv.messages.append({"role": "system", "content": system_prompt})

        async with self._lock:
            self._evict_if_needed()
            self._conversations[conv_id] = conv

        logger.info("conversation created: %s (provider=%s)", conv_id, provider)
        return conv_id

    async def get(self, conv_id: str) -> Conversation | None:
        """Get a conversation by ID. Returns None if not found or expired."""
        async with self._lock:
            conv = self._conversations.get(conv_id)
            if conv is None:
                return None
            if self._is_expired(conv):
                del self._conversations[conv_id]
                logger.info("conversation expired: %s", conv_id)
                return None
            return conv

    async def append(self, conv_id: str, role: str, content: str) -> bool:
        """Append a message to a conversation. Returns False if not found."""
        async with self._lock:
            conv = self._conversations.get(conv_id)
            if conv is None or self._is_expired(conv):
                return False
            conv.append(role, content)
            return True

    async def get_messages(self, conv_id: str) -> list[dict[str, str]] | None:
        """Get all messages for a conversation. Returns None if not found."""
        conv = await self.get(conv_id)
        if conv is None:
            return None
        return list(conv.messages)

    async def delete(self, conv_id: str) -> bool:
        """Delete a conversation. Returns False if not found."""
        async with self._lock:
            if conv_id in self._conversations:
                del self._conversations[conv_id]
                logger.info("conversation deleted: %s", conv_id)
                return True
            return False

    async def clear(self) -> int:
        """Delete all conversations. Returns count deleted."""
        async with self._lock:
            count = len(self._conversations)
            self._conversations.clear()
            logger.info("all conversations cleared (%d)", count)
            return count

    async def list_conversations(self) -> list[dict]:
        """List all active conversations."""
        async with self._lock:
            self._cleanup_expired()
            return [
                {
                    "id": conv.id,
                    "provider": conv.provider,
                    "turn_count": conv.turn_count,
                    "message_count": len(conv.messages),
                    "created_at": conv.created_at,
                    "last_active": conv.last_active,
                    "metadata": conv.metadata,
                }
                for conv in self._conversations.values()
            ]

    @property
    def size(self) -> int:
        return len(self._conversations)

    def _is_expired(self, conv: Conversation) -> bool:
        return (time.time() - conv.last_active) > self._ttl

    def _evict_if_needed(self) -> None:
        """Evict oldest conversations if at capacity. Must be called under lock."""
        self._cleanup_expired()
        while len(self._conversations) >= self._max:
            oldest_id = min(
                self._conversations,
                key=lambda cid: self._conversations[cid].last_active,
            )
            del self._conversations[oldest_id]
            logger.info("conversation evicted (LRU): %s", oldest_id)

    def _cleanup_expired(self) -> None:
        """Remove all expired conversations. Must be called under lock."""
        now = time.time()
        expired = [
            cid for cid, conv in self._conversations.items()
            if (now - conv.last_active) > self._ttl
        ]
        for cid in expired:
            del self._conversations[cid]
        if expired:
            logger.info("expired conversations cleaned: %d", len(expired))
