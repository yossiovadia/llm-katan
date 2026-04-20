"""
Tests for multi-turn conversation support.

Tests the ConversationStore directly and multi-turn flows
through each provider (OpenAI, Anthropic, Vertex AI, Bedrock, Azure OpenAI).
"""

import json

import httpx
import pytest
import pytest_asyncio

from llm_katan.config import ServerConfig
from llm_katan.conversations import ConversationStore
from llm_katan.model import ModelBackend
from llm_katan.providers.anthropic import AnthropicProvider
from llm_katan.providers.azure_openai import AzureOpenAIProvider
from llm_katan.providers.bedrock import BedrockProvider
from llm_katan.providers.openai import OpenAIProvider
from llm_katan.providers.vertexai import VertexAIProvider
from llm_katan.server import ServerMetrics, create_app
from llm_katan.stats import PersistentStats


class MockBackend(ModelBackend):
    """Backend that returns canned responses reflecting the message count."""

    async def load_model(self):
        pass

    async def _generate_text(self, messages, max_tokens, temperature):
        user_msgs = [m for m in messages if m["role"] == "user"]
        last_user = user_msgs[-1]["content"] if user_msgs else ""
        generated = f"Reply to turn {len(user_msgs)}: {last_user}"
        return generated, len(messages) * 5, len(generated.split())


def make_app(providers=None):
    providers = providers or ["openai"]
    config = ServerConfig(
        model_name="test-model",
        served_model_name="test-model",
        port=8000,
        backend="echo",
        providers=providers,
        enable_conversations=True,
        conversation_ttl=3600,
        max_conversations=100,
    )
    app = create_app(config)
    backend = MockBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()
    app.state.stats = PersistentStats()

    store = ConversationStore(ttl_seconds=3600, max_conversations=100)
    app.state.conversations = store

    for p in providers:
        if p == "openai":
            OpenAIProvider(backend=backend, conversations=store).register_routes(app)
        elif p == "anthropic":
            AnthropicProvider(backend=backend, conversations=store).register_routes(app)
        elif p == "vertexai":
            VertexAIProvider(backend=backend, conversations=store).register_routes(app)
        elif p == "bedrock":
            BedrockProvider(backend=backend, conversations=store).register_routes(app)
        elif p == "azure_openai":
            AzureOpenAIProvider(backend=backend, conversations=store).register_routes(app)
    return app


# -----------------------------------------------------------------------
# ConversationStore unit tests
# -----------------------------------------------------------------------


class TestConversationStore:

    @pytest.fixture
    def store(self):
        return ConversationStore(ttl_seconds=3600, max_conversations=10)

    @pytest.mark.asyncio
    async def test_create_returns_id(self, store):
        conv_id = await store.create(provider="openai")
        assert conv_id.startswith("conv_")
        assert store.size == 1

    @pytest.mark.asyncio
    async def test_create_with_system_prompt(self, store):
        conv_id = await store.create(provider="openai", system_prompt="You are helpful.")
        msgs = await store.get_messages(conv_id)
        assert len(msgs) == 1
        assert msgs[0] == {"role": "system", "content": "You are helpful."}

    @pytest.mark.asyncio
    async def test_append_and_get(self, store):
        conv_id = await store.create(provider="openai")
        await store.append(conv_id, "user", "Hello")
        await store.append(conv_id, "assistant", "Hi there")
        msgs = await store.get_messages(conv_id)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, store):
        result = await store.get("conv_nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_append_nonexistent_returns_false(self, store):
        result = await store.append("conv_nonexistent", "user", "Hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete(self, store):
        conv_id = await store.create(provider="openai")
        assert store.size == 1
        deleted = await store.delete(conv_id)
        assert deleted is True
        assert store.size == 0
        assert await store.get(conv_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, store):
        deleted = await store.delete("conv_nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_clear(self, store):
        await store.create(provider="openai")
        await store.create(provider="anthropic")
        count = await store.clear()
        assert count == 2
        assert store.size == 0

    @pytest.mark.asyncio
    async def test_list_conversations(self, store):
        await store.create(provider="openai")
        await store.create(provider="anthropic")
        convos = await store.list_conversations()
        assert len(convos) == 2
        providers = {c["provider"] for c in convos}
        assert providers == {"openai", "anthropic"}

    @pytest.mark.asyncio
    async def test_turn_count(self, store):
        conv_id = await store.create(provider="openai")
        await store.append(conv_id, "user", "Q1")
        await store.append(conv_id, "assistant", "A1")
        await store.append(conv_id, "user", "Q2")
        await store.append(conv_id, "assistant", "A2")
        conv = await store.get(conv_id)
        assert conv.turn_count == 2

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        store = ConversationStore(ttl_seconds=3600, max_conversations=2)
        id1 = await store.create(provider="openai")
        id2 = await store.create(provider="openai")
        id3 = await store.create(provider="openai")
        assert store.size == 2
        assert await store.get(id1) is None
        assert await store.get(id2) is not None
        assert await store.get(id3) is not None


# -----------------------------------------------------------------------
# OpenAI multi-turn
# -----------------------------------------------------------------------


class TestOpenAIMultiTurn:

    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["openai"])
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_first_turn_creates_conversation(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert data["conversation_id"].startswith("conv_")

    @pytest.mark.asyncio
    async def test_second_turn_uses_conversation(self, client):
        r1 = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
            headers={"Authorization": "Bearer test"},
        )
        conv_id = r1.json()["conversation_id"]

        r2 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Follow up"}],
                "conversation_id": conv_id,
            },
            headers={"Authorization": "Bearer test"},
        )
        data = r2.json()
        assert data["conversation_id"] == conv_id
        assert "turn 2" in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_multi_turn_grows_history(self, client):
        r1 = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Turn 1"}]},
            headers={"Authorization": "Bearer test"},
        )
        conv_id = r1.json()["conversation_id"]

        for i in range(2, 6):
            r = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": f"Turn {i}"}],
                    "conversation_id": conv_id,
                },
                headers={"Authorization": "Bearer test"},
            )
            assert r.status_code == 200
            assert f"turn {i}" in r.json()["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_without_conversation_id_still_works(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "One-shot"}]},
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        assert "conversation_id" in resp.json()

    @pytest.mark.asyncio
    async def test_streaming_includes_conversation_id(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        first_data = None
        for line in lines:
            if line.startswith("data: ") and line != "data: [DONE]":
                first_data = json.loads(line[6:])
                break
        assert first_data is not None
        assert "conversation_id" in first_data


# -----------------------------------------------------------------------
# Anthropic multi-turn
# -----------------------------------------------------------------------


class TestAnthropicMultiTurn:

    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["anthropic"])
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_first_turn_returns_conversation_id(self, client):
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "metadata" in data
        assert data["metadata"]["conversation_id"].startswith("conv_")

    @pytest.mark.asyncio
    async def test_multi_turn_via_metadata(self, client):
        r1 = await client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
        )
        conv_id = r1.json()["metadata"]["conversation_id"]

        r2 = await client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Follow up"}],
                "metadata": {"conversation_id": conv_id},
            },
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
        )
        data = r2.json()
        assert data["metadata"]["conversation_id"] == conv_id
        assert "turn 2" in data["content"][0]["text"]


# -----------------------------------------------------------------------
# Vertex AI multi-turn
# -----------------------------------------------------------------------


class TestVertexAIMultiTurn:

    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["vertexai"])
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_first_turn_returns_conversation_id(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "conversationId" in data
        assert data["conversationId"].startswith("conv_")

    @pytest.mark.asyncio
    async def test_multi_turn_via_body(self, client):
        r1 = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
            headers={"Authorization": "Bearer test"},
        )
        conv_id = r1.json()["conversationId"]

        r2 = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "Follow up"}]}],
                "conversation_id": conv_id,
            },
            headers={"Authorization": "Bearer test"},
        )
        data = r2.json()
        assert data["conversationId"] == conv_id


# -----------------------------------------------------------------------
# Bedrock multi-turn
# -----------------------------------------------------------------------


class TestBedrockMultiTurn:

    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["bedrock"])
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    def _headers(self):
        return {
            "Authorization": "AWS4-HMAC-SHA256 Credential=AKID/20240101/us-east-1/bedrock/aws4_request, SignedHeaders=host, Signature=abc",
            "x-amz-date": "20240101T000000Z",
        }

    @pytest.mark.asyncio
    async def test_converse_returns_session_id(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json={"messages": [{"role": "user", "content": [{"text": "Hello"}]}]},
            headers=self._headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "sessionId" in data
        assert data["sessionId"].startswith("conv_")

    @pytest.mark.asyncio
    async def test_converse_multi_turn(self, client):
        r1 = await client.post(
            "/model/anthropic.claude-v2/converse",
            json={"messages": [{"role": "user", "content": [{"text": "Hello"}]}]},
            headers=self._headers(),
        )
        session_id = r1.json()["sessionId"]

        r2 = await client.post(
            "/model/anthropic.claude-v2/converse",
            json={
                "messages": [{"role": "user", "content": [{"text": "Follow up"}]}],
                "sessionId": session_id,
            },
            headers=self._headers(),
        )
        data = r2.json()
        assert data["sessionId"] == session_id
        assert "turn 2" in data["output"]["message"]["content"][0]["text"]


# -----------------------------------------------------------------------
# Azure OpenAI multi-turn
# -----------------------------------------------------------------------


class TestAzureOpenAIMultiTurn:

    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["azure_openai"])
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_first_turn_returns_conversation_id(self, client):
        resp = await client.post(
            "/openai/deployments/gpt-4/chat/completions?api-version=2024-02-01",
            json={"messages": [{"role": "user", "content": "Hello"}]},
            headers={"api-key": "test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert data["conversation_id"].startswith("conv_")

    @pytest.mark.asyncio
    async def test_multi_turn(self, client):
        r1 = await client.post(
            "/openai/deployments/gpt-4/chat/completions?api-version=2024-02-01",
            json={"messages": [{"role": "user", "content": "Hello"}]},
            headers={"api-key": "test"},
        )
        conv_id = r1.json()["conversation_id"]

        r2 = await client.post(
            "/openai/deployments/gpt-4/chat/completions?api-version=2024-02-01",
            json={
                "messages": [{"role": "user", "content": "Follow up"}],
                "conversation_id": conv_id,
            },
            headers={"api-key": "test"},
        )
        data = r2.json()
        assert data["conversation_id"] == conv_id
        assert "turn 2" in data["choices"][0]["message"]["content"]


# -----------------------------------------------------------------------
# Conversation management endpoints
# -----------------------------------------------------------------------


class TestConversationEndpoints:

    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["openai"])
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_list_empty(self, client):
        resp = await client.get("/conversations")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    @pytest.mark.asyncio
    async def test_list_after_chat(self, client):
        await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer test"},
        )
        resp = await client.get("/conversations")
        assert resp.json()["count"] == 1

    @pytest.mark.asyncio
    async def test_get_conversation_detail(self, client):
        r1 = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer test"},
        )
        conv_id = r1.json()["conversation_id"]
        resp = await client.get(f"/conversations/{conv_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == conv_id
        assert len(data["messages"]) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_delete_conversation(self, client):
        r1 = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer test"},
        )
        conv_id = r1.json()["conversation_id"]
        resp = await client.delete(f"/conversations/{conv_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        resp = await client.get(f"/conversations/{conv_id}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_clear_all(self, client):
        for i in range(3):
            await client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": f"Msg {i}"}]},
                headers={"Authorization": "Bearer test"},
            )
        resp = await client.delete("/conversations")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 3

    @pytest.mark.asyncio
    async def test_health_shows_conversations(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert data["conversations"] is True
        assert "active_conversations" in data
