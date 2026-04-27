"""Tests for multimodal (image) content and JSON mode across providers."""

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import EchoBackend
from llm_katan.providers.anthropic import AnthropicProvider
from llm_katan.providers.azure_openai import AzureOpenAIProvider
from llm_katan.providers.bedrock import BedrockProvider
from llm_katan.providers.openai import OpenAIProvider
from llm_katan.providers.vertexai import VertexAIProvider
from llm_katan.server import ServerMetrics, create_app
from llm_katan.stats import PersistentStats


def make_app(providers):
    config = ServerConfig(
        model_name="test-model", served_model_name="test-model",
        port=8000, backend="echo", providers=providers,
    )
    app = create_app(config)
    backend = EchoBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()
    app.state.stats = PersistentStats()
    provider_map = {
        "openai": OpenAIProvider, "anthropic": AnthropicProvider,
        "vertexai": VertexAIProvider, "bedrock": BedrockProvider,
        "azure_openai": AzureOpenAIProvider,
    }
    for p in providers:
        provider_map[p](backend=backend).register_routes(app)
    return app


TINY_PNG_DATA_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# ── OpenAI multimodal ──


class TestOpenAIMultimodal:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["openai"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_image_url_content(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URI}},
                    ],
                }],
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert "[image:image/png]" in content
        assert "What is in this image?" in content

    async def test_mixed_text_and_image(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First image:"},
                        {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URI}},
                        {"type": "text", "text": "Second image:"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
                    ],
                }],
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert "[image:image/png]" in content
        assert "[image:url]" in content

    async def test_plain_string_content_still_works(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert "hello" in content

    async def test_null_content_accepted(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "test", "arguments": "{}"},
                    }]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "result"},
                ],
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["finish_reason"] == "stop"


# ── Anthropic multimodal ──


class TestAnthropicMultimodal:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["anthropic"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_image_content_block(self, client):
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "test", "max_tokens": 100,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR..."},
                        },
                    ],
                }],
            },
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 200
        text = resp.json()["content"][0]["text"]
        assert "[image:image/png]" in text

    async def test_string_content_still_works(self, client):
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "test", "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 200
        text = resp.json()["content"][0]["text"]
        assert "hello" in text


# ── Vertex AI multimodal ──


class TestVertexAIMultimodal:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["vertexai"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_inline_data_image(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={
                "contents": [{
                    "role": "user",
                    "parts": [
                        {"text": "What is this?"},
                        {"inlineData": {"mimeType": "image/jpeg", "data": "base64data..."}},
                    ],
                }],
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        assert "[image:image/jpeg]" in text

    async def test_text_only_still_works(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        assert "hello" in text


# ── Bedrock multimodal ──


class TestBedrockMultimodal:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["bedrock"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    def _headers(self):
        return {
            "Authorization": (
                "AWS4-HMAC-SHA256 Credential=AKID/20240101/us-east-1"
                "/bedrock/aws4_request, SignedHeaders=host, Signature=abc"
            ),
            "x-amz-date": "20240101T000000Z",
        }

    async def test_image_content_block(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json={
                "messages": [{
                    "role": "user",
                    "content": [
                        {"text": "What is this?"},
                        {"image": {"source": {"format": "png", "bytes": "base64..."}}},
                    ],
                }],
            },
            headers=self._headers(),
        )
        assert resp.status_code == 200
        text = resp.json()["output"]["message"]["content"][0]["text"]
        assert "[image:png]" in text

    async def test_text_only_still_works(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json={"messages": [{"role": "user", "content": [{"text": "hello"}]}]},
            headers=self._headers(),
        )
        assert resp.status_code == 200
        text = resp.json()["output"]["message"]["content"][0]["text"]
        assert "hello" in text


# ── Azure OpenAI multimodal ──


class TestAzureMultimodal:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["azure_openai"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_image_url_content(self, client):
        resp = await client.post(
            "/openai/deployments/gpt-4/chat/completions",
            json={
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URI}},
                    ],
                }],
            },
            headers={"api-key": "test"},
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert "[image:image/png]" in content


# ── JSON mode — all providers ──


class TestJSONModeOpenAI:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["openai"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_json_object_response(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "give me json"}],
                "response_format": {"type": "json_object"},
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        assert "response" in parsed

    async def test_no_response_format_returns_plain_text(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer test"},
        )
        content = resp.json()["choices"][0]["message"]["content"]
        with pytest.raises(json.JSONDecodeError):
            json.loads(content)


class TestJSONModeAnthropic:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["anthropic"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_json_object_response(self, client):
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "test", "max_tokens": 100,
                "messages": [{"role": "user", "content": "give me json"}],
                "response_format": {"type": "json_object"},
            },
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 200
        text = resp.json()["content"][0]["text"]
        parsed = json.loads(text)
        assert "response" in parsed

    async def test_no_response_format_returns_plain_text(self, client):
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "test", "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
        )
        text = resp.json()["content"][0]["text"]
        with pytest.raises(json.JSONDecodeError):
            json.loads(text)


class TestJSONModeAzure:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["azure_openai"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_json_object_response(self, client):
        resp = await client.post(
            "/openai/deployments/gpt-4/chat/completions",
            json={
                "messages": [{"role": "user", "content": "give me json"}],
                "response_format": {"type": "json_object"},
            },
            headers={"api-key": "test"},
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        assert "response" in parsed


class TestJSONModeVertexNative:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["vertexai"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_response_format_json_object(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "give me json"}]}],
                "response_format": {"type": "json_object"},
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
        assert "response" in parsed

    async def test_response_mime_type_json(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "give me json"}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
        assert "response" in parsed

    async def test_no_response_format_returns_plain_text(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
            headers={"Authorization": "Bearer test"},
        )
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        with pytest.raises(json.JSONDecodeError):
            json.loads(text)


class TestJSONModeBedrock:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["bedrock"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    def _headers(self):
        return {
            "Authorization": (
                "AWS4-HMAC-SHA256 Credential=AKID/20240101/us-east-1"
                "/bedrock/aws4_request, SignedHeaders=host, Signature=abc"
            ),
            "x-amz-date": "20240101T000000Z",
        }

    async def test_response_format_json_object(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json={
                "messages": [{"role": "user", "content": [{"text": "give me json"}]}],
                "response_format": {"type": "json_object"},
            },
            headers=self._headers(),
        )
        assert resp.status_code == 200
        text = resp.json()["output"]["message"]["content"][0]["text"]
        parsed = json.loads(text)
        assert "response" in parsed

    async def test_no_response_format_returns_plain_text(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json={"messages": [{"role": "user", "content": [{"text": "hello"}]}]},
            headers=self._headers(),
        )
        text = resp.json()["output"]["message"]["content"][0]["text"]
        with pytest.raises(json.JSONDecodeError):
            json.loads(text)
