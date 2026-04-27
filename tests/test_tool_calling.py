"""Tests for tool calling support across all providers."""

import json

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


WEATHER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

WEATHER_TOOL_ANTHROPIC = {
    "name": "get_weather",
    "description": "Get weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}

WEATHER_TOOL_VERTEX = {
    "functionDeclarations": [{
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
        },
    }],
}

WEATHER_TOOL_BEDROCK = {
    "tools": [{
        "toolSpec": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                },
            },
        },
    }],
}


# ── OpenAI ──


class TestOpenAIToolCalling:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["openai"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_tool_call_response(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test", "messages": [{"role": "user", "content": "What is the weather in SF?"}],
                "tools": [WEATHER_TOOL_OPENAI],
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        assert data["choices"][0]["message"]["content"] is None
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["id"].startswith("call_")
        args = json.loads(tc["function"]["arguments"])
        assert "location" in args
        assert args["unit"] == "celsius"

    async def test_tool_call_has_usage(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test", "messages": [{"role": "user", "content": "weather"}],
                "tools": [WEATHER_TOOL_OPENAI],
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.json()["usage"]["prompt_tokens"] > 0

    async def test_tool_result_accepted(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "call_abc", "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
                    }]},
                    {"role": "tool", "tool_call_id": "call_abc", "content": "72F, sunny"},
                ],
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    async def test_no_tools_returns_text(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] is not None
        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    async def test_multiple_tools_picks_first(self, client):
        second_tool = {
            "type": "function",
            "function": {"name": "search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
        }
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test", "messages": [{"role": "user", "content": "test"}],
                "tools": [WEATHER_TOOL_OPENAI, second_tool],
            },
            headers={"Authorization": "Bearer test"},
        )
        tc = resp.json()["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"


# ── Anthropic ──


class TestAnthropicToolCalling:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["anthropic"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_tool_use_response(self, client):
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "test", "max_tokens": 100,
                "messages": [{"role": "user", "content": "weather in SF?"}],
                "tools": [WEATHER_TOOL_ANTHROPIC],
            },
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stop_reason"] == "tool_use"
        content = data["content"]
        tool_block = next(b for b in content if b["type"] == "tool_use")
        assert tool_block["name"] == "get_weather"
        assert tool_block["id"].startswith("toolu_")
        assert "location" in tool_block["input"]

    async def test_tool_result_accepted(self, client):
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "test", "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"location": "SF"}},
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "72F, sunny"},
                    ]},
                ],
            },
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 200
        assert resp.json()["stop_reason"] == "end_turn"

    async def test_no_tools_returns_text(self, client):
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "test", "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
        )
        assert resp.json()["stop_reason"] == "end_turn"
        assert resp.json()["content"][0]["type"] == "text"


# ── Vertex AI ──


class TestVertexAIToolCalling:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["vertexai"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_function_call_response(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "weather in SF?"}]}],
                "tools": [WEATHER_TOOL_VERTEX],
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        fc = parts[0].get("functionCall")
        assert fc is not None
        assert fc["name"] == "get_weather"
        assert "location" in fc["args"]

    async def test_function_response_accepted(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "weather?"}]},
                    {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"location": "SF"}}}]},
                    {"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"result": "72F"}}}]},
                ],
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200

    async def test_no_tools_returns_text(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
            headers={"Authorization": "Bearer test"},
        )
        assert "text" in resp.json()["candidates"][0]["content"]["parts"][0]


# ── Bedrock Converse ──


class TestBedrockToolCalling:
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

    async def test_tool_use_response(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json={
                "messages": [{"role": "user", "content": [{"text": "weather in SF?"}]}],
                "toolConfig": WEATHER_TOOL_BEDROCK,
            },
            headers=self._headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stopReason"] == "tool_use"
        content = data["output"]["message"]["content"]
        tool_block = next(b for b in content if "toolUse" in b)
        assert tool_block["toolUse"]["name"] == "get_weather"
        assert tool_block["toolUse"]["toolUseId"].startswith("tooluse_")
        assert "location" in tool_block["toolUse"]["input"]

    async def test_tool_result_accepted(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json={
                "messages": [
                    {"role": "user", "content": [{"text": "weather?"}]},
                    {"role": "assistant", "content": [
                        {"toolUse": {"toolUseId": "tu_abc", "name": "get_weather", "input": {"location": "SF"}}},
                    ]},
                    {"role": "user", "content": [
                        {"toolResult": {"toolUseId": "tu_abc", "content": [{"text": "72F"}]}},
                    ]},
                ],
            },
            headers=self._headers(),
        )
        assert resp.status_code == 200

    async def test_no_tools_returns_text(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json={"messages": [{"role": "user", "content": [{"text": "hello"}]}]},
            headers=self._headers(),
        )
        assert "text" in resp.json()["output"]["message"]["content"][0]
        assert resp.json()["stopReason"] == "end_turn"


# ── Azure OpenAI ──


class TestAzureToolCalling:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["azure_openai"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_tool_call_response(self, client):
        resp = await client.post(
            "/openai/deployments/gpt-4/chat/completions",
            json={
                "messages": [{"role": "user", "content": "weather in SF?"}],
                "tools": [WEATHER_TOOL_OPENAI],
            },
            headers={"api-key": "test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"

    async def test_azure_filters_on_tool_response(self, client):
        resp = await client.post(
            "/openai/deployments/gpt-4/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "tools": [WEATHER_TOOL_OPENAI],
            },
            headers={"api-key": "test"},
        )
        data = resp.json()
        assert "prompt_filter_results" in data
        assert "content_filter_results" in data["choices"][0]

    async def test_no_tools_returns_text(self, client):
        resp = await client.post(
            "/openai/deployments/gpt-4/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"api-key": "test"},
        )
        assert resp.json()["choices"][0]["finish_reason"] == "stop"
