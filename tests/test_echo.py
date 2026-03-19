"""Tests for the echo backend."""

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import EchoBackend
from llm_katan.providers.anthropic import AnthropicProvider
from llm_katan.providers.openai import OpenAIProvider
from llm_katan.server import ServerMetrics, create_app


def make_app(providers=None):
    providers = providers or ["openai"]
    config = ServerConfig(
        model_name="echo-model",
        served_model_name="echo-test",
        port=8000,
        backend="echo",
        providers=providers,
    )
    app = create_app(config)
    backend = EchoBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()

    for p in providers:
        if p == "openai":
            OpenAIProvider(backend=backend).register_routes(app)
        elif p == "anthropic":
            AnthropicProvider(backend=backend).register_routes(app)
    return app


def openai_headers():
    return {"Content-Type": "application/json", "Authorization": "Bearer sk-test"}


def anthropic_headers():
    return {
        "Content-Type": "application/json",
        "x-api-key": "sk-ant-test",
        "anthropic-version": "2023-06-01",
    }


class TestEchoOpenAI:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["openai"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_echo_response_contains_metadata(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "echo-test", "messages": [{"role": "user", "content": "hello world"}]},
            headers=openai_headers(),
        )
        assert resp.status_code == 200
        text = resp.json()["choices"][0]["message"]["content"]
        assert "[echo]" in text
        assert "model=echo-test" in text
        assert "host=" in text
        assert "time=" in text
        assert "User: hello world" in text

    @pytest.mark.asyncio
    async def test_echo_contains_request_params(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "echo-test",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 200,
                "temperature": 0.3,
            },
            headers=openai_headers(),
        )
        text = resp.json()["choices"][0]["message"]["content"]
        assert "max_tokens=200" in text
        assert "temperature=0.3" in text

    @pytest.mark.asyncio
    async def test_echo_shows_message_count(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "echo-test",
                "messages": [
                    {"role": "system", "content": "be helpful"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "how are you"},
                ],
            },
            headers=openai_headers(),
        )
        text = resp.json()["choices"][0]["message"]["content"]
        assert "messages=4" in text
        assert "User: how are you" in text

    @pytest.mark.asyncio
    async def test_echo_streaming(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "echo-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            headers=openai_headers(),
        )
        assert "text/event-stream" in resp.headers["content-type"]

        # Reassemble
        chunks = []
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line.removeprefix("data: "))
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    chunks.append(delta["content"])
        text = "".join(chunks)
        assert "[echo]" in text
        assert "User: hi" in text

    @pytest.mark.asyncio
    async def test_echo_has_usage(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "echo-test", "messages": [{"role": "user", "content": "hello"}]},
            headers=openai_headers(),
        )
        usage = resp.json()["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


class TestEchoAnthropic:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["anthropic"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_echo_anthropic_format(self, client):
        resp = await client.post(
            "/v1/messages",
            json={"model": "echo-test", "max_tokens": 100, "messages": [{"role": "user", "content": "hello"}]},
            headers=anthropic_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["id"].startswith("msg_")
        text = data["content"][0]["text"]
        assert "[echo]" in text
        assert "User: hello" in text

    @pytest.mark.asyncio
    async def test_echo_anthropic_usage(self, client):
        resp = await client.post(
            "/v1/messages",
            json={"model": "echo-test", "max_tokens": 100, "messages": [{"role": "user", "content": "hello"}]},
            headers=anthropic_headers(),
        )
        usage = resp.json()["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage

    @pytest.mark.asyncio
    async def test_echo_anthropic_streaming(self, client):
        resp = await client.post(
            "/v1/messages",
            json={"model": "echo-test", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}], "stream": True},
            headers=anthropic_headers(),
        )
        assert "text/event-stream" in resp.headers["content-type"]

        # Parse SSE and reassemble
        deltas = []
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
                if data.get("type") == "content_block_delta":
                    deltas.append(data["delta"]["text"])
        text = "".join(deltas)
        assert "[echo]" in text


class TestEchoBothProviders:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["openai", "anthropic"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_both_providers_work_with_echo(self, client):
        # OpenAI
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "echo-test", "messages": [{"role": "user", "content": "openai test"}]},
            headers=openai_headers(),
        )
        assert resp.status_code == 200
        assert "[echo]" in resp.json()["choices"][0]["message"]["content"]

        # Anthropic
        resp = await client.post(
            "/v1/messages",
            json={"model": "echo-test", "max_tokens": 100, "messages": [{"role": "user", "content": "anthropic test"}]},
            headers=anthropic_headers(),
        )
        assert resp.status_code == 200
        assert "[echo]" in resp.json()["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_health_shows_echo_backend(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert data["backend"] == "echo"
        assert data["providers"] == ["openai", "anthropic"]


class TestEchoConfig:
    def test_echo_backend_accepted(self):
        config = ServerConfig(model_name="test", backend="echo")
        assert config.backend == "echo"

    @pytest.mark.asyncio
    async def test_echo_no_torch_needed(self):
        """Echo backend should work without importing torch."""
        config = ServerConfig(model_name="test", backend="echo")
        backend = EchoBackend(config)
        await backend.load_model()
        text, pt, ct = await backend.generate_text(
            [{"role": "user", "content": "hi"}], 100, 0.7
        )
        assert "[echo]" in text
        assert pt > 0
        assert ct > 0
