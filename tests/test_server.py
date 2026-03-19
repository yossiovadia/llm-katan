"""Tests for llm-katan server core functionality."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import ModelBackend
from llm_katan.providers.openai import OpenAIProvider
from llm_katan.server import ServerMetrics, create_app


class MockBackend(ModelBackend):
    """Backend that returns canned responses without loading a real model."""

    async def load_model(self):
        pass

    async def _generate_text(self, messages, max_tokens, temperature):
        user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break
        generated = f"Mock response to: {user_msg}"
        return generated, 10, len(generated.split())


def create_test_app(require_auth=False):
    config = ServerConfig(
        model_name="test-model",
        served_model_name="gpt-test",
        port=8000,
        providers=["openai"],
        require_auth=require_auth,
    )
    app = create_app(config)
    backend = MockBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()

    provider = OpenAIProvider(backend=backend, require_auth=require_auth)
    provider.register_routes(app)
    return app


@pytest.fixture
def app():
    return create_test_app()


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "gpt-test"
    assert "providers" in data


@pytest.mark.asyncio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "LLM Katan"
    assert "providers" in data


@pytest.mark.asyncio
async def test_list_models(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "gpt-test"


@pytest.mark.asyncio
async def test_chat_completion(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert "Mock response to: hello" in data["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_chat_completion_streaming(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}], "stream": True},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    lines = resp.text.strip().split("\n")
    data_lines = [l for l in lines if l.startswith("data: ")]
    assert data_lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_metrics(client):
    await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
    )
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "llm_katan_requests_total" in resp.text


@pytest.mark.asyncio
async def test_missing_messages(client):
    resp = await client.post("/v1/chat/completions", json={"model": "gpt-test"})
    assert resp.status_code == 422
