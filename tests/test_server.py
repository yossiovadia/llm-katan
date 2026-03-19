"""Tests for llm-katan server using a mock backend."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import ModelBackend
from llm_katan.server import create_app


class MockBackend(ModelBackend):
    """Backend that returns canned responses without loading a real model."""

    async def load_model(self):
        pass

    async def _generate_text(self, messages, max_tokens, temperature):
        # Echo back the last user message
        user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break
        generated = f"Mock response to: {user_msg}"
        return generated, 10, len(generated.split())

    def get_model_info(self):
        return {
            "id": self.config.served_model_name,
            "object": "model",
            "created": 1234567890,
            "owned_by": "llm-katan",
        }


def create_test_app():
    config = ServerConfig(model_name="test-model", served_model_name="gpt-test", port=8000)
    app = create_app(config)

    # Override lifespan by injecting backend directly
    backend = MockBackend(config)
    app.state.backend = backend

    from llm_katan.server import ServerMetrics
    app.state.metrics = ServerMetrics()

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


@pytest.mark.asyncio
async def test_list_models(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "gpt-test"


@pytest.mark.asyncio
async def test_chat_completion(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "gpt-test"
    assert len(data["choices"]) == 1
    assert "Mock response to: hello" in data["choices"][0]["message"]["content"]
    assert data["usage"]["prompt_tokens"] == 10
    assert data["usage"]["total_tokens"] > 10


@pytest.mark.asyncio
async def test_chat_completion_streaming(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    data_lines = [l for l in lines if l.startswith("data: ")]

    # Should have content chunks + [DONE]
    assert len(data_lines) >= 2
    assert data_lines[-1] == "data: [DONE]"

    # First chunk should have delta content
    import json
    first = json.loads(data_lines[0].removeprefix("data: "))
    assert first["object"] == "chat.completion.chunk"
    assert "content" in first["choices"][0]["delta"]

    # Second-to-last chunk (before [DONE]) should have finish_reason=stop
    last_chunk = json.loads(data_lines[-2].removeprefix("data: "))
    assert last_chunk["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_metrics(client):
    # Make a request first
    await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
    )

    resp = await client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "llm_katan_requests_total" in text
    assert "llm_katan_prompt_tokens_total" in text
    assert "llm_katan_completion_tokens_total" in text


@pytest.mark.asyncio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "LLM Katan"
    assert data["model"] == "gpt-test"


@pytest.mark.asyncio
async def test_missing_messages(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test"},
    )
    assert resp.status_code == 422  # validation error
