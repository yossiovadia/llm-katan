"""Tests for Vertex AI OpenAI-compatible endpoint.

POST /v1/projects/{project}/locations/{location}/endpoints/{endpoint}/chat/completions
"""

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import ModelBackend
from llm_katan.providers.vertexai import VertexAIProvider
from llm_katan.server import ServerMetrics, create_app
from llm_katan.stats import PersistentStats


class MockBackend(ModelBackend):
    async def load_model(self):
        pass

    async def _generate_text(self, messages, max_tokens, temperature):
        user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break
        return f"Response to: {user_msg}", 10, len(f"Response to: {user_msg}")


def make_app():
    config = ServerConfig(model_name="test", served_model_name="vertex-test", port=8000, providers=["vertexai"])
    app = create_app(config)
    backend = MockBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()
    app.state.stats = PersistentStats()
    VertexAIProvider(backend=backend).register_routes(app)
    return app


@pytest_asyncio.fixture
async def client():
    app = make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


ENDPOINT = "/v1/projects/my-project/locations/us-central1/endpoints/gemini-pro/chat/completions"


def vertex_headers():
    return {"Content-Type": "application/json", "Authorization": "Bearer ya29.test-token"}


class TestOpenAICompatEndpoint:
    @pytest.mark.asyncio
    async def test_basic_request(self, client):
        resp = await client.post(
            ENDPOINT,
            json={"model": "gemini-pro", "messages": [{"role": "user", "content": "hello"}]},
            headers=vertex_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        assert "Response to: hello" in data["choices"][0]["message"]["content"]
        assert data["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_model_from_body(self, client):
        resp = await client.post(
            ENDPOINT,
            json={"model": "gemini-1.5-flash", "messages": [{"role": "user", "content": "hi"}]},
            headers=vertex_headers(),
        )
        assert resp.json()["model"] == "gemini-1.5-flash"

    @pytest.mark.asyncio
    async def test_model_from_url_fallback(self, client):
        """If no model in body, uses endpoint name from URL."""
        resp = await client.post(
            ENDPOINT,
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=vertex_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "gemini-pro"  # from URL path

    @pytest.mark.asyncio
    async def test_usage_openai_format(self, client):
        resp = await client.post(
            ENDPOINT,
            json={"model": "gemini-pro", "messages": [{"role": "user", "content": "hi"}]},
            headers=vertex_headers(),
        )
        usage = resp.json()["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        # Must be OpenAI field names, NOT Vertex names
        assert "promptTokenCount" not in usage

    @pytest.mark.asyncio
    async def test_multi_turn(self, client):
        resp = await client.post(
            ENDPOINT,
            json={
                "model": "gemini-pro",
                "messages": [
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                    {"role": "user", "content": "how are you"},
                ],
            },
            headers=vertex_headers(),
        )
        assert resp.status_code == 200
        assert "how are you" in resp.json()["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_streaming(self, client):
        resp = await client.post(
            ENDPOINT,
            json={"model": "gemini-pro", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            headers=vertex_headers(),
        )
        assert "text/event-stream" in resp.headers["content-type"]
        lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
        assert lines[-1] == "data: [DONE]"

    @pytest.mark.asyncio
    async def test_stream_reassembles(self, client):
        body = {"model": "gemini-pro", "messages": [{"role": "user", "content": "hello"}]}

        resp = await client.post(ENDPOINT, json=body, headers=vertex_headers())
        non_stream_text = resp.json()["choices"][0]["message"]["content"]

        body["stream"] = True
        resp = await client.post(ENDPOINT, json=body, headers=vertex_headers())
        chunks = []
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line.removeprefix("data: "))
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    chunks.append(delta["content"])
        assert "".join(chunks) == non_stream_text

    @pytest.mark.asyncio
    async def test_different_project_location(self, client):
        resp = await client.post(
            "/v1/projects/other-project/locations/europe-west1/endpoints/my-model/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers=vertex_headers(),
        )
        assert resp.status_code == 200


class TestOpenAICompatAuth:
    @pytest.mark.asyncio
    async def test_bearer_accepted(self, client):
        resp = await client.post(ENDPOINT, json={"messages": [{"role": "user", "content": "hi"}]}, headers=vertex_headers())
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_no_auth_rejected(self, client):
        resp = await client.post(ENDPOINT, json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_json_rejected(self, client):
        resp = await client.post(ENDPOINT, content=b"not json", headers={**vertex_headers(), "Content-Type": "application/json"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_messages_rejected(self, client):
        resp = await client.post(ENDPOINT, json={"model": "gemini-pro"}, headers=vertex_headers())
        assert resp.status_code == 400


class TestNativeEndpointStillWorks:
    @pytest.mark.asyncio
    async def test_generate_content(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
            headers=vertex_headers(),
        )
        assert resp.status_code == 200
        assert "candidates" in resp.json()

    @pytest.mark.asyncio
    async def test_stream_generate_content(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
            headers=vertex_headers(),
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
