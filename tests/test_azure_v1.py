"""Tests for Azure OpenAI v1 API endpoint (/openai/v1/chat/completions).

Covers the new Azure AI Foundry v1 path format alongside the legacy
/openai/deployments/{id}/ path.
"""


import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import ModelBackend
from llm_katan.providers.azure_openai import AzureOpenAIProvider
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
    config = ServerConfig(model_name="test", served_model_name="azure-test", port=8000, providers=["azure_openai"])
    app = create_app(config)
    backend = MockBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()
    app.state.stats = PersistentStats()
    AzureOpenAIProvider(backend=backend).register_routes(app)
    return app


@pytest_asyncio.fixture
async def client():
    app = make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def azure_headers():
    return {"Content-Type": "application/json", "api-key": "test-key"}


V1_ENDPOINT = "/openai/v1/chat/completions"
LEGACY_ENDPOINT = "/openai/deployments/gpt-4/chat/completions"


class TestV1Endpoint:
    @pytest.mark.asyncio
    async def test_basic_request(self, client):
        resp = await client.post(
            V1_ENDPOINT,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
            headers=azure_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert "Response to: hello" in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_model_from_body(self, client):
        """v1 API gets model from request body, not URL."""
        resp = await client.post(
            V1_ENDPOINT,
            json={"model": "my-model", "messages": [{"role": "user", "content": "hi"}]},
            headers=azure_headers(),
        )
        assert resp.json()["model"] == "my-model"

    @pytest.mark.asyncio
    async def test_no_model_uses_default(self, client):
        """If no model in body, falls back to served_model_name."""
        resp = await client.post(
            V1_ENDPOINT,
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=azure_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "azure-test"

    @pytest.mark.asyncio
    async def test_no_api_version_required(self, client):
        """v1 API doesn't require api-version query param."""
        resp = await client.post(
            V1_ENDPOINT,
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            headers=azure_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_streaming(self, client):
        resp = await client.post(
            V1_ENDPOINT,
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            headers=azure_headers(),
        )
        assert "text/event-stream" in resp.headers["content-type"]
        lines = [line.strip() for line in resp.text.strip().split("\n") if line.strip()]
        assert lines[-1] == "data: [DONE]"

    @pytest.mark.asyncio
    async def test_content_filter_results(self, client):
        resp = await client.post(
            V1_ENDPOINT,
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            headers=azure_headers(),
        )
        assert "content_filter_results" in resp.json()["choices"][0]
        assert "prompt_filter_results" in resp.json()


class TestV1Auth:
    @pytest.mark.asyncio
    async def test_api_key_accepted(self, client):
        resp = await client.post(V1_ENDPOINT, json={"messages": [{"role": "user", "content": "hi"}]}, headers=azure_headers())
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_bearer_accepted(self, client):
        resp = await client.post(
            V1_ENDPOINT,
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer entra-id-token"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_no_auth_rejected(self, client):
        resp = await client.post(V1_ENDPOINT, json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_json_rejected(self, client):
        resp = await client.post(V1_ENDPOINT, content=b"not json", headers={**azure_headers(), "Content-Type": "application/json"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_messages_rejected(self, client):
        resp = await client.post(V1_ENDPOINT, json={"model": "gpt-4"}, headers=azure_headers())
        assert resp.status_code == 400


class TestLegacyStillWorks:
    @pytest.mark.asyncio
    async def test_legacy_endpoint(self, client):
        resp = await client.post(
            LEGACY_ENDPOINT,
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=azure_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "gpt-4"  # from URL path

    @pytest.mark.asyncio
    async def test_legacy_with_api_version(self, client):
        resp = await client.post(
            f"{LEGACY_ENDPOINT}?api-version=2024-10-21",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=azure_headers(),
        )
        assert resp.status_code == 200
