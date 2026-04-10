"""Tests for API key validation (--validate-keys).

Validates that when key validation is enabled:
- Correct keys are accepted
- Wrong keys are rejected with error showing the expected key
- Missing headers still rejected
- Each provider validates its own key independently
- Default keys work out of the box
- CLI overrides work
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import DEFAULT_API_KEYS, ServerConfig
from llm_katan.model import ModelBackend
from llm_katan.providers.anthropic import AnthropicProvider
from llm_katan.providers.azure_openai import AzureOpenAIProvider
from llm_katan.providers.bedrock import BedrockProvider
from llm_katan.providers.openai import OpenAIProvider
from llm_katan.providers.vertexai import VertexAIProvider
from llm_katan.server import ServerMetrics, create_app


class MockBackend(ModelBackend):
    async def load_model(self):
        pass

    async def _generate_text(self, messages, max_tokens, temperature):
        return "ok", 5, 2


def make_app(providers, validate_keys=True, api_keys=None):
    config = ServerConfig(
        model_name="test",
        served_model_name="test",
        port=8000,
        providers=providers,
        validate_keys=validate_keys,
        api_keys=api_keys or {},
    )
    app = create_app(config)
    backend = MockBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()

    provider_map = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "vertexai": VertexAIProvider,
        "bedrock": BedrockProvider,
        "azure_openai": AzureOpenAIProvider,
    }
    for p in providers:
        cls = provider_map[p]
        expected_key = config.get_expected_key(p)
        provider = cls(backend=backend, expected_key=expected_key)
        provider.register_routes(app)

    return app


# ============================================================
# Config tests
# ============================================================

class TestConfig:
    def test_defaults_populated_when_validate_enabled(self):
        config = ServerConfig(model_name="test", validate_keys=True)
        assert config.api_keys["openai"] == DEFAULT_API_KEYS["openai"]
        assert config.api_keys["anthropic"] == DEFAULT_API_KEYS["anthropic"]

    def test_overrides_replace_defaults(self):
        config = ServerConfig(model_name="test", validate_keys=True, api_keys={"openai": "my-custom"})
        assert config.api_keys["openai"] == "my-custom"
        assert config.api_keys["anthropic"] == DEFAULT_API_KEYS["anthropic"]  # default kept

    def test_no_keys_when_validation_disabled(self):
        config = ServerConfig(model_name="test", validate_keys=False)
        assert config.get_expected_key("openai") is None

    def test_get_expected_key(self):
        config = ServerConfig(model_name="test", validate_keys=True)
        assert config.get_expected_key("openai") == DEFAULT_API_KEYS["openai"]
        assert config.get_expected_key("nonexistent") is None


# ============================================================
# OpenAI key validation
# ============================================================

class TestOpenAIKeyValidation:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["openai"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_correct_key_accepted(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {DEFAULT_API_KEYS['openai']}"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_wrong_key_rejected_with_hint(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401
        msg = resp.json()["error"]["message"]
        assert "wrong-key" in msg
        assert DEFAULT_API_KEYS["openai"] in msg

    @pytest.mark.asyncio
    async def test_missing_header_still_rejected(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401


# ============================================================
# Anthropic key validation
# ============================================================

class TestAnthropicKeyValidation:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["anthropic"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_correct_key_accepted(self, client):
        resp = await client.post(
            "/v1/messages",
            json={"model": "test", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": DEFAULT_API_KEYS["anthropic"], "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_wrong_key_rejected_with_hint(self, client):
        resp = await client.post(
            "/v1/messages",
            json={"model": "test", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "bad-key", "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 401
        msg = resp.json()["error"]["message"]
        assert "bad-key" in msg
        assert DEFAULT_API_KEYS["anthropic"] in msg


# ============================================================
# Azure OpenAI key validation
# ============================================================

class TestAzureKeyValidation:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["azure_openai"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_correct_api_key_accepted(self, client):
        resp = await client.post(
            "/openai/deployments/gpt-4/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"api-key": DEFAULT_API_KEYS["azure_openai"]},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_correct_bearer_accepted(self, client):
        resp = await client.post(
            "/openai/deployments/gpt-4/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {DEFAULT_API_KEYS['azure_openai']}"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_wrong_key_rejected_with_hint(self, client):
        resp = await client.post(
            "/openai/deployments/gpt-4/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"api-key": "nope"},
        )
        assert resp.status_code == 401
        msg = resp.json()["error"]["message"]
        assert "nope" in msg
        assert DEFAULT_API_KEYS["azure_openai"] in msg


# ============================================================
# Bedrock key validation
# ============================================================

class TestBedrockKeyValidation:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["bedrock"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_correct_sigv4_key_accepted(self, client):
        resp = await client.post(
            "/model/test/converse",
            json={"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
            headers={
                "Authorization": f"AWS4-HMAC-SHA256 Credential={DEFAULT_API_KEYS['bedrock']}/20260326/us-east-1/bedrock/aws4_request, SignedHeaders=host;x-amz-date, Signature=abc",
                "x-amz-date": "20260326T120000Z",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_correct_bearer_key_accepted(self, client):
        resp = await client.post(
            "/model/test/converse",
            json={"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
            headers={"Authorization": f"Bearer {DEFAULT_API_KEYS['bedrock']}"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_wrong_sigv4_key_rejected_with_hint(self, client):
        resp = await client.post(
            "/model/test/converse",
            json={"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
            headers={
                "Authorization": "AWS4-HMAC-SHA256 Credential=WRONG_KEY/20260326/us-east-1/bedrock/aws4_request, SignedHeaders=host;x-amz-date, Signature=abc",
                "x-amz-date": "20260326T120000Z",
            },
        )
        assert resp.status_code == 401
        msg = resp.json()["message"]
        assert "WRONG_KEY" in msg
        assert DEFAULT_API_KEYS["bedrock"] in msg


# ============================================================
# Vertex AI key validation
# ============================================================

class TestVertexKeyValidation:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["vertexai"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_correct_bearer_accepted(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            headers={"Authorization": f"Bearer {DEFAULT_API_KEYS['vertexai']}"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_correct_query_key_accepted(self, client):
        resp = await client.post(
            f"/v1beta/models/gemini-pro:generateContent?key={DEFAULT_API_KEYS['vertexai']}",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_wrong_bearer_rejected_with_hint(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401
        msg = resp.json()["error"]["message"]
        assert "wrong-token" in msg
        assert DEFAULT_API_KEYS["vertexai"] in msg

    @pytest.mark.asyncio
    async def test_wrong_query_key_rejected_with_hint(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent?key=wrong-key",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 401
        msg = resp.json()["error"]["message"]
        assert "wrong-key" in msg
        assert DEFAULT_API_KEYS["vertexai"] in msg


# ============================================================
# Custom key overrides
# ============================================================

class TestCustomKeyOverrides:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["openai", "anthropic"], api_keys={"openai": "my-custom-key"})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_custom_key_accepted(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer my-custom-key"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_default_key_rejected_when_overridden(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {DEFAULT_API_KEYS['openai']}"},
        )
        assert resp.status_code == 401
        assert "my-custom-key" in resp.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_non_overridden_provider_uses_default(self, client):
        resp = await client.post(
            "/v1/messages",
            json={"model": "test", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": DEFAULT_API_KEYS["anthropic"], "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 200


# ============================================================
# Validation disabled (backward compat)
# ============================================================

class TestValidationDisabled:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(["openai", "anthropic"], validate_keys=False)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_any_key_accepted_openai(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer literally-anything"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_any_key_accepted_anthropic(self, client):
        resp = await client.post(
            "/v1/messages",
            json={"model": "test", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "literally-anything", "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_header_still_rejected(self, client):
        """Even without key validation, missing headers are rejected."""
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401
