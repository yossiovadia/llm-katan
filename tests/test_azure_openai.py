"""Extensive tests for the Azure OpenAI provider.

Tests based on the official Azure OpenAI API spec at
learn.microsoft.com/en-us/azure/ai-services/openai/reference.
"""

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import ModelBackend
from llm_katan.providers.azure_openai import AzureOpenAIProvider
from llm_katan.server import ServerMetrics, create_app


class MockBackend(ModelBackend):
    async def load_model(self):
        pass

    async def _generate_text(self, messages, max_tokens, temperature):
        user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break
        generated = f"Response to: {user_msg}"
        return generated, 10, len(generated)


def make_app():
    config = ServerConfig(
        model_name="test-model",
        served_model_name="azure-test",
        port=8000,
        providers=["azure_openai"],
    )
    app = create_app(config)
    backend = MockBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()
    provider = AzureOpenAIProvider(backend=backend)
    provider.register_routes(app)
    return app


@pytest_asyncio.fixture
async def client():
    app = make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def azure_headers():
    return {
        "Content-Type": "application/json",
        "api-key": "test-azure-key",
    }


def base_request(**overrides):
    req = {
        "messages": [{"role": "user", "content": "hello"}],
    }
    req.update(overrides)
    return req


ENDPOINT = "/openai/deployments/gpt-4/chat/completions"
ENDPOINT_WITH_VERSION = "/openai/deployments/gpt-4/chat/completions?api-version=2024-10-21"


# ============================================================
# Request Format — URL Structure
# ============================================================

class TestRequestFormat:
    @pytest.mark.asyncio
    async def test_basic_request(self, client):
        resp = await client.post(ENDPOINT, json=base_request(), headers=azure_headers())
        assert resp.status_code == 200
        assert "Response to: hello" in resp.json()["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_with_api_version(self, client):
        resp = await client.post(ENDPOINT_WITH_VERSION, json=base_request(), headers=azure_headers())
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_deployment_id_in_response(self, client):
        """Azure uses deployment ID as model in response, not the model field from request."""
        resp = await client.post(ENDPOINT, json=base_request(model="some-other-model"), headers=azure_headers())
        assert resp.json()["model"] == "gpt-4"  # from URL, not request body

    @pytest.mark.asyncio
    async def test_different_deployment_id(self, client):
        resp = await client.post(
            "/openai/deployments/my-custom-deployment/chat/completions",
            json=base_request(),
            headers=azure_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "my-custom-deployment"

    @pytest.mark.asyncio
    async def test_model_field_optional(self, client):
        """In Azure, model field in body is optional — deployment ID is in URL."""
        resp = await client.post(ENDPOINT, json={"messages": [{"role": "user", "content": "hi"}]}, headers=azure_headers())
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_multi_turn(self, client):
        resp = await client.post(
            ENDPOINT,
            json=base_request(messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "how are you"},
            ]),
            headers=azure_headers(),
        )
        assert resp.status_code == 200
        assert "how are you" in resp.json()["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_with_params(self, client):
        resp = await client.post(
            ENDPOINT,
            json=base_request(max_tokens=100, temperature=0.5),
            headers=azure_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_messages(self, client):
        resp = await client.post(ENDPOINT, json={}, headers=azure_headers())
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "invalid_request"

    @pytest.mark.asyncio
    async def test_invalid_json(self, client):
        resp = await client.post(
            ENDPOINT,
            content=b"not json",
            headers={**azure_headers(), "Content-Type": "application/json"},
        )
        assert resp.status_code == 400


# ============================================================
# Response Format
# ============================================================

class TestResponseFormat:
    @pytest.mark.asyncio
    async def test_response_fields(self, client):
        resp = await client.post(ENDPOINT, json=base_request(), headers=azure_headers())
        data = resp.json()
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert isinstance(data["created"], int)
        assert isinstance(data["choices"], list)
        assert len(data["choices"]) == 1

    @pytest.mark.asyncio
    async def test_choice_structure(self, client):
        resp = await client.post(ENDPOINT, json=base_request(), headers=azure_headers())
        choice = resp.json()["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert choice["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_usage(self, client):
        resp = await client.post(ENDPOINT, json=base_request(), headers=azure_headers())
        usage = resp.json()["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    @pytest.mark.asyncio
    async def test_content_filter_results(self, client):
        """Azure-specific: content_filter_results on choices."""
        resp = await client.post(ENDPOINT, json=base_request(), headers=azure_headers())
        choice = resp.json()["choices"][0]
        assert "content_filter_results" in choice
        cfr = choice["content_filter_results"]
        for category in ("hate", "self_harm", "sexual", "violence"):
            assert category in cfr
            assert cfr[category]["filtered"] is False
            assert cfr[category]["severity"] == "safe"

    @pytest.mark.asyncio
    async def test_prompt_filter_results(self, client):
        """Azure-specific: prompt_filter_results at top level."""
        resp = await client.post(ENDPOINT, json=base_request(), headers=azure_headers())
        data = resp.json()
        assert "prompt_filter_results" in data
        assert data["prompt_filter_results"][0]["prompt_index"] == 0
        pfr = data["prompt_filter_results"][0]["content_filter_results"]
        for category in ("hate", "self_harm", "sexual", "violence"):
            assert category in pfr


# ============================================================
# Streaming
# ============================================================

class TestStreaming:
    def _parse_sse(self, text):
        chunks = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))
        return chunks

    @pytest.mark.asyncio
    async def test_stream_content_type(self, client):
        resp = await client.post(ENDPOINT, json=base_request(stream=True), headers=azure_headers())
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_stream_ends_with_done(self, client):
        resp = await client.post(ENDPOINT, json=base_request(stream=True), headers=azure_headers())
        lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
        assert lines[-1] == "data: [DONE]"

    @pytest.mark.asyncio
    async def test_stream_reassembles(self, client):
        body = base_request()
        resp = await client.post(ENDPOINT, json=body, headers=azure_headers())
        non_stream_text = resp.json()["choices"][0]["message"]["content"]

        body["stream"] = True
        resp = await client.post(ENDPOINT, json=body, headers=azure_headers())
        chunks = self._parse_sse(resp.text)
        stream_text = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        assert stream_text == non_stream_text

    @pytest.mark.asyncio
    async def test_stream_final_chunk(self, client):
        resp = await client.post(ENDPOINT, json=base_request(stream=True), headers=azure_headers())
        chunks = self._parse_sse(resp.text)
        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] == "stop"
        assert "usage" in last

    @pytest.mark.asyncio
    async def test_stream_consistent_id(self, client):
        resp = await client.post(ENDPOINT, json=base_request(stream=True), headers=azure_headers())
        chunks = self._parse_sse(resp.text)
        ids = {c["id"] for c in chunks}
        assert len(ids) == 1

    @pytest.mark.asyncio
    async def test_stream_model_is_deployment(self, client):
        resp = await client.post(ENDPOINT, json=base_request(stream=True), headers=azure_headers())
        chunks = self._parse_sse(resp.text)
        for c in chunks:
            assert c["model"] == "gpt-4"


# ============================================================
# Auth — api-key header
# ============================================================

class TestAuth:
    @pytest.mark.asyncio
    async def test_missing_api_key_rejected(self, client):
        resp = await client.post(
            ENDPOINT,
            json=base_request(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 401
        assert "api-key" in resp.json()["error"]["message"]
        assert resp.json()["error"]["code"] == "invalid_api_key"

    @pytest.mark.asyncio
    async def test_api_key_present_accepted(self, client):
        resp = await client.post(ENDPOINT, json=base_request(), headers=azure_headers())
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_any_api_key_value_accepted(self, client):
        resp = await client.post(
            ENDPOINT,
            json=base_request(),
            headers={"Content-Type": "application/json", "api-key": "literally-anything"},
        )
        assert resp.status_code == 200


# ============================================================
# Error Format
# ============================================================

class TestErrorFormat:
    @pytest.mark.asyncio
    async def test_error_structure(self, client):
        resp = await client.post(ENDPOINT, json={}, headers=azure_headers())
        data = resp.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert "type" in data["error"]
        assert "param" in data["error"]

    @pytest.mark.asyncio
    async def test_400_invalid_request(self, client):
        resp = await client.post(ENDPOINT, json={}, headers=azure_headers())
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "invalid_request"

    @pytest.mark.asyncio
    async def test_401_invalid_api_key(self, client):
        resp = await client.post(ENDPOINT, json=base_request(), headers={"Content-Type": "application/json"})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "invalid_api_key"
