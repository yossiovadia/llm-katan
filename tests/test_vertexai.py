"""Extensive tests for the Vertex AI / Gemini API provider.

Tests are based on the official Gemini API spec at
ai.google.dev/api/generate-content.
"""

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import ModelBackend
from llm_katan.providers.vertexai import VertexAIProvider
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
        served_model_name="gemini-test",
        port=8000,
        providers=["vertexai"],
    )
    app = create_app(config)
    backend = MockBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()
    provider = VertexAIProvider(backend=backend)
    provider.register_routes(app)
    return app


@pytest_asyncio.fixture
async def client():
    app = make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def gemini_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": "Bearer ya29.test-token",
    }


def base_request(**overrides):
    req = {
        "contents": [
            {"role": "user", "parts": [{"text": "hello"}]}
        ],
    }
    req.update(overrides)
    return req


# ============================================================
# Request Format Compliance
# ============================================================

class TestRequestFormat:
    @pytest.mark.asyncio
    async def test_basic_text_content(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        assert resp.status_code == 200
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        assert "Response to: hello" in text

    @pytest.mark.asyncio
    async def test_multi_part_content(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(contents=[
                {"role": "user", "parts": [{"text": "part one"}, {"text": "part two"}]}
            ]),
            headers=gemini_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_system_instruction(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(
                systemInstruction={"parts": [{"text": "You are a pirate"}]}
            ),
            headers=gemini_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(contents=[
                {"role": "user", "parts": [{"text": "hello"}]},
                {"role": "model", "parts": [{"text": "hi there"}]},
                {"role": "user", "parts": [{"text": "how are you"}]},
            ]),
            headers=gemini_headers(),
        )
        assert resp.status_code == 200
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        assert "how are you" in text

    @pytest.mark.asyncio
    async def test_generation_config(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(generationConfig={
                "maxOutputTokens": 100,
                "temperature": 0.5,
                "topP": 0.9,
                "topK": 40,
            }),
            headers=gemini_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_contents(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={},
            headers=gemini_headers(),
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["status"] == "INVALID_ARGUMENT"

    @pytest.mark.asyncio
    async def test_v1_endpoint(self, client):
        """v1 (non-beta) endpoint should also work."""
        resp = await client.post(
            "/v1/models/gemini-pro:generateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_model_name_from_url(self, client):
        """Model name comes from URL path, not request body."""
        resp = await client.post(
            "/v1beta/models/gemini-1.5-flash:generateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        assert resp.status_code == 200


# ============================================================
# Response Format Compliance
# ============================================================

class TestResponseFormat:
    @pytest.mark.asyncio
    async def test_candidates_structure(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        data = resp.json()
        assert "candidates" in data
        assert isinstance(data["candidates"], list)
        assert len(data["candidates"]) == 1

    @pytest.mark.asyncio
    async def test_candidate_content(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        candidate = resp.json()["candidates"][0]
        assert candidate["content"]["role"] == "model"
        assert isinstance(candidate["content"]["parts"], list)
        assert "text" in candidate["content"]["parts"][0]
        assert len(candidate["content"]["parts"][0]["text"]) > 0

    @pytest.mark.asyncio
    async def test_finish_reason(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        assert resp.json()["candidates"][0]["finishReason"] == "STOP"

    @pytest.mark.asyncio
    async def test_safety_ratings(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        ratings = resp.json()["candidates"][0]["safetyRatings"]
        assert isinstance(ratings, list)
        assert len(ratings) > 0
        for rating in ratings:
            assert "category" in rating
            assert rating["category"].startswith("HARM_CATEGORY_")
            assert "probability" in rating

    @pytest.mark.asyncio
    async def test_usage_metadata(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        usage = resp.json()["usageMetadata"]
        assert "promptTokenCount" in usage
        assert "candidatesTokenCount" in usage
        assert "totalTokenCount" in usage
        assert usage["totalTokenCount"] == usage["promptTokenCount"] + usage["candidatesTokenCount"]
        assert usage["promptTokenCount"] > 0
        assert usage["candidatesTokenCount"] > 0
        # Must NOT have OpenAI-style field names
        assert "prompt_tokens" not in usage
        assert "completion_tokens" not in usage

    @pytest.mark.asyncio
    async def test_model_version(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        assert resp.json()["modelVersion"] == "gemini-test"

    @pytest.mark.asyncio
    async def test_candidate_index(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        assert resp.json()["candidates"][0]["index"] == 0


# ============================================================
# Streaming Compliance
# ============================================================

class TestStreaming:
    def _parse_sse(self, text):
        """Parse SSE data lines into list of dicts."""
        chunks = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                chunks.append(json.loads(line.removeprefix("data: ")))
        return chunks

    @pytest.mark.asyncio
    async def test_stream_endpoint(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:streamGenerateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_stream_chunks_are_valid_json(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:streamGenerateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        chunks = self._parse_sse(resp.text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "candidates" in chunk

    @pytest.mark.asyncio
    async def test_stream_reassembles_to_same_content(self, client):
        body = base_request()

        # Non-streaming
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=body,
            headers=gemini_headers(),
        )
        non_stream_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

        # Streaming
        resp = await client.post(
            "/v1beta/models/gemini-pro:streamGenerateContent",
            json=body,
            headers=gemini_headers(),
        )
        chunks = self._parse_sse(resp.text)
        stream_text = "".join(
            c["candidates"][0]["content"]["parts"][0]["text"] for c in chunks
        )
        assert stream_text == non_stream_text

    @pytest.mark.asyncio
    async def test_stream_last_chunk_has_finish_reason(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:streamGenerateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        chunks = self._parse_sse(resp.text)
        last = chunks[-1]
        assert last["candidates"][0]["finishReason"] == "STOP"

    @pytest.mark.asyncio
    async def test_stream_last_chunk_has_usage(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:streamGenerateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        chunks = self._parse_sse(resp.text)
        last = chunks[-1]
        assert "usageMetadata" in last
        assert last["usageMetadata"]["promptTokenCount"] > 0
        assert last["usageMetadata"]["candidatesTokenCount"] > 0

    @pytest.mark.asyncio
    async def test_stream_content_role_is_model(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:streamGenerateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        chunks = self._parse_sse(resp.text)
        for chunk in chunks:
            assert chunk["candidates"][0]["content"]["role"] == "model"

    @pytest.mark.asyncio
    async def test_v1_stream_endpoint(self, client):
        resp = await client.post(
            "/v1/models/gemini-pro:streamGenerateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


# ============================================================
# Auth
# ============================================================

class TestAuth:
    @pytest.mark.asyncio
    async def test_missing_all_auth_rejected(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 401
        data = resp.json()
        assert data["error"]["status"] == "UNAUTHENTICATED"

    @pytest.mark.asyncio
    async def test_bearer_token_accepted(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(),
            headers=gemini_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_api_key_query_param_accepted(self, client):
        """Gemini API supports ?key= query parameter for auth."""
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent?key=AIzaSyTestKey123",
            json=base_request(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_api_key_query_param_streaming(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:streamGenerateContent?key=AIzaSyTestKey123",
            json=base_request(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200


# ============================================================
# Error Format
# ============================================================

class TestErrorFormat:
    @pytest.mark.asyncio
    async def test_error_structure(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={},
            headers=gemini_headers(),
        )
        data = resp.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert "status" in data["error"]
        assert isinstance(data["error"]["code"], int)

    @pytest.mark.asyncio
    async def test_400_uses_invalid_argument(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={},
            headers=gemini_headers(),
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["status"] == "INVALID_ARGUMENT"

    @pytest.mark.asyncio
    async def test_401_uses_unauthenticated(self, client):
        resp = await client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=base_request(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 401
        assert resp.json()["error"]["status"] == "UNAUTHENTICATED"
