"""Extensive tests for the Anthropic Messages API provider.

Tests are based on the official Anthropic API spec at
platform.claude.com/docs/en/api/messages.
"""

import asyncio
import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import ModelBackend
from llm_katan.providers.anthropic import AnthropicProvider
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


def make_app(require_auth=False):
    config = ServerConfig(
        model_name="test-model",
        served_model_name="claude-test",
        port=8000,
        providers=["anthropic"],
        require_auth=require_auth,
    )
    app = create_app(config)
    backend = MockBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()
    provider = AnthropicProvider(backend=backend, require_auth=require_auth)
    provider.register_routes(app)
    return app


@pytest_asyncio.fixture
async def client():
    app = make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def auth_client():
    app = make_app(require_auth=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def base_request(**overrides):
    """Build a valid Anthropic request, overriding specific fields."""
    req = {
        "model": "claude-test",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hello"}],
    }
    req.update(overrides)
    return req


def anthropic_headers():
    return {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }


# ============================================================
# Request Format Compliance
# ============================================================

class TestRequestFormat:
    @pytest.mark.asyncio
    async def test_basic_string_content(self, client):
        resp = await client.post("/v1/messages", json=base_request(), headers=anthropic_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "Response to: hello" in data["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_array_content_blocks(self, client):
        resp = await client.post(
            "/v1/messages",
            json=base_request(messages=[
                {"role": "user", "content": [{"type": "text", "text": "hello from array"}]}
            ]),
            headers=anthropic_headers(),
        )
        assert resp.status_code == 200
        assert "hello from array" in resp.json()["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_system_as_string(self, client):
        resp = await client.post(
            "/v1/messages",
            json=base_request(system="You are a pirate"),
            headers=anthropic_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_system_as_array(self, client):
        resp = await client.post(
            "/v1/messages",
            json=base_request(system=[{"type": "text", "text": "You are helpful"}]),
            headers=anthropic_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_multi_turn(self, client):
        resp = await client.post(
            "/v1/messages",
            json=base_request(messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "how are you"},
            ]),
            headers=anthropic_headers(),
        )
        assert resp.status_code == 200
        assert "how are you" in resp.json()["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_missing_model(self, client):
        body = base_request()
        del body["model"]
        resp = await client.post("/v1/messages", json=body, headers=anthropic_headers())
        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"
        assert "model" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_missing_max_tokens(self, client):
        body = base_request()
        del body["max_tokens"]
        resp = await client.post("/v1/messages", json=body, headers=anthropic_headers())
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["type"] == "invalid_request_error"
        assert "max_tokens" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_missing_messages(self, client):
        body = base_request()
        del body["messages"]
        resp = await client.post("/v1/messages", json=body, headers=anthropic_headers())
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_optional_params(self, client):
        resp = await client.post(
            "/v1/messages",
            json=base_request(temperature=0.5, top_p=0.9, top_k=40, stop_sequences=["END"]),
            headers=anthropic_headers(),
        )
        assert resp.status_code == 200


# ============================================================
# Response Format Compliance
# ============================================================

class TestResponseFormat:
    @pytest.mark.asyncio
    async def test_id_starts_with_msg(self, client):
        resp = await client.post("/v1/messages", json=base_request(), headers=anthropic_headers())
        assert resp.json()["id"].startswith("msg_")

    @pytest.mark.asyncio
    async def test_type_is_message(self, client):
        resp = await client.post("/v1/messages", json=base_request(), headers=anthropic_headers())
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_content_is_array_of_text_blocks(self, client):
        resp = await client.post("/v1/messages", json=base_request(), headers=anthropic_headers())
        content = resp.json()["content"]
        assert isinstance(content, list)
        assert len(content) >= 1
        assert content[0]["type"] == "text"
        assert isinstance(content[0]["text"], str)
        assert len(content[0]["text"]) > 0

    @pytest.mark.asyncio
    async def test_stop_reason(self, client):
        resp = await client.post("/v1/messages", json=base_request(), headers=anthropic_headers())
        data = resp.json()
        assert data["stop_reason"] == "end_turn"
        assert data["stop_sequence"] is None

    @pytest.mark.asyncio
    async def test_usage_fields(self, client):
        resp = await client.post("/v1/messages", json=base_request(), headers=anthropic_headers())
        usage = resp.json()["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        # Must NOT have OpenAI-style field names
        assert "prompt_tokens" not in usage
        assert "completion_tokens" not in usage
        assert usage["input_tokens"] > 0
        assert usage["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_model_echoed_back(self, client):
        resp = await client.post("/v1/messages", json=base_request(), headers=anthropic_headers())
        assert resp.json()["model"] == "claude-test"


# ============================================================
# Streaming Compliance
# ============================================================

class TestStreaming:
    def _parse_sse(self, text):
        """Parse SSE text into list of (event_type, data_dict) tuples."""
        events = []
        current_event = None
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("event: "):
                current_event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
                events.append((current_event, data))
                current_event = None
        return events

    @pytest.mark.asyncio
    async def test_event_sequence(self, client):
        resp = await client.post(
            "/v1/messages",
            json=base_request(stream=True),
            headers=anthropic_headers(),
        )
        events = self._parse_sse(resp.text)
        event_types = [e[0] for e in events]

        assert event_types[0] == "message_start"
        assert event_types[1] == "content_block_start"
        # Middle events are content_block_delta
        assert all(t == "content_block_delta" for t in event_types[2:-3])
        assert event_types[-3] == "content_block_stop"
        assert event_types[-2] == "message_delta"
        assert event_types[-1] == "message_stop"

    @pytest.mark.asyncio
    async def test_message_start_has_input_tokens(self, client):
        resp = await client.post("/v1/messages", json=base_request(stream=True), headers=anthropic_headers())
        events = self._parse_sse(resp.text)
        msg_start = events[0][1]
        assert msg_start["type"] == "message_start"
        assert msg_start["message"]["usage"]["input_tokens"] > 0
        assert msg_start["message"]["usage"]["output_tokens"] == 0
        assert msg_start["message"]["id"].startswith("msg_")
        assert msg_start["message"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_content_block_delta_format(self, client):
        resp = await client.post("/v1/messages", json=base_request(stream=True), headers=anthropic_headers())
        events = self._parse_sse(resp.text)
        deltas = [e[1] for e in events if e[0] == "content_block_delta"]
        assert len(deltas) > 0
        for d in deltas:
            assert d["type"] == "content_block_delta"
            assert d["index"] == 0
            assert d["delta"]["type"] == "text_delta"
            assert isinstance(d["delta"]["text"], str)

    @pytest.mark.asyncio
    async def test_message_delta_has_stop_reason_and_usage(self, client):
        resp = await client.post("/v1/messages", json=base_request(stream=True), headers=anthropic_headers())
        events = self._parse_sse(resp.text)
        msg_delta = [e[1] for e in events if e[0] == "message_delta"][0]
        assert msg_delta["type"] == "message_delta"
        assert msg_delta["delta"]["stop_reason"] == "end_turn"
        assert msg_delta["usage"]["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_stream_reassembles_to_same_content(self, client):
        body = base_request()

        # Non-streaming
        resp = await client.post("/v1/messages", json=body, headers=anthropic_headers())
        non_stream_text = resp.json()["content"][0]["text"]

        # Streaming
        body["stream"] = True
        resp = await client.post("/v1/messages", json=body, headers=anthropic_headers())
        events = self._parse_sse(resp.text)
        deltas = [e[1] for e in events if e[0] == "content_block_delta"]
        stream_text = "".join(d["delta"]["text"] for d in deltas)

        assert stream_text == non_stream_text

    @pytest.mark.asyncio
    async def test_stream_content_type(self, client):
        resp = await client.post("/v1/messages", json=base_request(stream=True), headers=anthropic_headers())
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_all_events_have_type_field(self, client):
        resp = await client.post("/v1/messages", json=base_request(stream=True), headers=anthropic_headers())
        events = self._parse_sse(resp.text)
        for event_name, data in events:
            assert "type" in data, f"Event {event_name} missing 'type' field"
            # The data type should match the event name
            assert data["type"] == event_name, (
                f"Event name {event_name!r} != data type {data['type']!r}"
            )


# ============================================================
# Auth
# ============================================================

class TestAuth:
    @pytest.mark.asyncio
    async def test_missing_api_key_when_required(self, auth_client):
        resp = await auth_client.post("/v1/messages", json=base_request(), headers=anthropic_headers())
        assert resp.status_code == 401
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "authentication_error"
        assert "x-api-key" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_api_key_present(self, auth_client):
        headers = {**anthropic_headers(), "x-api-key": "sk-ant-test-key"}
        resp = await auth_client.post("/v1/messages", json=base_request(), headers=headers)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_no_auth_required(self, client):
        # client fixture has require_auth=False, no x-api-key header
        resp = await client.post("/v1/messages", json=base_request(), headers=anthropic_headers())
        assert resp.status_code == 200


# ============================================================
# Error Format
# ============================================================

class TestErrorFormat:
    @pytest.mark.asyncio
    async def test_error_structure(self, client):
        body = base_request()
        del body["model"]
        resp = await client.post("/v1/messages", json=body, headers=anthropic_headers())
        data = resp.json()
        # Must have exactly this structure
        assert set(data.keys()) == {"type", "error"}
        assert data["type"] == "error"
        assert set(data["error"].keys()) == {"type", "message"}
        assert isinstance(data["error"]["message"], str)

    @pytest.mark.asyncio
    async def test_400_uses_invalid_request_error(self, client):
        body = base_request()
        del body["max_tokens"]
        resp = await client.post("/v1/messages", json=body, headers=anthropic_headers())
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"

    @pytest.mark.asyncio
    async def test_401_uses_authentication_error(self, auth_client):
        resp = await auth_client.post("/v1/messages", json=base_request(), headers=anthropic_headers())
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"
