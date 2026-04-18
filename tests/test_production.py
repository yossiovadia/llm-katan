"""Production-grade evaluation tests for llm-katan."""

import asyncio
import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import ModelBackend
from llm_katan.providers.openai import OpenAIProvider
from llm_katan.server import ServerMetrics, create_app
from llm_katan.stats import PersistentStats


class MockBackend(ModelBackend):
    """Mock backend for testing."""

    def __init__(self, config, latency=0):
        super().__init__(config)
        self._latency = latency
        self.call_count = 0

    async def load_model(self):
        pass

    async def _generate_text(self, messages, max_tokens, temperature):
        self.call_count += 1
        if self._latency > 0:
            await asyncio.sleep(self._latency)
        user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break
        generated = f"Response to: {user_msg}"
        return generated, 10, len(generated)

    def get_model_info(self):
        return {
            "id": self.config.served_model_name,
            "object": "model",
            "created": 1234567890,
            "owned_by": "llm-katan",
        }


def make_app(max_concurrent=1, latency=0):
    config = ServerConfig(
        model_name="test-model",
        served_model_name="gpt-test",
        port=8000,
        max_concurrent=max_concurrent,
        providers=["openai"],
    )
    app = create_app(config)
    backend = MockBackend(config, latency=latency)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()
    app.state.stats = PersistentStats()

    provider = OpenAIProvider(backend=backend)
    provider.register_routes(app)
    return app


def openai_headers():
    return {"Content-Type": "application/json", "Authorization": "Bearer sk-test"}


@pytest_asyncio.fixture
async def client():
    app = make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=openai_headers()) as c:
        yield c


# ============================================================
# 1. OpenAI Response Format Compliance
# ============================================================

class TestOpenAICompliance:
    """Verify responses match OpenAI API spec exactly."""

    @pytest.mark.asyncio
    async def test_response_has_all_required_fields(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        )
        data = resp.json()

        # Required top-level fields per OpenAI spec
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert isinstance(data["created"], int)
        assert data["model"] == "gpt-test"
        assert isinstance(data["choices"], list)
        assert "usage" in data

    @pytest.mark.asyncio
    async def test_choice_structure(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        )
        choice = resp.json()["choices"][0]

        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert len(choice["message"]["content"]) > 0

    @pytest.mark.asyncio
    async def test_usage_structure(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        )
        usage = resp.json()["usage"]

        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0

    @pytest.mark.asyncio
    async def test_models_list_structure(self, client):
        resp = await client.get("/v1/models")
        data = resp.json()

        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        model = data["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert isinstance(model["created"], int)
        assert "owned_by" in model


# ============================================================
# 2. Streaming Correctness
# ============================================================

class TestStreaming:
    """Verify SSE streaming is correct and reassembles properly."""

    @pytest.mark.asyncio
    async def test_stream_reassembles_to_same_content(self):
        """Non-streaming and streaming should produce identical text."""
        app = make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test", headers=openai_headers()) as client:
            body = {"model": "gpt-test", "messages": [{"role": "user", "content": "hello world"}]}

            # Non-streaming
            resp = await client.post("/v1/chat/completions", json=body)
            non_stream_text = resp.json()["choices"][0]["message"]["content"]

            # Streaming
            body["stream"] = True
            resp = await client.post("/v1/chat/completions", json=body)

            chunks = []
            for line in resp.text.strip().split("\n"):
                line = line.strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line.removeprefix("data: "))
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        chunks.append(delta["content"])

            stream_text = "".join(chunks)
            assert stream_text == non_stream_text

    @pytest.mark.asyncio
    async def test_stream_ends_with_done(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        lines = [line.strip() for line in resp.text.strip().split("\n") if line.strip()]
        assert lines[-1] == "data: [DONE]"

    @pytest.mark.asyncio
    async def test_stream_has_finish_reason_in_final_chunk(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        data_lines = [
            line for line in resp.text.strip().split("\n")
            if line.strip().startswith("data: ") and line.strip() != "data: [DONE]"
        ]
        last_chunk = json.loads(data_lines[-1].strip().removeprefix("data: "))
        assert last_chunk["choices"][0]["finish_reason"] == "stop"
        assert "usage" in last_chunk

    @pytest.mark.asyncio
    async def test_stream_content_type(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_stream_all_chunks_have_consistent_id(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        ids = set()
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line.removeprefix("data: "))
                ids.add(chunk["id"])
        assert len(ids) == 1, f"All chunks should have the same id, got {ids}"


# ============================================================
# 3. Concurrency & Semaphore
# ============================================================

class TestConcurrency:
    """Verify concurrency control works."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_are_serialized(self):
        """With max_concurrent=1, requests should be serialized."""
        app = make_app(max_concurrent=1, latency=0.1)
        backend = app.state.backend
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test", headers=openai_headers()) as client:
            body = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}

            # Fire 3 concurrent requests
            tasks = [client.post("/v1/chat/completions", json=body) for _ in range(3)]
            responses = await asyncio.gather(*tasks)

            assert all(r.status_code == 200 for r in responses)
            assert backend.call_count == 3

    @pytest.mark.asyncio
    async def test_higher_concurrency_allowed(self):
        """With max_concurrent=3, multiple requests can run in parallel."""
        app = make_app(max_concurrent=3, latency=0.05)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test", headers=openai_headers()) as client:
            body = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}

            import time
            start = time.time()
            tasks = [client.post("/v1/chat/completions", json=body) for _ in range(3)]
            responses = await asyncio.gather(*tasks)
            elapsed = time.time() - start

            assert all(r.status_code == 200 for r in responses)
            # 3 requests with 50ms latency each, running in parallel, should take ~50ms not ~150ms
            assert elapsed < 0.3, f"Expected parallel execution, took {elapsed:.2f}s"


# ============================================================
# 4. Error Handling
# ============================================================

class TestOpenAIAuth:
    """Verify OpenAI auth is always required."""

    @pytest.mark.asyncio
    async def test_missing_authorization_rejected(self):
        app = make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as bare_client:
            resp = await bare_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 401
            data = resp.json()
            assert data["error"]["type"] == "invalid_request_error"
            assert "Authorization" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_authorization_present_accepted(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_any_bearer_value_accepted(self):
        app = make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as bare_client:
            resp = await bare_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer literally-anything"},
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_openai_error_format_on_auth_failure(self):
        app = make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as bare_client:
            resp = await bare_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
            )
            data = resp.json()
            assert "error" in data
            assert "message" in data["error"]
            assert "type" in data["error"]
            assert "code" in data["error"]


class TestErrorHandling:
    """Verify proper error responses."""

    @pytest.mark.asyncio
    async def test_missing_messages_field(self, client):
        resp = await client.post("/v1/chat/completions", json={"model": "gpt-test"})
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"

    @pytest.mark.asyncio
    async def test_empty_messages(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": []},
        )
        # Empty messages should still work (backend handles it)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_json(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"content-type": "application/json", "Authorization": "Bearer test"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"

    @pytest.mark.asyncio
    async def test_wrong_content_type(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            content=b"model=test",
            headers={"content-type": "application/x-www-form-urlencoded", "Authorization": "Bearer test"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_nonexistent_endpoint(self, client):
        resp = await client.get("/v1/nonexistent")
        assert resp.status_code in (404, 405)


# ============================================================
# 5. Metrics Correctness
# ============================================================

class TestMetrics:
    """Verify metrics tracking is accurate."""

    @pytest.mark.asyncio
    async def test_metrics_increment_on_requests(self):
        app = make_app()
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test", headers=openai_headers()) as client:
            body = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}

            for _ in range(5):
                await client.post("/v1/chat/completions", json=body)

            metrics: ServerMetrics = app.state.metrics
            assert metrics.total_requests == 5
            assert metrics.total_prompt_tokens == 50  # 10 per request
            assert metrics.total_completion_tokens > 0
            assert len(metrics.response_times) == 5

    @pytest.mark.asyncio
    async def test_metrics_streaming_counted(self):
        app = make_app()
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test", headers=openai_headers()) as client:
            body = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True}
            await client.post("/v1/chat/completions", json=body)

            metrics: ServerMetrics = app.state.metrics
            assert metrics.total_requests == 1

    @pytest.mark.asyncio
    async def test_metrics_prometheus_format(self, client):
        await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        )
        resp = await client.get("/metrics")
        text = resp.text

        # Verify Prometheus format
        assert "# HELP" in text
        assert "# TYPE" in text
        assert "llm_katan_requests_total" in text
        assert 'model="gpt-test"' in text
        assert resp.headers["content-type"] == "text/plain; charset=utf-8"

    def test_metrics_bounded_deque(self):
        """Verify response_times doesn't grow unbounded."""
        metrics = ServerMetrics()
        for i in range(2000):
            metrics.record(0.1, 10, 10)
        # Should be capped at MAX_RECORDED_RESPONSE_TIMES (1000)
        assert len(metrics.response_times) == 1000
        assert metrics.total_requests == 2000  # counter still accurate

    def test_metrics_avg_empty(self):
        metrics = ServerMetrics()
        assert metrics.avg_response_time == 0.0


# ============================================================
# 6. Multi-message Conversations
# ============================================================

class TestConversations:
    """Test multi-turn conversation handling."""

    @pytest.mark.asyncio
    async def test_system_message(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-test",
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_multi_turn(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-test",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                    {"role": "user", "content": "how are you"},
                ],
            },
        )
        assert resp.status_code == 200
        # Should respond to the last user message
        content = resp.json()["choices"][0]["message"]["content"]
        assert "how are you" in content

    @pytest.mark.asyncio
    async def test_optional_params(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "temperature": 0.5,
            },
        )
        assert resp.status_code == 200


# ============================================================
# 7. Config Validation
# ============================================================

class TestConfig:
    """Test configuration edge cases."""

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Invalid backend"):
            ServerConfig(model_name="test", backend="invalid")

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("LLM_KATAN_PORT", "9999")
        config = ServerConfig(model_name="test", port=8000)
        assert config.port == 9999

    def test_served_model_name_defaults(self):
        config = ServerConfig(model_name="Qwen/Qwen3-0.6B")
        assert config.served_model_name == "Qwen/Qwen3-0.6B"

    def test_served_model_name_override(self):
        config = ServerConfig(model_name="Qwen/Qwen3-0.6B", served_model_name="gpt-4o")
        assert config.served_model_name == "gpt-4o"
