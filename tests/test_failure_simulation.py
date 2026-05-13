"""Tests for failure simulation features (echo backend only)."""

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import EchoBackend, SimulatedError
from llm_katan.providers.anthropic import AnthropicProvider
from llm_katan.providers.azure_openai import AzureOpenAIProvider
from llm_katan.providers.bedrock import BedrockProvider
from llm_katan.providers.openai import OpenAIProvider
from llm_katan.providers.vertexai import VertexAIProvider
from llm_katan.server import ServerMetrics, create_app
from llm_katan.stats import PersistentStats


def make_app(providers=None, **config_kwargs):
    providers = providers or ["openai"]
    config = ServerConfig(
        model_name="test-model",
        served_model_name="test-model",
        port=8000,
        backend="echo",
        providers=providers,
        **config_kwargs,
    )
    app = create_app(config)
    backend = EchoBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()
    app.state.stats = PersistentStats()

    provider_map = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "vertexai": VertexAIProvider,
        "bedrock": BedrockProvider,
        "azure_openai": AzureOpenAIProvider,
    }
    for p in providers:
        provider_map[p](backend=backend).register_routes(app)
    return app


def openai_headers():
    return {"Content-Type": "application/json", "Authorization": "Bearer sk-test"}


def openai_body():
    return {"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}


def anthropic_headers():
    return {"Content-Type": "application/json", "x-api-key": "test", "anthropic-version": "2023-06-01"}


def anthropic_body():
    return {"model": "test", "max_tokens": 100, "messages": [{"role": "user", "content": "hello"}]}


def bedrock_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": (
            "AWS4-HMAC-SHA256 Credential=AKID/20240101/us-east-1"
            "/bedrock/aws4_request, SignedHeaders=host, Signature=abc"
        ),
        "x-amz-date": "20240101T000000Z",
    }


def bedrock_body():
    return {"messages": [{"role": "user", "content": [{"text": "hello"}]}]}


def azure_headers():
    return {"Content-Type": "application/json", "api-key": "test"}


def azure_body():
    return {"messages": [{"role": "user", "content": "hello"}]}


def vertex_headers():
    return {"Content-Type": "application/json", "Authorization": "Bearer test"}


def vertex_body():
    return {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}


# --- SimulatedError unit tests ---


class TestSimulatedError:
    def test_attributes(self):
        err = SimulatedError(500, "boom")
        assert err.status_code == 500
        assert err.message == "boom"
        assert str(err) == "boom"


# --- Error rate ---


class TestErrorRate:
    @pytest_asyncio.fixture
    async def client(self):
        app = make_app(error_rate=1.0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    async def test_openai_500(self, client):
        resp = await client.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
        assert resp.status_code == 500
        assert "error" in resp.json()

    async def test_zero_error_rate_passes(self):
        app = make_app(error_rate=0.0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            assert resp.status_code == 200


# --- Timeout after N ---


class TestTimeoutAfter:
    async def test_openai_timeout_after_2(self):
        app = make_app(timeout_after=2)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r1 = await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            r2 = await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            assert r1.status_code == 200
            assert r2.status_code == 200

            r3 = await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            assert r3.status_code == 504


# --- Rate limit after N ---


class TestRateLimitAfter:
    async def test_openai_rate_limit_after_1(self):
        app = make_app(rate_limit_after=1)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r1 = await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            assert r1.status_code == 200

            r2 = await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            assert r2.status_code == 429


# --- Latency ---


class TestLatency:
    async def test_latency_adds_delay(self):
        import time

        app = make_app(latency_ms=200)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            start = time.time()
            resp = await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            elapsed = time.time() - start
            assert resp.status_code == 200
            assert elapsed >= 0.15


# --- Native error formats per provider ---


class TestNativeErrorFormats:
    async def test_openai_error_format(self):
        app = make_app(["openai"], error_rate=1.0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            assert resp.status_code == 500
            data = resp.json()
            assert "error" in data
            assert "message" in data["error"]
            assert data["error"]["type"] == "server_error"

    async def test_anthropic_error_format(self):
        app = make_app(["anthropic"], error_rate=1.0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/v1/messages", json=anthropic_body(), headers=anthropic_headers())
            assert resp.status_code == 500
            data = resp.json()
            assert data["type"] == "error"
            assert "error" in data
            assert data["error"]["type"] == "api_error"

    async def test_azure_error_format(self):
        app = make_app(["azure_openai"], error_rate=1.0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/openai/deployments/gpt-4/chat/completions",
                json=azure_body(), headers=azure_headers(),
            )
            assert resp.status_code == 500
            data = resp.json()
            assert "error" in data
            assert data["error"]["code"] == "internal_error"

    async def test_vertexai_error_format(self):
        app = make_app(["vertexai"], error_rate=1.0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/v1beta/models/gemini-pro:generateContent",
                json=vertex_body(), headers=vertex_headers(),
            )
            assert resp.status_code == 500
            data = resp.json()
            assert "error" in data
            assert data["error"]["status"] == "INTERNAL"

    async def test_bedrock_converse_error_format(self):
        app = make_app(["bedrock"], error_rate=1.0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/model/anthropic.claude-v2/converse",
                json=bedrock_body(), headers=bedrock_headers(),
            )
            assert resp.status_code == 500
            data = resp.json()
            assert "__type" in data
            assert "message" in data

    async def test_bedrock_invoke_error_format(self):
        app = make_app(["bedrock"], error_rate=1.0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/model/amazon.titan-text/invoke",
                json={"inputText": "hello"},
                headers=bedrock_headers(),
            )
            assert resp.status_code == 500
            data = resp.json()
            assert "__type" in data


# --- Rate limit returns 429 with correct native format ---


class TestRateLimitNativeFormats:
    async def test_openai_429(self):
        app = make_app(["openai"], rate_limit_after=1)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            resp = await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            assert resp.status_code == 429

    async def test_anthropic_429(self):
        app = make_app(["anthropic"], rate_limit_after=1)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            await c.post("/v1/messages", json=anthropic_body(), headers=anthropic_headers())
            resp = await c.post("/v1/messages", json=anthropic_body(), headers=anthropic_headers())
            assert resp.status_code == 429
            assert resp.json()["error"]["type"] == "rate_limit_error"


# --- Timeout returns 504 with correct native format ---


class TestTimeoutNativeFormats:
    async def test_openai_504(self):
        app = make_app(["openai"], timeout_after=1)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            resp = await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            assert resp.status_code == 504

    async def test_bedrock_504(self):
        app = make_app(["bedrock"], timeout_after=1)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            await c.post(
                "/model/anthropic.claude-v2/converse",
                json=bedrock_body(), headers=bedrock_headers(),
            )
            resp = await c.post(
                "/model/anthropic.claude-v2/converse",
                json=bedrock_body(), headers=bedrock_headers(),
            )
            assert resp.status_code == 504
            assert resp.json()["__type"] == "ModelTimeoutException"


# --- Counter is shared across providers ---


class TestSharedCounter:
    async def test_counter_shared_across_providers(self):
        app = make_app(["openai", "anthropic"], timeout_after=2)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r1 = await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            assert r1.status_code == 200

            r2 = await c.post("/v1/messages", json=anthropic_body(), headers=anthropic_headers())
            assert r2.status_code == 200

            r3 = await c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
            assert r3.status_code == 504


# --- Capacity-based overload (#19) ---


class TestMaxInflight:
    async def test_returns_503_when_capacity_exceeded(self):
        import asyncio
        app = make_app(max_inflight=1, latency_ms=500)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            tasks = [
                c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
                for _ in range(3)
            ]
            results = await asyncio.gather(*tasks)
            codes = sorted([r.status_code for r in results])
            assert 503 in codes, f"Expected at least one 503, got {codes}"

    async def test_within_capacity_all_succeed(self):
        import asyncio
        app = make_app(max_inflight=10)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            tasks = [
                c.post("/v1/chat/completions", json=openai_body(), headers=openai_headers())
                for _ in range(5)
            ]
            results = await asyncio.gather(*tasks)
            assert all(r.status_code == 200 for r in results)

    async def test_503_native_error_format(self):
        import asyncio
        app = make_app(["anthropic"], max_inflight=1, latency_ms=500)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            tasks = [
                c.post("/v1/messages", json=anthropic_body(), headers=anthropic_headers())
                for _ in range(3)
            ]
            results = await asyncio.gather(*tasks)
            err_responses = [r for r in results if r.status_code == 503]
            assert len(err_responses) > 0
            assert err_responses[0].json()["error"]["type"] == "overloaded_error"


# --- Per-chunk streaming latency (#18) ---


class TestChunkDelay:
    async def test_chunk_delay_adds_latency(self):
        import time
        app = make_app(chunk_delay_ms=100)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            start = time.time()
            resp = await c.post(
                "/v1/chat/completions",
                json={**openai_body(), "stream": True},
                headers=openai_headers(),
            )
            elapsed = time.time() - start
            assert resp.status_code == 200
            assert elapsed >= 0.3, f"Expected >= 300ms, got {elapsed*1000:.0f}ms"

    async def test_no_chunk_delay_is_fast(self):
        import time
        app = make_app(chunk_delay_ms=0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            start = time.time()
            resp = await c.post(
                "/v1/chat/completions",
                json={**openai_body(), "stream": True},
                headers=openai_headers(),
            )
            elapsed = time.time() - start
            assert resp.status_code == 200
            assert elapsed < 0.2


# --- TTFT and ITL simulation (#24) ---


class TestTTFT:
    async def test_ttft_delays_first_chunk(self):
        import time
        app = make_app(ttft_ms=300)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            start = time.time()
            resp = await c.post(
                "/v1/chat/completions",
                json={**openai_body(), "stream": True},
                headers=openai_headers(),
            )
            elapsed = time.time() - start
            assert resp.status_code == 200
            assert elapsed >= 0.25, f"TTFT should add ~300ms, got {elapsed*1000:.0f}ms"

    async def test_ttft_zero_is_fast(self):
        import time
        app = make_app(ttft_ms=0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            start = time.time()
            await c.post(
                "/v1/chat/completions",
                json={**openai_body(), "stream": True},
                headers=openai_headers(),
            )
            elapsed = time.time() - start
            assert elapsed < 0.2

    async def test_ttft_applies_to_non_streaming(self):
        import time
        app = make_app(ttft_ms=300)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            start = time.time()
            resp = await c.post(
                "/v1/chat/completions",
                json=openai_body(),
                headers=openai_headers(),
            )
            elapsed = time.time() - start
            assert resp.status_code == 200
            assert elapsed >= 0.25


class TestITL:
    async def test_itl_delays_between_chunks(self):
        import time
        app = make_app(itl_ms=100)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            start = time.time()
            resp = await c.post(
                "/v1/chat/completions",
                json={**openai_body(), "stream": True},
                headers=openai_headers(),
            )
            elapsed = time.time() - start
            assert resp.status_code == 200
            # Echo produces several chunks, each delayed 100ms
            assert elapsed >= 0.3, f"ITL should add significant delay, got {elapsed*1000:.0f}ms"

    async def test_itl_no_effect_on_non_streaming(self):
        import time
        app = make_app(itl_ms=500)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            start = time.time()
            resp = await c.post(
                "/v1/chat/completions",
                json=openai_body(),
                headers=openai_headers(),
            )
            elapsed = time.time() - start
            assert resp.status_code == 200
            assert elapsed < 0.3, f"ITL should not affect non-streaming, got {elapsed*1000:.0f}ms"

    async def test_ttft_and_itl_combined(self):
        import time
        app = make_app(ttft_ms=200, itl_ms=50)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            start = time.time()
            resp = await c.post(
                "/v1/chat/completions",
                json={**openai_body(), "stream": True},
                headers=openai_headers(),
            )
            elapsed = time.time() - start
            assert resp.status_code == 200
            # 200ms TTFT + several chunks * 50ms each
            assert elapsed >= 0.35
