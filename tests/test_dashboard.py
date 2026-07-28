"""Tests for DashboardMiddleware — verifies streaming requests bypass capture."""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import ModelBackend
from llm_katan.providers.anthropic import AnthropicProvider
from llm_katan.server import ServerMetrics, create_app
from llm_katan.stats import PersistentStats


class MockBackend(ModelBackend):
    async def load_model(self):
        pass

    async def _generate_text(self, prompt, max_tokens, temperature):
        generated = "dashboard test response"
        return generated, 10, len(generated)


def make_app():
    config = ServerConfig(
        model_name="test-model",
        served_model_name="claude-test",
        port=8000,
        providers=["anthropic"],
    )
    app = create_app(config)
    backend = MockBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()
    app.state.stats = PersistentStats()
    provider = AnthropicProvider(backend=backend)
    provider.register_routes(app)
    return app


def base_request(**overrides):
    req = {
        "model": "claude-test",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hello"}],
    }
    req.update(overrides)
    return req


HEADERS = {
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
    "x-api-key": "sk-ant-test-key",
}


@pytest_asyncio.fixture
async def client():
    app = make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
class TestDashboardStreamBypass:
    """Streaming requests should bypass DashboardMiddleware (no broadcast)."""

    async def test_non_streaming_triggers_broadcast(self, client):
        """Positive control: non-streaming request DOES broadcast an event."""
        with patch("llm_katan.server.broadcaster.broadcast", new_callable=AsyncMock) as mock_broadcast:
            resp = await client.post("/v1/messages", json=base_request(), headers=HEADERS)
            assert resp.status_code == 200
            mock_broadcast.assert_called_once()

    async def test_streaming_skips_broadcast(self, client):
        """Streaming request should NOT broadcast — the middleware returns early."""
        with patch("llm_katan.server.broadcaster.broadcast", new_callable=AsyncMock) as mock_broadcast:
            resp = await client.post(
                "/v1/messages", json=base_request(stream=True), headers=HEADERS
            )
            assert resp.status_code == 200
            mock_broadcast.assert_not_called()
