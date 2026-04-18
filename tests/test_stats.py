"""Tests for persistent request stats."""

import json

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import ModelBackend
from llm_katan.providers.openai import OpenAIProvider
from llm_katan.server import ServerMetrics, create_app
from llm_katan.stats import PersistentStats


class MockBackend(ModelBackend):
    async def load_model(self):
        pass

    async def _generate_text(self, messages, max_tokens, temperature):
        return "Mock response", 10, 5


# --- Unit tests for PersistentStats ---


class TestPersistentStatsInMemory:
    def test_starts_at_zero(self):
        stats = PersistentStats()
        assert stats.total == 0
        assert stats.providers == {}

    def test_record_increments(self):
        stats = PersistentStats()
        stats.record("openai")
        stats.record("openai")
        stats.record("anthropic")
        assert stats.total == 3
        assert stats.providers == {"openai": 2, "anthropic": 1}

    def test_get_returns_snapshot(self):
        stats = PersistentStats()
        stats.record("openai")
        data = stats.get()
        assert data == {"total": 1, "providers": {"openai": 1}}


class TestPersistentStatsWithFile:
    def test_creates_file_on_first_record(self, tmp_path):
        path = tmp_path / "stats.json"
        stats = PersistentStats(str(path))
        stats.record("openai")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total"] == 1
        assert data["providers"]["openai"] == 1

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "stats.json"
        stats = PersistentStats(str(path))
        stats.record("anthropic")
        assert path.exists()

    def test_loads_existing_stats(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text(json.dumps({"total": 42, "providers": {"bedrock": 42}}))
        stats = PersistentStats(str(path))
        assert stats.total == 42
        assert stats.providers == {"bedrock": 42}

    def test_accumulates_across_instances(self, tmp_path):
        path = tmp_path / "stats.json"
        stats1 = PersistentStats(str(path))
        stats1.record("openai")
        stats1.record("openai")
        stats1.record("anthropic")

        stats2 = PersistentStats(str(path))
        assert stats2.total == 3
        stats2.record("openai")
        assert stats2.total == 4
        assert stats2.providers == {"openai": 3, "anthropic": 1}

    def test_handles_corrupted_file(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text("not json{{{")
        stats = PersistentStats(str(path))
        assert stats.total == 0

    def test_handles_missing_fields(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text(json.dumps({"extra": "field"}))
        stats = PersistentStats(str(path))
        assert stats.total == 0
        assert stats.providers == {}


# --- Integration tests with the server ---


def create_stats_test_app(stats_file=None):
    config = ServerConfig(
        model_name="test-model",
        served_model_name="gpt-test",
        providers=["openai"],
        stats_file=stats_file,
    )
    app = create_app(config)
    backend = MockBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()
    app.state.stats = PersistentStats(stats_file)

    provider = OpenAIProvider(backend=backend)
    provider.register_routes(app)
    return app


def openai_headers():
    return {"Content-Type": "application/json", "Authorization": "Bearer sk-test"}


def chat_body():
    return {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}


@pytest_asyncio.fixture
async def stats_client(tmp_path):
    stats_file = str(tmp_path / "stats.json")
    app = create_stats_test_app(stats_file)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, stats_file


class TestStatsEndpoint:
    async def test_stats_endpoint_returns_json(self, stats_client):
        client, _ = stats_client
        resp = await client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "providers" in data

    async def test_stats_increment_on_request(self, stats_client):
        client, _ = stats_client
        await client.post("/v1/chat/completions", json=chat_body(), headers=openai_headers())
        await client.post("/v1/chat/completions", json=chat_body(), headers=openai_headers())

        resp = await client.get("/stats")
        data = resp.json()
        assert data["total"] == 2
        assert data["providers"]["openai"] == 2

    async def test_stats_persisted_to_file(self, stats_client):
        client, stats_file = stats_client
        await client.post("/v1/chat/completions", json=chat_body(), headers=openai_headers())

        file_data = json.loads(open(stats_file).read())
        assert file_data["total"] == 1

    async def test_stats_in_metrics(self, stats_client):
        client, _ = stats_client
        await client.post("/v1/chat/completions", json=chat_body(), headers=openai_headers())

        resp = await client.get("/metrics")
        text = resp.text
        assert "llm_katan_lifetime_requests_total 1" in text
        assert 'llm_katan_lifetime_provider_requests_total{provider="openai"} 1' in text

    async def test_stats_not_counted_for_non_provider_routes(self, stats_client):
        client, _ = stats_client
        await client.get("/health")
        await client.get("/stats")
        await client.get("/metrics")

        resp = await client.get("/stats")
        assert resp.json()["total"] == 0
