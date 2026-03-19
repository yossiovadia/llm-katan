"""Extensive tests for the AWS Bedrock provider.

Tests cover:
- Converse API (unified, model-agnostic)
- ConverseStream API (streaming)
- InvokeModel with Anthropic Claude format
- InvokeModel with generic/Titan format
- Auth and error handling
"""

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_katan.config import ServerConfig
from llm_katan.model import ModelBackend
from llm_katan.providers.bedrock import BedrockProvider
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
        served_model_name="bedrock-test",
        port=8000,
        providers=["bedrock"],
    )
    app = create_app(config)
    backend = MockBackend(config)
    app.state.backend = backend
    app.state.metrics = ServerMetrics()
    provider = BedrockProvider(backend=backend)
    provider.register_routes(app)
    return app


@pytest_asyncio.fixture
async def client():
    app = make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def bedrock_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": "AWS4-HMAC-SHA256 Credential=test",
    }


def converse_request(**overrides):
    req = {
        "messages": [
            {"role": "user", "content": [{"text": "hello"}]}
        ],
    }
    req.update(overrides)
    return req


# ============================================================
# Converse API — Request Format
# ============================================================

class TestConverseRequest:
    @pytest.mark.asyncio
    async def test_basic_request(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json=converse_request(),
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200
        text = resp.json()["output"]["message"]["content"][0]["text"]
        assert "Response to: hello" in text

    @pytest.mark.asyncio
    async def test_multi_turn(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json=converse_request(messages=[
                {"role": "user", "content": [{"text": "hello"}]},
                {"role": "assistant", "content": [{"text": "hi there"}]},
                {"role": "user", "content": [{"text": "how are you"}]},
            ]),
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200
        assert "how are you" in resp.json()["output"]["message"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_system_prompt(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json=converse_request(system=[{"text": "You are a pirate"}]),
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_inference_config(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json=converse_request(inferenceConfig={
                "maxTokens": 100,
                "temperature": 0.5,
                "topP": 0.9,
            }),
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_messages(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json={},
            headers=bedrock_headers(),
        )
        assert resp.status_code == 400
        assert resp.json()["__type"] == "ValidationException"

    @pytest.mark.asyncio
    async def test_any_model_id(self, client):
        resp = await client.post(
            "/model/amazon.titan-text-express-v1/converse",
            json=converse_request(),
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200


# ============================================================
# Converse API — Response Format
# ============================================================

class TestConverseResponse:
    @pytest.mark.asyncio
    async def test_output_structure(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json=converse_request(),
            headers=bedrock_headers(),
        )
        data = resp.json()
        assert "output" in data
        assert "message" in data["output"]
        assert data["output"]["message"]["role"] == "assistant"
        assert isinstance(data["output"]["message"]["content"], list)
        assert "text" in data["output"]["message"]["content"][0]

    @pytest.mark.asyncio
    async def test_stop_reason(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json=converse_request(),
            headers=bedrock_headers(),
        )
        assert resp.json()["stopReason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_usage(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json=converse_request(),
            headers=bedrock_headers(),
        )
        usage = resp.json()["usage"]
        assert "inputTokens" in usage
        assert "outputTokens" in usage
        assert "totalTokens" in usage
        assert usage["totalTokens"] == usage["inputTokens"] + usage["outputTokens"]
        # Must NOT have OpenAI-style names
        assert "prompt_tokens" not in usage

    @pytest.mark.asyncio
    async def test_metrics(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse",
            json=converse_request(),
            headers=bedrock_headers(),
        )
        metrics = resp.json()["metrics"]
        assert "latencyMs" in metrics
        assert isinstance(metrics["latencyMs"], int)


# ============================================================
# ConverseStream API
# ============================================================

class TestConverseStream:
    def _parse_sse(self, text):
        chunks = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                chunks.append(json.loads(line.removeprefix("data: ")))
        return chunks

    @pytest.mark.asyncio
    async def test_stream_endpoint(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse-stream",
            json=converse_request(),
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_stream_event_sequence(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/converse-stream",
            json=converse_request(),
            headers=bedrock_headers(),
        )
        chunks = self._parse_sse(resp.text)

        # First: messageStart
        assert "messageStart" in chunks[0]
        assert chunks[0]["messageStart"]["role"] == "assistant"

        # Second: contentBlockStart
        assert "contentBlockStart" in chunks[1]

        # Middle: contentBlockDelta(s)
        deltas = [c for c in chunks if "contentBlockDelta" in c]
        assert len(deltas) > 0

        # Then: contentBlockStop
        stop_chunks = [c for c in chunks if "contentBlockStop" in c]
        assert len(stop_chunks) == 1

        # Then: messageStop
        msg_stop = [c for c in chunks if "messageStop" in c]
        assert len(msg_stop) == 1
        assert msg_stop[0]["messageStop"]["stopReason"] == "end_turn"

        # Last: metadata
        assert "metadata" in chunks[-1]

    @pytest.mark.asyncio
    async def test_stream_reassembles(self, client):
        body = converse_request()

        # Non-streaming
        resp = await client.post("/model/test/converse", json=body, headers=bedrock_headers())
        non_stream_text = resp.json()["output"]["message"]["content"][0]["text"]

        # Streaming
        resp = await client.post("/model/test/converse-stream", json=body, headers=bedrock_headers())
        chunks = self._parse_sse(resp.text)
        deltas = [c["contentBlockDelta"]["delta"]["text"] for c in chunks if "contentBlockDelta" in c]
        stream_text = "".join(deltas)

        assert stream_text == non_stream_text

    @pytest.mark.asyncio
    async def test_stream_metadata_has_usage(self, client):
        resp = await client.post(
            "/model/test/converse-stream",
            json=converse_request(),
            headers=bedrock_headers(),
        )
        chunks = self._parse_sse(resp.text)
        metadata = chunks[-1]["metadata"]
        assert metadata["usage"]["inputTokens"] > 0
        assert metadata["usage"]["outputTokens"] > 0
        assert "latencyMs" in metadata["metrics"]


# ============================================================
# InvokeModel — Anthropic Claude format
# ============================================================

class TestInvokeAnthropic:
    @pytest.mark.asyncio
    async def test_invoke_claude(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/invoke",
            json={
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello from bedrock"}],
            },
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        # Response should be in Anthropic Messages format
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["id"].startswith("msg_")
        assert data["content"][0]["type"] == "text"
        assert "hello from bedrock" in data["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_invoke_claude_usage(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/invoke",
            json={
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=bedrock_headers(),
        )
        usage = resp.json()["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage

    @pytest.mark.asyncio
    async def test_invoke_claude_with_system(self, client):
        resp = await client.post(
            "/model/anthropic.claude-3-sonnet/invoke",
            json={
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "system": "You are helpful",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_invoke_claude_array_content(self, client):
        resp = await client.post(
            "/model/anthropic.claude-v2/invoke",
            json={
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "array content"}]}],
            },
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200
        assert "array content" in resp.json()["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_invoke_claude_model_echoed(self, client):
        resp = await client.post(
            "/model/anthropic.claude-3-haiku/invoke",
            json={
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=bedrock_headers(),
        )
        assert resp.json()["model"] == "anthropic.claude-3-haiku"


# ============================================================
# InvokeModel — Generic / Titan format
# ============================================================

class TestInvokeGeneric:
    @pytest.mark.asyncio
    async def test_invoke_titan(self, client):
        resp = await client.post(
            "/model/amazon.titan-text-express-v1/invoke",
            json={"inputText": "hello from titan"},
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "outputText" in data["results"][0]
        assert "hello from titan" in data["results"][0]["outputText"]
        assert data["results"][0]["completionReason"] == "FINISH"
        assert "inputTextTokenCount" in data

    @pytest.mark.asyncio
    async def test_invoke_titan_with_config(self, client):
        resp = await client.post(
            "/model/amazon.titan-text-express-v1/invoke",
            json={
                "inputText": "hello",
                "textGenerationConfig": {"maxTokenCount": 50, "temperature": 0.3},
            },
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200


# ============================================================
# InvokeModel — Meta Llama format
# ============================================================

class TestInvokeMeta:
    @pytest.mark.asyncio
    async def test_invoke_llama(self, client):
        resp = await client.post(
            "/model/meta.llama3-70b-instruct-v1/invoke",
            json={"prompt": "hello from llama", "max_gen_len": 100},
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "generation" in data
        assert "hello from llama" in data["generation"]
        assert data["stop_reason"] == "stop"
        assert "prompt_token_count" in data
        assert "generation_token_count" in data


# ============================================================
# InvokeModel — Cohere Command format
# ============================================================

class TestInvokeCohere:
    @pytest.mark.asyncio
    async def test_invoke_cohere(self, client):
        resp = await client.post(
            "/model/cohere.command-r-v1/invoke",
            json={"message": "hello from cohere", "max_tokens": 100},
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "hello from cohere" in data["text"]
        assert data["finish_reason"] == "COMPLETE"
        assert data["meta"]["billed_units"]["input_tokens"] > 0
        assert data["meta"]["billed_units"]["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_invoke_cohere_with_history(self, client):
        resp = await client.post(
            "/model/cohere.command-r-plus-v1/invoke",
            json={
                "message": "how are you",
                "chat_history": [
                    {"role": "USER", "message": "hello"},
                    {"role": "CHATBOT", "message": "hi there"},
                ],
            },
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200
        assert "how are you" in resp.json()["text"]

    @pytest.mark.asyncio
    async def test_invoke_cohere_with_preamble(self, client):
        resp = await client.post(
            "/model/cohere.command-r-v1/invoke",
            json={"message": "hi", "preamble": "You are a pirate"},
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200


# ============================================================
# InvokeModel — Mistral format
# ============================================================

class TestInvokeMistral:
    @pytest.mark.asyncio
    async def test_invoke_mistral(self, client):
        resp = await client.post(
            "/model/mistral.mistral-7b-instruct-v0/invoke",
            json={"prompt": "<s>[INST] hello from mistral [/INST]", "max_tokens": 100},
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "outputs" in data
        assert isinstance(data["outputs"], list)
        assert "text" in data["outputs"][0]
        assert "hello from mistral" in data["outputs"][0]["text"]
        assert data["outputs"][0]["stop_reason"] == "stop"


# ============================================================
# InvokeModel — DeepSeek format
# ============================================================

class TestInvokeDeepSeek:
    @pytest.mark.asyncio
    async def test_invoke_deepseek(self, client):
        resp = await client.post(
            "/model/deepseek.r1-v1/invoke",
            json={"prompt": "hello from deepseek", "max_tokens": 100},
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert isinstance(data["choices"], list)
        assert "text" in data["choices"][0]
        assert "hello from deepseek" in data["choices"][0]["text"]
        assert data["choices"][0]["stop_reason"] == "stop"


# ============================================================
# InvokeModel — AI21 Jamba format
# ============================================================

class TestInvokeAI21:
    @pytest.mark.asyncio
    async def test_invoke_jamba(self, client):
        resp = await client.post(
            "/model/ai21.jamba-instruct-v1/invoke",
            json={
                "messages": [{"role": "user", "content": "hello from ai21"}],
                "max_tokens": 100,
            },
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "hello from ai21" in data["choices"][0]["message"]["content"]
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]


# ============================================================
# InvokeModel — Amazon Nova format
# ============================================================

class TestInvokeNova:
    @pytest.mark.asyncio
    async def test_invoke_nova(self, client):
        resp = await client.post(
            "/model/amazon.nova-pro-v1/invoke",
            json={
                "messages": [{"role": "user", "content": [{"text": "hello from nova"}]}],
            },
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["output"]["message"]["role"] == "assistant"
        assert "hello from nova" in data["output"]["message"]["content"][0]["text"]
        assert data["stopReason"] == "end_turn"
        assert data["usage"]["inputTokens"] > 0

    @pytest.mark.asyncio
    async def test_invoke_nova_with_system(self, client):
        resp = await client.post(
            "/model/amazon.nova-micro-v1/invoke",
            json={
                "system": [{"text": "You are helpful"}],
                "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            },
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_invoke_nova_with_config(self, client):
        resp = await client.post(
            "/model/amazon.nova-lite-v1/invoke",
            json={
                "messages": [{"role": "user", "content": [{"text": "hi"}]}],
                "inferenceConfig": {"maxTokens": 50, "temperature": 0.3},
            },
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200


# ============================================================
# InvokeModel — Unknown model falls back to Titan
# ============================================================

class TestInvokeFallback:
    @pytest.mark.asyncio
    async def test_unknown_model_uses_titan_format(self, client):
        resp = await client.post(
            "/model/some.unknown-model-v1/invoke",
            json={"inputText": "hello from unknown"},
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200
        assert "results" in resp.json()
        assert "outputText" in resp.json()["results"][0]


# ============================================================
# Auth
# ============================================================

class TestAuth:
    @pytest.mark.asyncio
    async def test_missing_auth_converse(self, client):
        resp = await client.post(
            "/model/test/converse",
            json=converse_request(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 401
        assert "Authorization" in resp.json()["message"]

    @pytest.mark.asyncio
    async def test_missing_auth_invoke(self, client):
        resp = await client.post(
            "/model/test/invoke",
            json={"inputText": "hi"},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_present_accepted(self, client):
        resp = await client.post(
            "/model/test/converse",
            json=converse_request(),
            headers=bedrock_headers(),
        )
        assert resp.status_code == 200


# ============================================================
# Error Format
# ============================================================

class TestErrorFormat:
    @pytest.mark.asyncio
    async def test_error_structure(self, client):
        resp = await client.post(
            "/model/test/converse",
            json={},
            headers=bedrock_headers(),
        )
        data = resp.json()
        assert "message" in data
        assert "__type" in data

    @pytest.mark.asyncio
    async def test_400_validation_exception(self, client):
        resp = await client.post(
            "/model/test/converse",
            json={},
            headers=bedrock_headers(),
        )
        assert resp.status_code == 400
        assert resp.json()["__type"] == "ValidationException"

    @pytest.mark.asyncio
    async def test_401_unrecognized_client(self, client):
        resp = await client.post(
            "/model/test/converse",
            json=converse_request(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 401
        assert "Exception" in resp.json()["__type"]
