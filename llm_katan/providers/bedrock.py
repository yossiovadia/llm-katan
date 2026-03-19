"""
AWS Bedrock Converse API provider for llm-katan.

Implements the Bedrock Converse API per the official spec at
docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html

Endpoints:
  POST /model/{modelId}/converse
  POST /model/{modelId}/converse-stream

Also supports the InvokeModel endpoint for Anthropic Claude models:
  POST /model/{modelId}/invoke

The Converse API is the unified Bedrock format (model-agnostic).
InvokeModel with Anthropic models uses the Anthropic Messages format
inside the Bedrock envelope (with anthropic_version: "bedrock-2023-05-31").

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_katan.model import ModelBackend

from . import register_provider
from .base import Provider

logger = logging.getLogger(__name__)


def _bedrock_error(status_code: int, message: str) -> JSONResponse:
    """Build a Bedrock error response."""
    code_map = {
        400: "ValidationException",
        401: "UnrecognizedClientException",
        403: "AccessDeniedException",
        404: "ResourceNotFoundException",
        408: "ModelTimeoutException",
        429: "ThrottlingException",
        500: "InternalServerException",
        503: "ServiceUnavailableException",
    }
    return JSONResponse(
        status_code=status_code,
        content={"message": message, "__type": code_map.get(status_code, "InternalServerException")},
    )


def _extract_text_from_content(content: list) -> str:
    """Extract text from Bedrock Converse content blocks."""
    parts = []
    for block in content:
        if isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return "\n".join(parts)


def _extract_system_text(system: list | None) -> str | None:
    """Extract text from Bedrock system blocks."""
    if not system:
        return None
    parts = []
    for block in system:
        if isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return "\n".join(parts) if parts else None


class BedrockProvider(Provider):
    name = "bedrock"
    auth_header = "Authorization"

    def register_routes(self, app: FastAPI) -> None:
        # Converse API (unified, model-agnostic)
        @app.post("/model/{model_id}/converse")
        async def converse(model_id: str, raw_request: Request):
            return await self._handle_converse(model_id, raw_request, app, stream=False)

        @app.post("/model/{model_id}/converse-stream")
        async def converse_stream(model_id: str, raw_request: Request):
            return await self._handle_converse(model_id, raw_request, app, stream=True)

        # InvokeModel (model-specific format — we support Anthropic Claude format)
        @app.post("/model/{model_id}/invoke")
        async def invoke_model(model_id: str, raw_request: Request):
            return await self._handle_invoke(model_id, raw_request, app)

    # ----------------------------------------------------------------
    # Converse API handler
    # ----------------------------------------------------------------

    async def _handle_converse(self, model_id: str, raw_request: Request, app: FastAPI, stream: bool):
        client_ip = raw_request.client.host if raw_request.client else "unknown"

        auth_err = self.check_auth(dict(raw_request.headers))
        if auth_err:
            logger.warning("bedrock | %s | 401 | %s", client_ip, auth_err)
            return _bedrock_error(401, auth_err)

        try:
            body = await raw_request.json()
        except Exception:
            logger.warning("bedrock | %s | 400 | invalid JSON", client_ip)
            return _bedrock_error(400, "Invalid JSON in request body")

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            logger.warning("bedrock | %s | 400 | missing messages", client_ip)
            return _bedrock_error(400, "messages: field required")

        inference_config = body.get("inferenceConfig", {})
        max_tokens = inference_config.get("maxTokens", self.backend.config.max_tokens)
        temperature = inference_config.get("temperature", self.backend.config.temperature)

        logger.info(
            "bedrock converse | %s | model=%s messages=%d stream=%s max_tokens=%s temp=%s",
            client_ip, model_id, len(messages), stream, max_tokens, temperature,
        )

        metrics = app.state.metrics
        start_time = time.time()

        # Convert Bedrock messages to backend format
        backend_messages = []
        system_text = _extract_system_text(body.get("system"))
        if system_text:
            backend_messages.append({"role": "system", "content": system_text})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            text = _extract_text_from_content(content)
            backend_messages.append({"role": role, "content": text})

        generated_text, prompt_tokens, completion_tokens = await self.backend.generate_text(
            backend_messages, max_tokens, temperature
        )

        if stream:
            return StreamingResponse(
                self._stream_converse(
                    generated_text, prompt_tokens, completion_tokens,
                    metrics, start_time, client_ip,
                ),
                media_type="application/vnd.amazon.eventstream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        elapsed = time.time() - start_time
        metrics.record(elapsed, prompt_tokens, completion_tokens)
        logger.info("bedrock converse | %s | 200 | %d tokens | %.3fs", client_ip, prompt_tokens + completion_tokens, elapsed)

        return self._converse_response(generated_text, prompt_tokens, completion_tokens, elapsed)

    @staticmethod
    def _converse_response(text, input_tokens, output_tokens, latency_ms_float):
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": text}],
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "totalTokens": input_tokens + output_tokens,
            },
            "metrics": {
                "latencyMs": int(latency_ms_float * 1000),
            },
        }

    @staticmethod
    async def _stream_converse(text, input_tokens, output_tokens, metrics, start_time, client_ip="unknown"):
        # messageStart
        yield f"data: {json.dumps({'messageStart': {'role': 'assistant'}})}\n\n"

        # contentBlockStart
        yield f"data: {json.dumps({'contentBlockStart': {'start': {'text': {}}, 'contentBlockIndex': 0}})}\n\n"

        # contentBlockDelta chunks
        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i: i + chunk_size]
            yield f"data: {json.dumps({'contentBlockDelta': {'delta': {'text': chunk_text}, 'contentBlockIndex': 0}})}\n\n"

        # contentBlockStop
        yield f"data: {json.dumps({'contentBlockStop': {'contentBlockIndex': 0}})}\n\n"

        # messageStop
        yield f"data: {json.dumps({'messageStop': {'stopReason': 'end_turn'}})}\n\n"

        # metadata (final event)
        elapsed = time.time() - start_time
        yield f"data: {json.dumps({'metadata': {'usage': {'inputTokens': input_tokens, 'outputTokens': output_tokens, 'totalTokens': input_tokens + output_tokens}, 'metrics': {'latencyMs': int(elapsed * 1000)}}})}\n\n"

        metrics.record(elapsed, input_tokens, output_tokens)
        logger.info("bedrock converse | %s | 200 | stream | %d tokens | %.3fs", client_ip, input_tokens + output_tokens, elapsed)

    # ----------------------------------------------------------------
    # InvokeModel handler (Anthropic Claude format inside Bedrock)
    # ----------------------------------------------------------------

    async def _handle_invoke(self, model_id: str, raw_request: Request, app: FastAPI):
        client_ip = raw_request.client.host if raw_request.client else "unknown"

        auth_err = self.check_auth(dict(raw_request.headers))
        if auth_err:
            logger.warning("bedrock invoke | %s | 401 | %s", client_ip, auth_err)
            return _bedrock_error(401, auth_err)

        try:
            body = await raw_request.json()
        except Exception:
            logger.warning("bedrock invoke | %s | 400 | invalid JSON", client_ip)
            return _bedrock_error(400, "Invalid JSON in request body")

        # Detect if this is an Anthropic Claude model
        if "anthropic" in model_id.lower() or "claude" in model_id.lower():
            return await self._invoke_anthropic(model_id, body, app, client_ip)

        # Default: treat as generic text model (Amazon Titan style)
        return await self._invoke_generic(model_id, body, app, client_ip)

    async def _invoke_anthropic(self, model_id: str, body: dict, app: FastAPI, client_ip: str):
        """Handle InvokeModel for Anthropic Claude models — uses Anthropic Messages format."""
        messages = body.get("messages")
        if not messages:
            logger.warning("bedrock invoke | %s | 400 | missing messages", client_ip)
            return _bedrock_error(400, "messages: field required")

        max_tokens = body.get("max_tokens", self.backend.config.max_tokens)
        temperature = body.get("temperature", self.backend.config.temperature)

        logger.info(
            "bedrock invoke anthropic | %s | model=%s messages=%d max_tokens=%s",
            client_ip, model_id, len(messages), max_tokens,
        )

        metrics = app.state.metrics
        start_time = time.time()

        # Convert Anthropic messages to backend format
        backend_messages = []
        system_text = body.get("system")
        if isinstance(system_text, str) and system_text:
            backend_messages.append({"role": "system", "content": system_text})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(
                    b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = str(content)
            backend_messages.append({"role": role, "content": text})

        generated_text, prompt_tokens, completion_tokens = await self.backend.generate_text(
            backend_messages, max_tokens, temperature
        )

        elapsed = time.time() - start_time
        metrics.record(elapsed, prompt_tokens, completion_tokens)
        logger.info("bedrock invoke anthropic | %s | 200 | %d tokens | %.3fs", client_ip, prompt_tokens + completion_tokens, elapsed)

        # Return Anthropic Messages format (as Bedrock does for Claude)
        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": generated_text}],
            "model": model_id,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            },
        }

    async def _invoke_generic(self, model_id: str, body: dict, app: FastAPI, client_ip: str):
        """Handle InvokeModel for generic models (Amazon Titan style)."""
        input_text = body.get("inputText", "")
        gen_config = body.get("textGenerationConfig", {})
        max_tokens = gen_config.get("maxTokenCount", self.backend.config.max_tokens)
        temperature = gen_config.get("temperature", self.backend.config.temperature)

        logger.info("bedrock invoke generic | %s | model=%s max_tokens=%s", client_ip, model_id, max_tokens)

        metrics = app.state.metrics
        start_time = time.time()

        backend_messages = [{"role": "user", "content": input_text}]
        generated_text, prompt_tokens, completion_tokens = await self.backend.generate_text(
            backend_messages, max_tokens, temperature
        )

        elapsed = time.time() - start_time
        metrics.record(elapsed, prompt_tokens, completion_tokens)
        logger.info("bedrock invoke generic | %s | 200 | %d tokens | %.3fs", client_ip, prompt_tokens + completion_tokens, elapsed)

        # Amazon Titan-style response
        return {
            "inputTextTokenCount": prompt_tokens,
            "results": [
                {
                    "tokenCount": completion_tokens,
                    "outputText": generated_text,
                    "completionReason": "FINISH",
                }
            ],
        }


register_provider("bedrock", BedrockProvider)
