"""
Anthropic Messages API provider for llm-katan.

Implements the Anthropic Messages API (`POST /v1/messages`) per the official spec
at platform.claude.com/docs/en/api/messages.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import register_provider
from .base import Provider

logger = logging.getLogger(__name__)


class AnthropicMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str | list  # string or array of content blocks


class AnthropicMessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: list[AnthropicMessage]
    system: str | list | None = None  # top-level system prompt
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = False
    metadata: dict | None = None


def _anthropic_error(status_code: int, message: str) -> JSONResponse:
    """Build an Anthropic-format error response."""
    error_types = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
        500: "api_error",
        529: "overloaded_error",
    }
    error_type = error_types.get(status_code, "api_error")
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


def _extract_text_from_content(content: str | list) -> str:
    """Extract plain text from Anthropic content (string or array of blocks)."""
    if isinstance(content, str):
        return content
    # Array of content blocks
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, dict) and block.get("type") == "tool_result":
            # Extract text from tool results
            inner = block.get("content", "")
            if isinstance(inner, str):
                parts.append(inner)
    return "\n".join(parts)


def _extract_system_text(system: str | list | None) -> str | None:
    """Extract plain text from Anthropic system field."""
    if system is None:
        return None
    if isinstance(system, str):
        return system
    # Array of text blocks
    parts = []
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts) if parts else None


class AnthropicProvider(Provider):
    name = "anthropic"
    auth_header = "x-api-key"

    def register_routes(self, app: FastAPI) -> None:
        @app.post("/v1/messages")
        async def messages(raw_request: Request):
            client_ip = raw_request.client.host if raw_request.client else "unknown"

            # Auth check
            auth_err = self.check_auth(dict(raw_request.headers))
            if auth_err:
                logger.warning("anthropic | %s | 401 | %s", client_ip, auth_err)
                return _anthropic_error(401, auth_err)

            # Check anthropic-version header
            if not raw_request.headers.get("anthropic-version"):
                logger.warning("anthropic | %s | missing anthropic-version header", client_ip)

            # Parse request body
            try:
                body = await raw_request.json()
            except Exception:
                logger.warning("anthropic | %s | 400 | invalid JSON", client_ip)
                return _anthropic_error(400, "invalid JSON in request body")

            # Validate required fields
            if "model" not in body:
                logger.warning("anthropic | %s | 400 | missing model", client_ip)
                return _anthropic_error(400, "model: field required")
            if "max_tokens" not in body:
                logger.warning("anthropic | %s | 400 | missing max_tokens", client_ip)
                return _anthropic_error(400, "max_tokens: field required")
            if "messages" not in body or not isinstance(body.get("messages"), list):
                logger.warning("anthropic | %s | 400 | missing messages", client_ip)
                return _anthropic_error(400, "messages: field required")

            try:
                request = AnthropicMessagesRequest(**body)
            except Exception as e:
                logger.warning("anthropic | %s | 400 | %s", client_ip, e)
                return _anthropic_error(400, str(e))

            logger.info(
                "anthropic | %s | model=%s messages=%d stream=%s max_tokens=%s temp=%s",
                client_ip, request.model, len(request.messages),
                request.stream, request.max_tokens, request.temperature,
            )

            metrics = app.state.metrics
            start_time = time.time()

            # Convert Anthropic messages to backend format
            backend_messages = []

            # System prompt is top-level in Anthropic
            system_text = _extract_system_text(request.system)
            if system_text:
                backend_messages.append({"role": "system", "content": system_text})

            for msg in request.messages:
                text = _extract_text_from_content(msg.content)
                backend_messages.append({"role": msg.role, "content": text})

            max_tokens = request.max_tokens
            temperature = request.temperature if request.temperature is not None else self.backend.config.temperature

            generated_text, prompt_tokens, completion_tokens = await self.backend.generate_text(
                backend_messages, max_tokens, temperature
            )

            msg_id = f"msg_{uuid.uuid4().hex[:24]}"
            model_name = self.backend.config.served_model_name

            if request.stream:
                return StreamingResponse(
                    self._stream_response(
                        msg_id, model_name, generated_text,
                        prompt_tokens, completion_tokens,
                        metrics, start_time, client_ip,
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )

            elapsed = time.time() - start_time
            metrics.record(elapsed, prompt_tokens, completion_tokens)

            logger.info("anthropic | %s | 200 | %d tokens | %.3fs", client_ip, prompt_tokens + completion_tokens, elapsed)

            resp_body = self._full_response(msg_id, model_name, generated_text, prompt_tokens, completion_tokens)
            return resp_body

    @staticmethod
    def _full_response(msg_id, model, text, input_tokens, output_tokens):
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }

    @staticmethod
    async def _stream_response(msg_id, model, text, input_tokens, output_tokens, metrics, start_time, client_ip="unknown"):
        # message_start
        msg_start = {
            'type': 'message_start',
            'message': {
                'id': msg_id, 'type': 'message', 'role': 'assistant',
                'content': [], 'model': model,
                'stop_reason': None, 'stop_sequence': None,
                'usage': {'input_tokens': input_tokens, 'output_tokens': 0},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(msg_start)}\n\n"

        # content_block_start
        yield (
            f"event: content_block_start\n"
            f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        )

        # content_block_delta chunks
        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i: i + chunk_size]
            yield (
                f"event: content_block_delta\n"
                f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': chunk_text}})}\n\n"
            )

        # content_block_stop
        yield (
            f"event: content_block_stop\n"
            f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        )

        # message_delta
        msg_delta = {
            'type': 'message_delta',
            'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
            'usage': {'output_tokens': output_tokens},
        }
        yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"

        # message_stop
        yield (
            f"event: message_stop\n"
            f"data: {json.dumps({'type': 'message_stop'})}\n\n"
        )

        elapsed = time.time() - start_time
        metrics.record(elapsed, input_tokens, output_tokens)
        logger.info("anthropic | %s | 200 | stream | %d tokens | %.3fs", client_ip, input_tokens + output_tokens, elapsed)


register_provider("anthropic", AnthropicProvider)
