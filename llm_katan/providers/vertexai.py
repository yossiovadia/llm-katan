"""
Google Vertex AI / Gemini API provider for llm-katan.

Implements the Gemini generateContent API per the official spec at
ai.google.dev/api/generate-content.

Supports:
  Native Gemini:
    POST /v1beta/models/{model}:generateContent
    POST /v1beta/models/{model}:streamGenerateContent
    POST /v1/models/{model}:generateContent
    POST /v1/models/{model}:streamGenerateContent

  OpenAI-compatible (Vertex AI):
    POST /v1/projects/{project}/locations/{location}/endpoints/{endpoint}/chat/completions

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_katan.model import SimulatedError

from . import register_provider
from .base import Provider

logger = logging.getLogger(__name__)


def _gemini_error(status_code: int, message: str) -> JSONResponse:
    """Build a Gemini/Vertex AI error response."""
    status_map = {
        400: "INVALID_ARGUMENT",
        401: "UNAUTHENTICATED",
        403: "PERMISSION_DENIED",
        404: "NOT_FOUND",
        429: "RESOURCE_EXHAUSTED",
        500: "INTERNAL",
        503: "UNAVAILABLE",
    }
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": status_code,
                "message": message,
                "status": status_map.get(status_code, "UNKNOWN"),
            }
        },
    )


def _extract_text_from_parts(parts: list) -> str:
    """Extract text from Gemini content parts."""
    texts = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            texts.append(part["text"])
    return "\n".join(texts)


class VertexAIProvider(Provider):
    name = "vertexai"
    auth_header = "Authorization"

    def _normalize_key(self, raw_value: str) -> str:
        """Strip 'Bearer ' prefix from Authorization header."""
        if raw_value.startswith("Bearer "):
            return raw_value[7:]
        return raw_value

    def check_auth_with_request(self, headers: dict, query_params) -> str | None:
        """Vertex/Gemini supports Authorization: Bearer OR ?key= query parameter."""
        # Try header auth first
        header_err = self.check_auth(headers)
        if header_err is None:
            return None

        # If header was present but key was wrong, don't fall through to ?key=
        if "invalid API key" in (header_err or ""):
            return header_err

        # Fall back to ?key= query param (Gemini API style)
        key_param = query_params.get("key")
        if key_param:
            if self.expected_key is not None and key_param != self.expected_key:
                return (
                    f"invalid API key for vertexai: "
                    f"got '{key_param}', expected '{self.expected_key}'"
                )
            return None

        return "missing Authorization header or ?key= query parameter"

    def register_routes(self, app: FastAPI) -> None:
        # Native Gemini API: /v1beta/models/{model}:generateContent
        @app.post("/v1beta/models/{model}:generateContent")
        async def generate_content(model: str, raw_request: Request):
            return await self._handle_request(model, raw_request, app, stream=False)

        @app.post("/v1beta/models/{model}:streamGenerateContent")
        async def stream_generate_content(model: str, raw_request: Request):
            return await self._handle_request(model, raw_request, app, stream=True)

        @app.post("/v1/models/{model}:generateContent")
        async def generate_content_v1(model: str, raw_request: Request):
            return await self._handle_request(model, raw_request, app, stream=False)

        @app.post("/v1/models/{model}:streamGenerateContent")
        async def stream_generate_content_v1(model: str, raw_request: Request):
            return await self._handle_request(model, raw_request, app, stream=True)

        # OpenAI-compatible Vertex AI endpoint
        @app.post("/v1/projects/{project}/locations/{location}/endpoints/{endpoint}/chat/completions")
        async def vertex_openai_compat(project: str, location: str, endpoint: str, raw_request: Request):
            return await self._handle_openai_compat(endpoint, raw_request, app)

    async def _handle_request(self, model: str, raw_request: Request, app: FastAPI, stream: bool):
        client_ip = raw_request.client.host if raw_request.client else "unknown"

        # Auth check (header or ?key= query param)
        auth_err = self.check_auth_with_request(dict(raw_request.headers), raw_request.query_params)
        if auth_err:
            logger.warning("vertexai | %s | 401 | %s", client_ip, auth_err)
            return _gemini_error(401, auth_err)

        # Parse request body
        try:
            body = await raw_request.json()
        except Exception:
            logger.warning("vertexai | %s | 400 | invalid JSON", client_ip)
            return _gemini_error(400, "Invalid JSON in request body")

        # Validate required fields
        contents = body.get("contents")
        if not contents or not isinstance(contents, list):
            logger.warning("vertexai | %s | 400 | missing contents", client_ip)
            return _gemini_error(400, "contents: field required")

        # Extract generation config
        gen_config = body.get("generationConfig", {})
        max_tokens = gen_config.get("maxOutputTokens", self.backend.config.max_tokens)
        temperature = gen_config.get("temperature", self.backend.config.temperature)

        logger.info(
            "vertexai | %s | model=%s contents=%d stream=%s max_tokens=%s temp=%s",
            client_ip, model, len(contents), stream, max_tokens, temperature,
        )

        metrics = app.state.metrics
        start_time = time.time()

        # Convert Gemini contents to backend messages
        backend_messages = []

        # System instruction is top-level in Gemini
        sys_instruction = body.get("systemInstruction")
        if sys_instruction and "parts" in sys_instruction:
            sys_text = _extract_text_from_parts(sys_instruction["parts"])
            if sys_text:
                backend_messages.append({"role": "system", "content": sys_text})

        for content in contents:
            role = content.get("role", "user")
            # Gemini uses "model" for assistant role
            backend_role = "assistant" if role == "model" else "user"
            parts = content.get("parts", [])
            text = _extract_text_from_parts(parts)
            backend_messages.append({"role": backend_role, "content": text})

        try:
            generated_text, prompt_tokens, completion_tokens = await self.backend.generate_text(
                backend_messages, max_tokens, temperature
            )
        except SimulatedError as e:
            logger.warning("vertexai | %s | %d | simulated: %s", client_ip, e.status_code, e.message)
            return _gemini_error(e.status_code, e.message)

        model_name = self.backend.config.served_model_name

        if stream:
            return StreamingResponse(
                self._stream_response(
                    model_name, generated_text, prompt_tokens, completion_tokens,
                    metrics, start_time, client_ip,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        elapsed = time.time() - start_time
        metrics.record(elapsed, prompt_tokens, completion_tokens)
        logger.info("vertexai | %s | 200 | %d tokens | %.3fs", client_ip, prompt_tokens + completion_tokens, elapsed)

        resp_body = self._full_response(model_name, generated_text, prompt_tokens, completion_tokens)
        return resp_body

    async def _handle_openai_compat(self, endpoint: str, raw_request: Request, app):
        """Handle OpenAI-compatible Vertex AI endpoint. Same format as OpenAI."""
        client_ip = raw_request.client.host if raw_request.client else "unknown"

        auth_err = self.check_auth_with_request(dict(raw_request.headers), raw_request.query_params)
        if auth_err:
            logger.warning("vertexai openai-compat | %s | 401 | %s", client_ip, auth_err)
            return _gemini_error(401, auth_err)

        try:
            body = await raw_request.json()
        except Exception:
            return _gemini_error(400, "Invalid JSON in request body")

        messages_raw = body.get("messages")
        if not messages_raw or not isinstance(messages_raw, list):
            return _gemini_error(400, "messages: field required")

        model_name = body.get("model", endpoint)
        max_tokens = body.get("max_tokens", self.backend.config.max_tokens)
        temperature = body.get("temperature", self.backend.config.temperature)
        stream = body.get("stream", False)

        logger.info(
            "vertexai openai-compat | %s | model=%s messages=%d stream=%s",
            client_ip, model_name, len(messages_raw), stream,
        )

        metrics = app.state.metrics
        start_time = time.time()

        backend_messages = []
        for msg in messages_raw:
            if isinstance(msg, dict):
                backend_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        try:
            generated_text, prompt_tokens, completion_tokens = await self.backend.generate_text(
                backend_messages, max_tokens, temperature
            )
        except SimulatedError as e:
            logger.warning("vertexai openai-compat | %s | %d | simulated: %s", client_ip, e.status_code, e.message)
            return _gemini_error(e.status_code, e.message)

        response_id = f"chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())

        if stream:
            async def stream_response():
                chunk_size = 4
                for i in range(0, len(generated_text), chunk_size):
                    chunk = {"id": response_id, "object": "chat.completion.chunk", "created": created, "model": model_name,
                             "choices": [{"index": 0, "delta": {"content": generated_text[i:i+chunk_size]}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                final = {
                    "id": response_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
                elapsed = time.time() - start_time
                metrics.record(elapsed, prompt_tokens, completion_tokens)
                logger.info(
                    "vertexai openai-compat | %s | 200 | stream | %d tokens | %.3fs",
                    client_ip, prompt_tokens + completion_tokens, elapsed,
                )

            return StreamingResponse(stream_response(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

        elapsed = time.time() - start_time
        metrics.record(elapsed, prompt_tokens, completion_tokens)
        logger.info("vertexai openai-compat | %s | 200 | %d tokens | %.3fs", client_ip, prompt_tokens + completion_tokens, elapsed)

        return {
            "id": response_id, "object": "chat.completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": generated_text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @staticmethod
    def _full_response(model, text, prompt_tokens, completion_tokens):
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": text}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": [
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "probability": "NEGLIGIBLE",
                        },
                    ],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": prompt_tokens,
                "candidatesTokenCount": completion_tokens,
                "totalTokenCount": prompt_tokens + completion_tokens,
            },
            "modelVersion": model,
        }

    @staticmethod
    async def _stream_response(model, text, prompt_tokens, completion_tokens, metrics, start_time, client_ip="unknown"):
        # Stream in chunks, each chunk is a complete GenerateContentResponse
        chunk_size = 4
        chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

        for i, chunk_text in enumerate(chunks):
            is_last = i == len(chunks) - 1
            chunk_response = {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": chunk_text}],
                            "role": "model",
                        },
                        "index": 0,
                    }
                ],
            }

            # Last chunk gets finishReason and usage
            if is_last:
                chunk_response["candidates"][0]["finishReason"] = "STOP"
                chunk_response["candidates"][0]["safetyRatings"] = [
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "NEGLIGIBLE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "NEGLIGIBLE"},
                ]
                chunk_response["usageMetadata"] = {
                    "promptTokenCount": prompt_tokens,
                    "candidatesTokenCount": completion_tokens,
                    "totalTokenCount": prompt_tokens + completion_tokens,
                }
                chunk_response["modelVersion"] = model

            yield f"data: {json.dumps(chunk_response)}\n\n"

        elapsed = time.time() - start_time
        metrics.record(elapsed, prompt_tokens, completion_tokens)
        logger.info(
            "vertexai | %s | 200 | stream | %d tokens | %.3fs",
            client_ip, prompt_tokens + completion_tokens, elapsed,
        )


register_provider("vertexai", VertexAIProvider)
