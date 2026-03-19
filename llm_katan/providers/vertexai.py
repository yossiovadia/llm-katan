"""
Google Vertex AI / Gemini API provider for llm-katan.

Implements the Gemini generateContent API per the official spec at
ai.google.dev/api/generate-content.

Supports both the Gemini API URL format:
  POST /v1beta/models/{model}:generateContent
  POST /v1beta/models/{model}:streamGenerateContent

And the Vertex AI URL format:
  POST /v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent
  POST /v1/projects/{project}/locations/{location}/publishers/google/models/{model}:streamGenerateContent

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_katan.model import ModelBackend

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

    def register_routes(self, app: FastAPI) -> None:
        # Gemini API format: /v1beta/models/{model}:generateContent
        @app.post("/v1beta/models/{model}:generateContent")
        async def generate_content(model: str, raw_request: Request):
            return await self._handle_request(model, raw_request, app, stream=False)

        @app.post("/v1beta/models/{model}:streamGenerateContent")
        async def stream_generate_content(model: str, raw_request: Request):
            return await self._handle_request(model, raw_request, app, stream=True)

        # Also support /v1/ prefix (non-beta)
        @app.post("/v1/models/{model}:generateContent")
        async def generate_content_v1(model: str, raw_request: Request):
            return await self._handle_request(model, raw_request, app, stream=False)

        @app.post("/v1/models/{model}:streamGenerateContent")
        async def stream_generate_content_v1(model: str, raw_request: Request):
            return await self._handle_request(model, raw_request, app, stream=True)

    async def _handle_request(self, model: str, raw_request: Request, app: FastAPI, stream: bool):
        client_ip = raw_request.client.host if raw_request.client else "unknown"

        # Auth check
        auth_err = self.check_auth(dict(raw_request.headers))
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

        generated_text, prompt_tokens, completion_tokens = await self.backend.generate_text(
            backend_messages, max_tokens, temperature
        )

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
        logger.info(
            "vertexai | %s | 200 | %d tokens | %.3fs",
            client_ip, prompt_tokens + completion_tokens, elapsed,
        )

        return self._full_response(model_name, generated_text, prompt_tokens, completion_tokens)

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
