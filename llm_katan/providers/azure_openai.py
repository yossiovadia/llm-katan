"""
Azure OpenAI API provider for llm-katan.

Implements the Azure OpenAI chat completions API per the official spec at
learn.microsoft.com/en-us/azure/ai-services/openai/reference.

URL format:
  POST /openai/deployments/{deployment-id}/chat/completions?api-version={version}

Same request/response format as OpenAI, but different URL structure and auth
header (api-key instead of Authorization: Bearer).

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import register_provider
from .base import Provider

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None  # optional in Azure (deployment ID is in URL)
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = False


def _azure_error(status_code: int, message: str) -> JSONResponse:
    """Build an Azure OpenAI error response."""
    code_map = {
        400: "invalid_request",
        401: "invalid_api_key",
        403: "access_denied",
        404: "deployment_not_found",
        429: "rate_limit_exceeded",
        500: "internal_error",
    }
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code_map.get(status_code, "internal_error"),
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "param": None,
            }
        },
    )


class AzureOpenAIProvider(Provider):
    name = "azure_openai"
    auth_header = "api-key"  # primary, but we also accept Authorization: Bearer (Entra ID)

    def extract_key_value(self, headers: dict) -> str | None:
        """Azure: extract key from api-key header or Authorization: Bearer."""
        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower == "api-key":
                return value
            if key_lower == "authorization" and value.startswith("Bearer "):
                return value[7:]  # strip "Bearer "
        return None

    def check_auth(self, headers: dict) -> str | None:
        """Azure supports two auth methods: api-key header OR Authorization: Bearer (Entra ID/AAD)."""
        key_value = self.extract_key_value(headers)
        if key_value is None:
            return "missing api-key header or Authorization: Bearer token (Entra ID)"

        if self.expected_key is not None and key_value != self.expected_key:
            return (
                f"invalid API key for azure_openai: "
                f"got '{key_value}', expected '{self.expected_key}'"
            )
        return None

    def register_routes(self, app: FastAPI) -> None:
        @app.post("/openai/deployments/{deployment_id}/chat/completions")
        async def azure_chat_completions(deployment_id: str, raw_request: Request):
            client_ip = raw_request.client.host if raw_request.client else "unknown"

            # Auth check
            auth_err = self.check_auth(dict(raw_request.headers))
            if auth_err:
                logger.warning("azure_openai | %s | 401 | %s", client_ip, auth_err)
                return _azure_error(401, auth_err)

            # Parse request
            try:
                body = await raw_request.json()
            except Exception:
                logger.warning("azure_openai | %s | 400 | invalid JSON", client_ip)
                return _azure_error(400, "Invalid JSON in request body")

            try:
                request = ChatCompletionRequest(**body)
            except Exception as e:
                logger.warning("azure_openai | %s | 400 | %s", client_ip, e)
                return _azure_error(400, str(e))

            # api-version from query params (log it but don't enforce)
            api_version = raw_request.query_params.get("api-version", "unknown")

            logger.info(
                "azure_openai | %s | deployment=%s api-version=%s messages=%d stream=%s max_tokens=%s temp=%s",
                client_ip, deployment_id, api_version, len(request.messages),
                request.stream, request.max_tokens, request.temperature,
            )

            metrics = app.state.metrics
            start_time = time.time()

            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            max_tokens = request.max_tokens if request.max_tokens is not None else self.backend.config.max_tokens
            temperature = request.temperature if request.temperature is not None else self.backend.config.temperature

            generated_text, prompt_tokens, completion_tokens = await self.backend.generate_text(
                messages, max_tokens, temperature
            )

            response_id = f"chatcmpl-{int(time.time() * 1000)}"
            created = int(time.time())
            model_name = deployment_id  # Azure uses deployment ID as model in response

            if request.stream:
                async def stream_response():
                    chunk_size = 4
                    for i in range(0, len(generated_text), chunk_size):
                        chunk_text = generated_text[i: i + chunk_size]
                        yield f"data: {json.dumps(self._stream_chunk(response_id, created, model_name, chunk_text))}\n\n"

                    yield f"data: {json.dumps(self._final_chunk(response_id, created, model_name, prompt_tokens, completion_tokens))}\n\n"
                    yield "data: [DONE]\n\n"
                    elapsed = time.time() - start_time
                    metrics.record(elapsed, prompt_tokens, completion_tokens)
                    logger.info(
                        "azure_openai | %s | 200 | stream | %d tokens | %.3fs",
                        client_ip, prompt_tokens + completion_tokens, elapsed,
                    )

                return StreamingResponse(
                    stream_response(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )

            elapsed = time.time() - start_time
            metrics.record(elapsed, prompt_tokens, completion_tokens)

            logger.info("azure_openai | %s | 200 | %d tokens | %.3fs", client_ip, prompt_tokens + completion_tokens, elapsed)

            resp_body = self._full_response(response_id, created, model_name, generated_text, prompt_tokens, completion_tokens)
            return resp_body

    @staticmethod
    def _full_response(response_id, created, model, text, prompt_tokens, completion_tokens):
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    },
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "prompt_filter_results": [
                {
                    "prompt_index": 0,
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    },
                }
            ],
        }

    @staticmethod
    def _stream_chunk(response_id, created, model, text):
        return {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                }
            ],
        }

    @staticmethod
    def _final_chunk(response_id, created, model, prompt_tokens, completion_tokens):
        return {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


register_provider("azure_openai", AzureOpenAIProvider)
