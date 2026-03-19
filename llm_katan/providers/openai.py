"""
OpenAI Chat Completions API provider for llm-katan.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from llm_katan.model import ModelBackend

from . import register_provider
from .base import Provider

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = False


class OpenAIProvider(Provider):
    name = "openai"
    auth_header = "Authorization"

    def register_routes(self, app: FastAPI) -> None:
        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
            client_ip = raw_request.client.host if raw_request.client else "unknown"

            # Auth check
            auth_err = self.check_auth(dict(raw_request.headers))
            if auth_err:
                logger.warning(
                    "openai | %s | 401 | %s", client_ip, auth_err,
                )
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "message": auth_err,
                            "type": "invalid_request_error",
                            "code": "invalid_api_key",
                        }
                    },
                )

            logger.info(
                "openai | %s | model=%s messages=%d stream=%s max_tokens=%s temp=%s",
                client_ip, request.model, len(request.messages),
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
            model_name = self.backend.config.served_model_name

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
                        "openai | %s | 200 | stream | %d tokens | %.3fs",
                        client_ip, prompt_tokens + completion_tokens, elapsed,
                    )

                return StreamingResponse(
                    stream_response(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )

            elapsed = time.time() - start_time
            metrics.record(elapsed, prompt_tokens, completion_tokens)

            logger.info(
                "openai | %s | 200 | %d tokens | %.3fs",
                client_ip, prompt_tokens + completion_tokens, elapsed,
            )

            return self._full_response(response_id, created, model_name, generated_text, prompt_tokens, completion_tokens)

        @app.get("/v1/models")
        async def list_models():
            return {"object": "list", "data": [self.backend.get_model_info()]}

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
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
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
                    "logprobs": None,
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
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


register_provider("openai", OpenAIProvider)
