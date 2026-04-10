"""
AWS Bedrock Converse API provider for llm-katan.

Implements the Bedrock Converse API per the official spec at
docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html

Endpoints:
  POST /model/{modelId}/converse
  POST /model/{modelId}/converse-stream

Also supports the InvokeModel endpoint with all model families:
  POST /model/{modelId}/invoke

The Converse API is the unified Bedrock format (model-agnostic).
InvokeModel routes to model-family-specific handlers based on the model ID:
Anthropic Claude, Amazon Nova, Amazon Titan, Meta Llama, Cohere,
Mistral, DeepSeek, AI21 Jamba.

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

    def _extract_sigv4_access_key(self, auth_value: str) -> str | None:
        """Extract the access key ID from SigV4 Authorization header.

        Format: AWS4-HMAC-SHA256 Credential=AKID/.../aws4_request, ...
        """
        try:
            cred_part = auth_value.split("Credential=")[1].split(",")[0]
            return cred_part.split("/")[0]
        except (IndexError, AttributeError):
            return None

    def check_auth(self, headers: dict) -> str | None:
        """Validate AWS SigV4 auth headers for Bedrock.

        Checks:
        1. Authorization header exists and starts with AWS4-HMAC-SHA256 or Bearer
        2. x-amz-date header is present (for SigV4)
        3. x-amz-security-token header (logged if missing, not rejected)
        4. Key value matches expected (when validate_keys enabled)
        """
        auth_value = None
        has_amz_date = False
        has_security_token = False

        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower == "authorization":
                auth_value = value
            elif key_lower == "x-amz-date":
                has_amz_date = True
            elif key_lower == "x-amz-security-token":
                has_security_token = True

        if auth_value is None:
            return "missing Authorization header"

        if not auth_value.startswith("AWS4-HMAC-SHA256"):
            if not auth_value.startswith("Bearer "):
                return "Authorization header must start with 'AWS4-HMAC-SHA256' or 'Bearer'"

        if auth_value.startswith("AWS4-HMAC-SHA256") and not has_amz_date:
            return "missing x-amz-date header (required for SigV4)"

        if auth_value.startswith("AWS4-HMAC-SHA256") and not has_security_token:
            logger.info("bedrock | x-amz-security-token not present (optional, only needed for temporary credentials)")

        # Key validation
        if self.expected_key is not None:
            if auth_value.startswith("AWS4-HMAC-SHA256"):
                actual = self._extract_sigv4_access_key(auth_value)
            else:
                actual = auth_value[7:]  # strip "Bearer "

            if actual != self.expected_key:
                return (
                    f"invalid API key for bedrock: "
                    f"got '{actual}', expected '{self.expected_key}'"
                )

        return None

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

        resp_body = self._converse_response(generated_text, prompt_tokens, completion_tokens, elapsed)
        return resp_body

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
    # InvokeModel dispatcher — routes to model-family-specific handler
    # ----------------------------------------------------------------

    # Model family detection: (prefix_in_model_id, handler_method_name)
    _MODEL_FAMILIES = [
        (("anthropic.", "claude"), "_invoke_anthropic"),
        (("amazon.nova",), "_invoke_nova"),
        (("meta.llama",), "_invoke_meta"),
        (("cohere.",), "_invoke_cohere"),
        (("mistral.",), "_invoke_mistral"),
        (("deepseek.",), "_invoke_deepseek"),
        (("ai21.",), "_invoke_ai21"),
        (("amazon.titan",), "_invoke_titan"),
    ]

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

        model_lower = model_id.lower()
        for prefixes, method_name in self._MODEL_FAMILIES:
            if any(model_lower.startswith(p) or p in model_lower for p in prefixes):
                handler = getattr(self, method_name)
                return await handler(model_id, body, app, client_ip)

        # Fallback: Amazon Titan format
        return await self._invoke_titan(model_id, body, app, client_ip)

    # ----------------------------------------------------------------
    # Helper: run generation and record metrics
    # ----------------------------------------------------------------

    async def _run_invoke(self, family: str, model_id: str, backend_messages: list,
                          max_tokens: int, temperature: float, app: FastAPI, client_ip: str):
        """Common invoke logic: generate text, record metrics, log."""
        metrics = app.state.metrics
        start_time = time.time()

        generated_text, prompt_tokens, completion_tokens = await self.backend.generate_text(
            backend_messages, max_tokens, temperature
        )

        elapsed = time.time() - start_time
        metrics.record(elapsed, prompt_tokens, completion_tokens)
        logger.info(
            "bedrock invoke %s | %s | model=%s | 200 | %d tokens | %.3fs",
            family, client_ip, model_id, prompt_tokens + completion_tokens, elapsed,
        )
        return generated_text, prompt_tokens, completion_tokens

    def _extract_anthropic_messages(self, body: dict) -> list[dict]:
        """Convert Anthropic-style messages to backend format."""
        backend_messages = []
        system_text = body.get("system")
        if isinstance(system_text, str) and system_text:
            backend_messages.append({"role": "system", "content": system_text})

        for msg in body.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = str(content)
            backend_messages.append({"role": role, "content": text})
        return backend_messages

    # ----------------------------------------------------------------
    # Model family handlers
    # ----------------------------------------------------------------

    async def _invoke_anthropic(self, model_id, body, app, client_ip):
        """Anthropic Claude — Anthropic Messages format."""
        messages = body.get("messages")
        if not messages:
            return _bedrock_error(400, "messages: field required")

        max_tokens = body.get("max_tokens", self.backend.config.max_tokens)
        temperature = body.get("temperature", self.backend.config.temperature)
        logger.info("bedrock invoke anthropic | %s | model=%s messages=%d", client_ip, model_id, len(messages))

        backend_messages = self._extract_anthropic_messages(body)
        text, pt, ct = await self._run_invoke("anthropic", model_id, backend_messages, max_tokens, temperature, app, client_ip)

        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": model_id,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": pt, "output_tokens": ct},
        }

    async def _invoke_nova(self, model_id, body, app, client_ip):
        """Amazon Nova — uses messages with content blocks + inferenceConfig (like Converse)."""
        messages = body.get("messages")
        if not messages:
            return _bedrock_error(400, "messages: field required")

        inf_config = body.get("inferenceConfig", {})
        max_tokens = inf_config.get("maxTokens", self.backend.config.max_tokens)
        temperature = inf_config.get("temperature", self.backend.config.temperature)
        logger.info("bedrock invoke nova | %s | model=%s messages=%d", client_ip, model_id, len(messages))

        backend_messages = []
        system = body.get("system")
        if system:
            sys_text = "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
            if sys_text:
                backend_messages.append({"role": "system", "content": sys_text})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            text = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
            backend_messages.append({"role": role, "content": text})

        text, pt, ct = await self._run_invoke("nova", model_id, backend_messages, max_tokens, temperature, app, client_ip)

        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": text}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": pt, "outputTokens": ct, "totalTokens": pt + ct},
        }

    async def _invoke_meta(self, model_id, body, app, client_ip):
        """Meta Llama — prompt/generation format."""
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_gen_len", self.backend.config.max_tokens)
        temperature = body.get("temperature", self.backend.config.temperature)
        logger.info("bedrock invoke meta | %s | model=%s", client_ip, model_id)

        backend_messages = [{"role": "user", "content": prompt}]
        text, pt, ct = await self._run_invoke("meta", model_id, backend_messages, max_tokens, temperature, app, client_ip)

        return {
            "generation": text,
            "prompt_token_count": pt,
            "generation_token_count": ct,
            "stop_reason": "stop",
        }

    async def _invoke_cohere(self, model_id, body, app, client_ip):
        """Cohere Command — message/chat_history format."""
        message = body.get("message", "")
        max_tokens = body.get("max_tokens", self.backend.config.max_tokens)
        temperature = body.get("temperature", self.backend.config.temperature)
        logger.info("bedrock invoke cohere | %s | model=%s", client_ip, model_id)

        backend_messages = []
        preamble = body.get("preamble")
        if preamble:
            backend_messages.append({"role": "system", "content": preamble})

        for turn in body.get("chat_history", []):
            role = "assistant" if turn.get("role") == "CHATBOT" else "user"
            backend_messages.append({"role": role, "content": turn.get("message", "")})

        backend_messages.append({"role": "user", "content": message})

        text, pt, ct = await self._run_invoke("cohere", model_id, backend_messages, max_tokens, temperature, app, client_ip)

        return {
            "response_id": uuid.uuid4().hex[:24],
            "text": text,
            "generation_id": uuid.uuid4().hex[:24],
            "finish_reason": "COMPLETE",
            "meta": {
                "api_version": {"version": "1"},
                "billed_units": {"input_tokens": pt, "output_tokens": ct},
            },
        }

    async def _invoke_mistral(self, model_id, body, app, client_ip):
        """Mistral — prompt/outputs format."""
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", self.backend.config.max_tokens)
        temperature = body.get("temperature", self.backend.config.temperature)
        logger.info("bedrock invoke mistral | %s | model=%s", client_ip, model_id)

        backend_messages = [{"role": "user", "content": prompt}]
        text, pt, ct = await self._run_invoke("mistral", model_id, backend_messages, max_tokens, temperature, app, client_ip)

        return {
            "outputs": [{"text": text, "stop_reason": "stop"}],
        }

    async def _invoke_deepseek(self, model_id, body, app, client_ip):
        """DeepSeek — prompt/choices format."""
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", self.backend.config.max_tokens)
        temperature = body.get("temperature", self.backend.config.temperature)
        logger.info("bedrock invoke deepseek | %s | model=%s", client_ip, model_id)

        backend_messages = [{"role": "user", "content": prompt}]
        text, pt, ct = await self._run_invoke("deepseek", model_id, backend_messages, max_tokens, temperature, app, client_ip)

        return {
            "choices": [{"text": text, "stop_reason": "stop"}],
        }

    async def _invoke_ai21(self, model_id, body, app, client_ip):
        """AI21 Jamba — OpenAI-like messages/choices format."""
        messages = body.get("messages")
        if not messages:
            return _bedrock_error(400, "messages: field required")

        max_tokens = body.get("max_tokens", self.backend.config.max_tokens)
        temperature = body.get("temperature", self.backend.config.temperature)
        logger.info("bedrock invoke ai21 | %s | model=%s messages=%d", client_ip, model_id, len(messages))

        backend_messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
        text, pt, ct = await self._run_invoke("ai21", model_id, backend_messages, max_tokens, temperature, app, client_ip)

        return {
            "id": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        }

    async def _invoke_titan(self, model_id, body, app, client_ip):
        """Amazon Titan — inputText/outputText format (also the fallback)."""
        input_text = body.get("inputText", "")
        gen_config = body.get("textGenerationConfig", {})
        max_tokens = gen_config.get("maxTokenCount", self.backend.config.max_tokens)
        temperature = gen_config.get("temperature", self.backend.config.temperature)
        logger.info("bedrock invoke titan | %s | model=%s", client_ip, model_id)

        backend_messages = [{"role": "user", "content": input_text}]
        text, pt, ct = await self._run_invoke("titan", model_id, backend_messages, max_tokens, temperature, app, client_ip)

        return {
            "inputTextTokenCount": pt,
            "results": [
                {"tokenCount": ct, "outputText": text, "completionReason": "FINISH"}
            ],
        }


register_provider("bedrock", BedrockProvider)
