"""
FastAPI server implementation for LLM Katan.

Provides OpenAI-compatible endpoints for lightweight LLM serving.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from .config import ServerConfig
from .model import ModelBackend, create_backend

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("llm-katan")
except PackageNotFoundError:
    __version__ = "0.2.0"

logger = logging.getLogger(__name__)

MAX_RECORDED_RESPONSE_TIMES = 1000


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = False


class ServerMetrics:
    """Request metrics with bounded memory."""

    def __init__(self):
        self.total_requests: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.response_times: deque[float] = deque(maxlen=MAX_RECORDED_RESPONSE_TIMES)
        self.start_time: float = time.time()

    def record(self, response_time: float, prompt_tokens: int, completion_tokens: int):
        self.total_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.response_times.append(response_time)

    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config: ServerConfig = app.state.config
    logger.info("Starting LLM Katan with model: %s", config.model_name)
    logger.info("Backend: %s | Served as: %s", config.backend, config.served_model_name)

    backend = create_backend(config)
    await backend.load_model()

    app.state.backend = backend
    app.state.metrics = ServerMetrics()

    logger.info("LLM Katan ready on %s:%d", config.host, config.port)
    yield
    logger.info("Shutting down LLM Katan")


def create_app(config: ServerConfig) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="LLM Katan",
        description="One tiny model, every LLM API.",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.config = config

    def _get_backend() -> ModelBackend:
        backend = getattr(app.state, "backend", None)
        if backend is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return backend

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": config.served_model_name,
            "backend": config.backend,
        }

    @app.get("/v1/models")
    async def list_models():
        backend = _get_backend()
        return {"object": "list", "data": [backend.get_model_info()]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        backend = _get_backend()
        metrics: ServerMetrics = app.state.metrics
        start_time = time.time()

        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        if request.stream:

            async def stream_response():
                prompt_tokens = 0
                completion_tokens = 0
                async for chunk in backend.generate(
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=True,
                ):
                    # Capture usage from final chunk
                    if "usage" in chunk:
                        prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                        completion_tokens = chunk["usage"].get("completion_tokens", 0)
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
                metrics.record(time.time() - start_time, prompt_tokens, completion_tokens)

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # Non-streaming
        response = await backend.generate(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=False,
        ).__anext__()

        elapsed = time.time() - start_time
        usage = response.get("usage", {})
        metrics.record(elapsed, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

        logger.info(
            "Response: %s | %d tokens | %.2fs",
            config.served_model_name,
            usage.get("total_tokens", 0),
            elapsed,
        )

        return response

    @app.get("/metrics")
    async def get_metrics():
        metrics: ServerMetrics = app.state.metrics
        uptime = time.time() - metrics.start_time

        text = (
            f'# HELP llm_katan_requests_total Total requests\n'
            f'# TYPE llm_katan_requests_total counter\n'
            f'llm_katan_requests_total{{model="{config.served_model_name}"}} '
            f'{metrics.total_requests}\n\n'
            f'# HELP llm_katan_prompt_tokens_total Total prompt tokens\n'
            f'# TYPE llm_katan_prompt_tokens_total counter\n'
            f'llm_katan_prompt_tokens_total{{model="{config.served_model_name}"}} '
            f'{metrics.total_prompt_tokens}\n\n'
            f'# HELP llm_katan_completion_tokens_total Total completion tokens\n'
            f'# TYPE llm_katan_completion_tokens_total counter\n'
            f'llm_katan_completion_tokens_total{{model="{config.served_model_name}"}} '
            f'{metrics.total_completion_tokens}\n\n'
            f'# HELP llm_katan_response_time_seconds Avg response time\n'
            f'# TYPE llm_katan_response_time_seconds gauge\n'
            f'llm_katan_response_time_seconds{{model="{config.served_model_name}"}} '
            f'{metrics.avg_response_time:.4f}\n\n'
            f'# HELP llm_katan_uptime_seconds Server uptime\n'
            f'# TYPE llm_katan_uptime_seconds gauge\n'
            f'llm_katan_uptime_seconds{{model="{config.served_model_name}"}} {uptime:.2f}\n'
        )
        return PlainTextResponse(content=text, media_type="text/plain")

    @app.get("/")
    async def root():
        return {
            "name": "LLM Katan",
            "version": __version__,
            "model": config.served_model_name,
            "backend": config.backend,
            "docs": "/docs",
        }

    return app


async def run_server(config: ServerConfig):
    """Run the server with uvicorn."""
    import uvicorn

    app = create_app(config)
    server = uvicorn.Server(
        uvicorn.Config(app, host=config.host, port=config.port, log_level="info")
    )
    await server.serve()
