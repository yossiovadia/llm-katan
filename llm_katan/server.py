"""
FastAPI server implementation for LLM Katan.

Mounts provider-specific routes (OpenAI, Anthropic, etc.) on a single server.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import ServerConfig
from .events import broadcaster, make_event
from .model import create_backend
from .providers import get_provider

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("llm-katan")
except PackageNotFoundError:
    __version__ = "0.8.1"

logger = logging.getLogger(__name__)

MAX_RECORDED_RESPONSE_TIMES = 1000

# Load dashboard HTML once at import time
_DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text()


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


# Provider detection from URL path
_PROVIDER_ROUTES = {
    "/v1/chat/completions": "openai",
    "/v1/messages": "anthropic",
    "/v1/models": "openai",
}


def _detect_provider(path: str) -> str | None:
    """Detect which provider handled a request from the URL path."""
    if path in _PROVIDER_ROUTES:
        return _PROVIDER_ROUTES[path]
    if path.startswith("/v1beta/models/") or path.startswith("/v1/models/"):
        if ":generateContent" in path or ":streamGenerateContent" in path:
            return "vertexai"
    if path.startswith("/model/"):
        if "/converse" in path or "/invoke" in path:
            return "bedrock"
    if path.startswith("/openai/deployments/"):
        return "azure_openai"
    return None


class DashboardMiddleware(BaseHTTPMiddleware):
    """Captures request/response for every provider endpoint and broadcasts to dashboard."""

    _SKIP = {"/", "/health", "/metrics", "/dashboard", "/docs", "/redoc", "/openapi.json", "/ws/events"}

    async def dispatch(self, request, call_next):
        path = request.url.path
        if path in self._SKIP:
            return await call_next(request)

        provider = _detect_provider(path)
        if provider is None:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        method = request.method

        # Capture request body
        req_body = None
        try:
            body_bytes = await request.body()
            if body_bytes:
                req_body = json.loads(body_bytes)
        except Exception:
            pass

        # Capture request headers (skip noisy ones)
        skip_headers = {"host", "content-length", "connection", "accept-encoding", "accept"}
        req_headers = {k: v for k, v in request.headers.items() if k.lower() not in skip_headers}

        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start

        # Capture response body
        resp_body = None
        resp_body_bytes = b""
        async for chunk in response.body_iterator:
            if isinstance(chunk, str):
                resp_body_bytes += chunk.encode()
            else:
                resp_body_bytes += chunk

        try:
            resp_body = json.loads(resp_body_bytes)
        except Exception:
            # Streaming or non-JSON response
            resp_body = {"_raw": resp_body_bytes[:500].decode(errors="replace")} if resp_body_bytes else None

        # Rebuild response with consumed body
        from starlette.responses import Response
        new_response = Response(
            content=resp_body_bytes,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

        # Broadcast to dashboard
        event = make_event(
            provider=provider,
            method=method,
            path=path,
            status_code=response.status_code,
            client_ip=client_ip,
            latency_ms=int(elapsed * 1000),
            request_headers=req_headers,
            request_body=req_body,
            response_body=resp_body,
        )
        await broadcaster.broadcast(event)

        return new_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    config: ServerConfig = app.state.config
    logger.info("Starting LLM Katan with model: %s", config.model_name)
    logger.info("Backend: %s | Served as: %s", config.backend, config.served_model_name)
    logger.info("Providers: %s", ", ".join(config.providers))

    backend = create_backend(config)
    await backend.load_model()

    app.state.backend = backend
    app.state.metrics = ServerMetrics()

    # Register provider routes
    for provider_name in config.providers:
        provider_cls = get_provider(provider_name)
        provider = provider_cls(backend=backend)
        provider.register_routes(app)
        logger.info("Registered provider: %s", provider_name)

    logger.info("Dashboard: http://%s:%d/dashboard", config.host, config.port)
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
    app.add_middleware(DashboardMiddleware)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": config.served_model_name,
            "backend": config.backend,
            "providers": config.providers,
        }

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

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return _DASHBOARD_HTML

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket):
        await broadcaster.connect(ws)
        try:
            while True:
                await ws.receive_text()  # keep connection alive
        except WebSocketDisconnect:
            broadcaster.disconnect(ws)

    @app.get("/")
    async def root():
        return {
            "name": "LLM Katan",
            "version": __version__,
            "model": config.served_model_name,
            "backend": config.backend,
            "providers": config.providers,
            "docs": "/docs",
            "dashboard": "/dashboard",
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
