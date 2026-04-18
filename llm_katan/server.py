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
from .stats import PersistentStats

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("llm-katan")
except PackageNotFoundError:
    __version__ = "0.12.1"

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


def _detect_provider(path: str, headers: dict | None = None) -> str | None:
    """Detect which provider handled a request from the URL path and headers."""
    if path in _PROVIDER_ROUTES:
        provider = _PROVIDER_ROUTES[path]
        # Check if this is actually Bedrock's OpenAI-compatible endpoint (SigV4 auth on /v1/chat/completions)
        if provider == "openai" and headers:
            auth = headers.get("authorization", "")
            if auth.startswith("AWS4-HMAC-SHA256"):
                return "bedrock (openai-compat)"
        return provider
    if path.startswith("/v1beta/models/") or path.startswith("/v1/models/"):
        if ":generateContent" in path or ":streamGenerateContent" in path:
            return "vertexai"
    if path.startswith("/model/"):
        if "/converse" in path or "/invoke" in path:
            return "bedrock"
    if path.startswith("/openai/deployments/") or path.startswith("/openai/v1/"):
        return "azure_openai"
    if path.startswith("/v1/projects/") and "/chat/completions" in path:
        return "vertexai (openai-compat)"
    return None


class DashboardMiddleware(BaseHTTPMiddleware):
    """Captures request/response for every provider endpoint and broadcasts to dashboard."""

    _SKIP = {"/", "/health", "/metrics", "/stats", "/dashboard", "/docs", "/redoc", "/openapi.json", "/ws/events"}

    async def dispatch(self, request, call_next):
        path = request.url.path
        if path in self._SKIP:
            return await call_next(request)

        provider = _detect_provider(path, dict(request.headers))
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

        stats = getattr(request.app.state, "stats", None)
        if stats:
            stats.record(provider)

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
    app.state.stats = PersistentStats(config.stats_file)
    if config.stats_file:
        logger.info("Persistent stats: %s (%d total)", config.stats_file, app.state.stats.total)

    # Register provider routes
    for provider_name in config.providers:
        provider_cls = get_provider(provider_name)
        expected_key = config.get_expected_key(provider_name)
        provider = provider_cls(backend=backend, expected_key=expected_key)
        provider.register_routes(app)
        if expected_key:
            logger.info("Registered provider: %s (key validation: enabled)", provider_name)
        else:
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
        stats: PersistentStats = app.state.stats
        stats_data = stats.get()
        text += (
            f'\n# HELP llm_katan_lifetime_requests_total Total requests across all sessions\n'
            f'# TYPE llm_katan_lifetime_requests_total counter\n'
            f'llm_katan_lifetime_requests_total {stats_data["total"]}\n'
        )
        for prov, count in stats_data["providers"].items():
            text += (
                f'llm_katan_lifetime_provider_requests_total{{provider="{prov}"}} {count}\n'
            )

        return PlainTextResponse(content=text, media_type="text/plain")

    @app.get("/stats")
    async def get_stats():
        stats: PersistentStats = app.state.stats
        return stats.get()

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


def _generate_self_signed_cert():
    """Generate a self-signed TLS cert and key, return (certfile, keyfile) paths."""
    import datetime
    import ipaddress
    import tempfile

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "llm-katan"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("*"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv4Address("0.0.0.0")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    certfile = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    certfile.write(cert.public_bytes(serialization.Encoding.PEM))
    certfile.close()

    keyfile = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    keyfile.write(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    keyfile.close()

    return certfile.name, keyfile.name


async def run_server(config: ServerConfig):
    """Run the server with uvicorn."""
    import uvicorn

    app = create_app(config)

    ssl_kwargs = {}
    if config.tls:
        certfile, keyfile = _generate_self_signed_cert()
        ssl_kwargs = {"ssl_certfile": certfile, "ssl_keyfile": keyfile}
        logger.info("TLS enabled (self-signed certificate)")

    protocol = "https" if config.tls else "http"
    logger.info("Server URL: %s://%s:%d", protocol, config.host, config.port)

    server = uvicorn.Server(
        uvicorn.Config(app, host=config.host, port=config.port, log_level="info", **ssl_kwargs)
    )
    await server.serve()
