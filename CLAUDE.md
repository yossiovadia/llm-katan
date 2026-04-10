# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

LLM Katan is a lightweight test server that exposes real LLM provider API formats (OpenAI, Anthropic, Vertex AI, AWS Bedrock, Azure OpenAI) backed by a single local model or an echo backend. It does **not** proxy to real providers — each provider is a formatting layer around the same inference backend. "Katan" means "small" in Hebrew.

## Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_anthropic.py -v

# Run a single test
pytest tests/test_anthropic.py::TestAuth::test_missing_auth_header_rejected -v

# Lint
ruff check .

# Lint with auto-fix
ruff check . --fix

# Run server (echo mode — no model download, no GPU)
llm-katan --model my-test-model --backend echo --providers openai,anthropic,vertexai,bedrock,azure_openai

# Run server (real model)
llm-katan --model Qwen/Qwen3-0.6B --providers openai,anthropic
```

## Architecture

```
CLI (cli.py / Click)
  → ServerConfig (config.py)
  → create_app() (server.py)
  → ModelBackend (model.py) — loads model once
  → Provider.register_routes() — mounts per-provider FastAPI routes
  → Uvicorn serves it all
```

**Providers** (`llm_katan/providers/`): Each provider subclasses `Provider` (in `base.py`) and implements two responsibilities:
1. Parse incoming requests from the provider's native format into a common form (messages, max_tokens, temperature)
2. Format the backend's output into the provider's native response format (including SSE streaming)

Providers: `openai.py`, `anthropic.py`, `vertexai.py`, `bedrock.py`, `azure_openai.py`

**Backends** (`model.py`): `ModelBackend` ABC with three implementations:
- `TransformersBackend` — HuggingFace transformers (default)
- `VLLMBackend` — vLLM for efficient inference
- `EchoBackend` — returns request metadata instantly, no model needed

**Dashboard** (`events.py` + `dashboard.html`): `DashboardMiddleware` captures all request/response pairs and broadcasts them via WebSocket to the live dashboard at `/dashboard`.

**Metrics**: `ServerMetrics` in `server.py` tracks request counts, token usage, and latency. Exposed at `/metrics` in Prometheus format.

## Key Patterns

- **Async-first**: FastAPI + async generators for SSE streaming. Tests use `pytest-asyncio` with `asyncio_mode = "auto"`.
- **Concurrency control**: Async semaphore in `ModelBackend` limits concurrent inference requests.
- **Auth per provider**: Each provider validates its own native auth headers (Bearer for OpenAI, x-api-key for Anthropic, AWS SigV4 for Bedrock, etc.). Auth is checked in the provider, not middleware.
- **Bedrock model families**: Bedrock auto-detects model family from the model ID prefix (8 families: Claude, Nova, Titan, Llama, Command, Mistral, DeepSeek, Jamba) and uses the appropriate request/response format.
- **Tests use a mock backend**: Test fixtures create the app with `EchoBackend` via `create_test_app()` so tests don't need a real model.

## Code Style

- Ruff for linting: line-length 100, target Python 3.10
- Rules: E, F, I, W (pycodestyle errors/warnings, pyflakes, isort)
