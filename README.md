# LLM Katan

One tiny model, every LLM API. A lightweight server that exposes real provider API formats (OpenAI, Anthropic, Vertex AI, AWS Bedrock, Azure OpenAI) backed by a single local model or an echo backend. Built for testing AI gateways, API translation layers, and multi-provider routing without burning API keys or cloud credits.

Katan means "small" in Hebrew.

## Features

- **Multi-Provider** — OpenAI, Anthropic, Vertex AI, AWS Bedrock (all 8 model families), Azure OpenAI
- **Real Inference** — runs actual tiny models (Qwen3-0.6B) via HuggingFace transformers or vLLM
- **Echo Mode** — instant startup, no model download, no GPU, no torch dependency
- **Auth Validation** — each provider requires its native auth header
- **Streaming** — all providers support SSE streaming in their native format
- **Live Dashboard** — real-time WebSocket-powered view of every request/response at `/dashboard`
- **Prometheus Metrics** — request counts, token usage, latency at `/metrics`
- **192 Tests** — extensive coverage for every provider, format, and edge case

## Quick Start

```bash
pip install llm-katan

# Echo mode (instant, no dependencies)
llm-katan --model my-test-model --backend echo --providers openai,anthropic,vertexai,bedrock,azure_openai

# Real model (needs torch + transformers)
llm-katan --model Qwen/Qwen3-0.6B --providers openai,anthropic,vertexai,bedrock,azure_openai
```

Then open `http://localhost:8000/dashboard` to watch requests flow through in real-time.

## How It Works

The server does not proxy to real providers. Each provider is a formatting layer around the same backend:

```
Request (any provider format)
       |
Provider (openai / anthropic / vertexai / bedrock / azure_openai)
  - Parses provider-specific request
  - Extracts: messages, max_tokens, temperature
       |
Backend (echo or real model)
  - Generates text (or echoes request metadata)
       |
Provider (same one)
  - Formats response in provider's native format
  - Returns to client
```

No translation chain, no SDK calls, no cloud API costs.

## Supported Providers

**OpenAI** (`--providers openai`)
- `POST /v1/chat/completions` — Auth: `Authorization: Bearer <key>`
- `GET /v1/models`

**Anthropic** (`--providers anthropic`)
- `POST /v1/messages` — Auth: `x-api-key: <key>`

**Vertex AI / Gemini** (`--providers vertexai`)
- `POST /v1beta/models/{model}:generateContent` — Auth: `Authorization: Bearer <token>`
- `POST /v1beta/models/{model}:streamGenerateContent`

**AWS Bedrock** (`--providers bedrock`)
- `POST /model/{modelId}/converse` — Auth: `Authorization: AWS4-HMAC-SHA256 <sig>`
- `POST /model/{modelId}/converse-stream`
- `POST /model/{modelId}/invoke` — auto-detects model family:

| Family | Model ID Prefix | Request Format |
|--------|----------------|----------------|
| Anthropic Claude | `anthropic.*` | `messages[]`, `max_tokens`, `system` |
| Amazon Nova | `amazon.nova*` | `messages[].content[].text`, `inferenceConfig` |
| Amazon Titan | `amazon.titan*` | `inputText`, `textGenerationConfig` |
| Meta Llama | `meta.llama*` | `prompt`, `max_gen_len` |
| Cohere Command | `cohere.*` | `message`, `chat_history[]` |
| Mistral | `mistral.*` | `prompt`, `max_tokens` |
| DeepSeek | `deepseek.*` | `prompt`, `max_tokens` |
| AI21 Jamba | `ai21.*` | `messages[]` (OpenAI-like) |

**Azure OpenAI** (`--providers azure_openai`)
- `POST /openai/deployments/{id}/chat/completions` — Auth: `api-key: <key>`

**Shared endpoints** (no auth)
- `GET /` — server info
- `GET /health` — health check
- `GET /metrics` — Prometheus metrics
- `GET /dashboard` — live request/response dashboard
- `GET /docs` — Swagger UI

## Example Requests

```bash
# OpenAI
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer test-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}'

# Anthropic
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: test-key" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}'

# Vertex AI
curl -X POST http://localhost:8000/v1beta/models/gemini-pro:generateContent \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}'

# Bedrock Converse
curl -X POST http://localhost:8000/model/anthropic.claude-v2/converse \
  -H "Authorization: AWS4-HMAC-SHA256 Credential=test" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":[{"text":"Hello"}]}]}'

# Azure OpenAI
curl -X POST "http://localhost:8000/openai/deployments/gpt-4/chat/completions?api-version=2024-10-21" \
  -H "api-key: test-key" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

## CLI Options

```
llm-katan [OPTIONS]

Required:
  -m, --model TEXT              Model name (or any string in echo mode)

Optional:
  -b, --backend [transformers|vllm|echo]  Backend (default: transformers)
  --providers TEXT              Comma-separated providers (default: openai)
  -p, --port INTEGER            Port (default: 8000)
  -n, --served-model-name TEXT  Model name in API responses
  --max-tokens INTEGER          Max tokens (default: 512)
  -t, --temperature FLOAT       Temperature (default: 0.7)
  -d, --device [auto|cpu|cuda]  Device (default: auto)
  --quantize/--no-quantize      CPU int8 quantization (default: enabled)
  --max-concurrent INTEGER      Concurrent requests (default: 1)
  --log-level [debug|info|warning|error]  Log level (default: INFO)
```

## Development

```bash
git clone https://github.com/yossiovadia/llm-katan.git
cd llm-katan
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Apache-2.0

---

*Created by [Yossi Ovadia](https://github.com/yossiovadia)*

### Contributors

- [Noy Itzikowitz](https://github.com/noyitz)
