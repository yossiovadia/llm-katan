# LLM Katan

One tiny model, every LLM API. A lightweight server that exposes real provider API formats (OpenAI, Anthropic, Vertex AI, AWS Bedrock, Azure OpenAI) backed by a single local model or an echo backend. Built for testing AI gateways, API translation layers, and multi-provider routing without burning API keys or cloud credits.

Katan means "small" in Hebrew.

## Features

- **Multi-Provider** — OpenAI, Anthropic, Vertex AI, AWS Bedrock (all 8 model families), Azure OpenAI
- **Real Inference** — runs actual tiny models (Qwen3-0.6B) via HuggingFace transformers or vLLM
- **Echo Mode** — instant startup, no model download, no GPU, no torch dependency
- **Auth Validation** — each provider requires its native auth header
- **Streaming** — all providers support SSE streaming in their native format
- **Live Dashboard** — real-time WebSocket-powered view of every request/response at `/dashboard`, with estimated cost savings
- **Cost Savings Tracker** — per-provider token tracking with estimated API cost savings, editable pricing via REST API
- **Prometheus Metrics** — request counts, token usage, latency at `/metrics`
- **Tool Calling** — all providers accept tool definitions and return tool call responses in native format (including Anthropic streaming tool_use)
- **Multimodal** — image content blocks accepted across all providers (OpenAI image_url, Anthropic image, Vertex inlineData, Bedrock image)
- **JSON Mode** — `response_format: {type: "json_object"}` returns valid JSON
- **Failure Simulation** — inject errors, latency, timeouts, and rate limits for gateway resilience testing
- **Inference Benchmarking** — configurable TTFT, inter-token latency, and chunk delay for streaming performance testing
- **363+ Tests** — extensive coverage for every provider, format, and edge case

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
- `GET /stats` — lifetime request counts, token usage, and estimated cost savings
- `GET /pricing` — current per-provider pricing table
- `PUT /pricing` — update pricing (JSON body, merged into current)
- `GET /dashboard` — live request/response dashboard with cost savings
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
  --tls                         Enable HTTPS with self-signed cert
  --tls-cert PATH               Custom TLS certificate (use with --tls-key)
  --tls-key PATH                Custom TLS private key
  --validate-keys               Enforce API key validation
  --api-keys TEXT               Override keys: openai=mykey,anthropic=mykey2
  --stats-file PATH             Persistent stats file (default: ~/.llm-katan/stats.json)
  --no-auto-tool-providers TEXT  Comma-separated providers that skip auto tool responses
  --workers, -w INTEGER          Number of uvicorn worker processes (default: 1)
  --disable-dashboard            Disable HTML dashboard (required with --workers > 1)
  --log-level [debug|info|warning|error]  Log level (default: INFO)

Failure Simulation (echo backend only):
  --error-rate FLOAT            Probability (0.0-1.0) of returning HTTP 500 (default: 0.0)
  --latency-ms INTEGER          Artificial delay per response in ms (default: 0)
  --timeout-after INTEGER       Return 504 after N successful requests (default: 0 = disabled)
  --rate-limit-after INTEGER    Return 429 after N requests (default: 0 = disabled)
  --max-inflight INTEGER        Return 503 when concurrent requests exceed limit (default: 0 = disabled)
  --chunk-delay-ms INTEGER      Delay between SSE streaming chunks in ms (default: 0)
  --ttft-ms INTEGER             Time to First Token delay in ms (default: 0)
  --itl-ms INTEGER              Inter-Token Latency between chunks in ms (default: 0)
```

## Tool Calling

When tools are included in a request, the simulator returns a tool call response using the first tool with dummy arguments generated from its parameter schema. This lets you test API translation pipelines that need to handle tool calling across providers.

```bash
# OpenAI — returns tool_calls with finish_reason: "tool_calls"
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "What is the weather in SF?"}],
    "tools": [{"type": "function", "function": {
      "name": "get_weather",
      "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
    }}]
  }'

# Anthropic — returns content block with type: "tool_use", stop_reason: "tool_use"
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: test-key" -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test", "max_tokens": 100,
    "messages": [{"role": "user", "content": "Weather in SF?"}],
    "tools": [{"name": "get_weather", "input_schema": {
      "type": "object", "properties": {"location": {"type": "string"}}
    }}]
  }'
```

Each provider uses its native tool calling format:

| Provider | Request field | Response format | Stop reason |
|----------|--------------|-----------------|-------------|
| OpenAI | `tools[].function` | `message.tool_calls[]` | `tool_calls` |
| Anthropic | `tools[].input_schema` | `content[{type: "tool_use"}]` | `tool_use` |
| Vertex AI | `tools[].functionDeclarations` | `parts[{functionCall}]` | `STOP` |
| Bedrock | `toolConfig.tools[].toolSpec` | `content[{toolUse}]` | `tool_use` |
| Azure | `tools[].function` | `message.tool_calls[]` | `tool_calls` |

Tool result messages (`role: "tool"` in OpenAI, `tool_result` blocks in Anthropic, `functionResponse` in Vertex, `toolResult` in Bedrock) are accepted in follow-up requests.

## Multimodal

Image content blocks are accepted in all providers. In echo mode, images are described as `[image:mime/type]` in the response without processing the actual image data.

```bash
# OpenAI — image_url content blocks
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
    ]}]
  }'
```

| Provider | Image format | Echo output |
|----------|-------------|-------------|
| OpenAI / Azure | `{type: "image_url", image_url: {url}}` | `[image:image/png]` |
| Anthropic | `{type: "image", source: {media_type, data}}` | `[image:image/png]` |
| Vertex AI | `{inlineData: {mimeType, data}}` | `[image:image/jpeg]` |
| Bedrock | `{image: {source: {format, bytes}}}` | `[image:png]` |

## Failure Simulation

When testing AI gateways and load balancers, you need to verify they handle provider failures correctly — retries, failover, circuit breaking. These flags let you simulate real-world failure modes without touching a real provider:

```bash
# 30% of requests fail with HTTP 500
llm-katan -m test --backend echo --error-rate 0.3 --providers openai

# Every response takes 2 seconds (simulates a slow provider)
llm-katan -m test --backend echo --latency-ms 2000 --providers openai

# Works fine for 100 requests, then goes down (504)
llm-katan -m test --backend echo --timeout-after 100 --providers openai

# Works fine for 50 requests, then rate-limits (429)
llm-katan -m test --backend echo --rate-limit-after 50 --providers openai

# Combine: slow + flaky
llm-katan -m test --backend echo --latency-ms 500 --error-rate 0.1 --providers openai

# Simulate realistic streaming latency (200ms TTFT, 30ms between tokens)
llm-katan -m test --backend echo --ttft-ms 200 --itl-ms 30 --providers openai

# Reject when more than 10 requests are in-flight (capacity overload)
llm-katan -m test --backend echo --max-inflight 10 --providers openai
```

Errors are returned in each provider's native error format — an OpenAI 429 looks different from a Bedrock 429 or an Anthropic 429, just like the real providers. The request counter for `--timeout-after` and `--rate-limit-after` is shared across all providers on the same instance.

**Example use case:** Run two llm-katan instances — one healthy, one with `--error-rate 0.3`. Point your AI gateway at both and verify it detects the degraded instance and shifts traffic to the healthy one.

## Cost Savings

The dashboard and `/stats` endpoint track estimated API cost savings — how much you would have spent if these requests hit real providers. Token counts are tracked per provider and multiplied by configurable pricing.

```bash
# View current stats with savings
curl -s http://localhost:8000/stats | python3 -m json.tool

# View pricing table
curl -s http://localhost:8000/pricing | python3 -m json.tool

# Update pricing (e.g., after a provider price change)
curl -X PUT http://localhost:8000/pricing \
  -H "Content-Type: application/json" \
  -d '{"anthropic": {"input": 3.00, "output": 15.00}}'
```

Default pricing (per 1M tokens):

| Provider | Input | Output | Based on |
|----------|-------|--------|----------|
| OpenAI | $2.50 | $10.00 | GPT-4o |
| Anthropic | $3.00 | $15.00 | Claude Sonnet |
| Vertex AI | $1.25 | $5.00 | Gemini 1.5 Pro |
| Bedrock | $3.00 | $15.00 | Claude via Bedrock |
| Azure OpenAI | $2.50 | $10.00 | GPT-4o via Azure |

Pricing overrides persist to the stats file and survive restarts.

## Selective Tool Responses

By default, any request with tools gets a tool call response. Use `--no-auto-tool-providers` to disable this per provider — useful when pointing AI agent clients (like Claude Code) at the simulator:

```bash
# Anthropic returns text even with tools, other providers return tool_calls normally
llm-katan -m test --backend echo --providers openai,anthropic --no-auto-tool-providers anthropic
```

The flag takes a comma-separated list of provider names. A listed provider still **accepts** tools in the request, but always responds with plain text instead of a tool call:

| Provider | In the list | Not in the list (default) |
|----------|-------------|---------------------------|
| Anthropic | text response, `stop_reason: end_turn` | tool_use block, `stop_reason: tool_use` |
| OpenAI | text response, `finish_reason: stop` | `tool_calls`, `finish_reason: tool_calls` |

**Why this exists:** AI agent clients run a tool-execution loop — they call a tool, feed the result back, and call again. Against the echo backend, every turn returns a tool call (the backend has no real model to decide otherwise), so the client loops forever on dummy tool calls. Listing the provider the client uses breaks the loop by returning text. Claude Code talks to the Anthropic provider, which is why the deployed dev instance runs `--no-auto-tool-providers anthropic`.

**Scope and impact — know before you set it:**

- **It's instance-wide.** Every request to a listed provider skips tools, for all clients. There's no per-request or per-client override — if you need both behaviors at once, run a second instance without the flag.
- **The behavior change is silent.** A client expecting a tool call from a listed provider just gets text — no error, no warning. If a tool-calling test suddenly sees plain text, check this flag first.
- **It disables tools for the *whole* provider, not specific tools.** You can't allow some tool calls and block others on the same provider.

So the common split is: list the provider your agent client uses (e.g. `anthropic` for Claude Code), and leave the providers your tool-calling tests hit (e.g. `openai` for gateway E2E tests) off the list.

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
