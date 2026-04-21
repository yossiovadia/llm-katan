"""
Command Line Interface for LLM Katan.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import asyncio
import logging
import sys
from pathlib import Path

import click

from .config import ServerConfig
from .server import run_server

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("llm-katan")
except PackageNotFoundError:
    __version__ = "0.14.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model", "-m",
    required=True,
    help="Model name to load (e.g., 'Qwen/Qwen3-0.6B')",
)
@click.option(
    "--served-model-name", "--name", "-n",
    help="Model name to serve via API (defaults to model name)",
)
@click.option("--port", "-p", default=8000, type=int, help="Port (default: 8000)")
@click.option("--host", "-h", default="0.0.0.0", help="Host (default: 0.0.0.0)")
@click.option(
    "--backend", "-b",
    type=click.Choice(["transformers", "vllm", "echo"], case_sensitive=False),
    default="transformers",
    help="Backend: transformers, vllm, or echo (default: transformers)",
)
@click.option("--max-tokens", "--max", default=512, type=int, help="Max tokens (default: 512)")
@click.option("--temperature", "-t", default=0.7, type=float, help="Temperature (default: 0.7)")
@click.option(
    "--device", "-d",
    type=click.Choice(["auto", "cpu", "cuda"], case_sensitive=False),
    default="auto",
    help="Device (default: auto)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Log level (default: INFO)",
)
@click.option(
    "--quantize/--no-quantize",
    default=True,
    help="Enable int8 quantization for CPU (default: enabled)",
)
@click.option(
    "--max-concurrent",
    default=1, type=int,
    help="Max concurrent inference requests (default: 1)",
)
@click.option(
    "--providers",
    default="openai",
    help="Comma-separated list of API providers to enable (default: openai)",
)
@click.option(
    "--tls",
    is_flag=True,
    default=False,
    help="Enable HTTPS with auto-generated self-signed certificate",
)
@click.option(
    "--tls-cert",
    default="",
    help="Path to TLS certificate file (PEM). Enables HTTPS with custom cert (use with --tls-key).",
)
@click.option(
    "--tls-key",
    default="",
    help="Path to TLS private key file (PEM). Requires --tls-cert.",
)
@click.option(
    "--validate-keys",
    is_flag=True,
    default=False,
    help="Validate API key values (not just header presence). Uses defaults or --api-keys overrides.",
)
@click.option(
    "--api-keys",
    default="",
    help="Override API keys per provider: openai=mykey,anthropic=mykey2 (requires --validate-keys)",
)
@click.option(
    "--stats-file",
    default=str(Path.home() / ".llm-katan" / "stats.json"),
    show_default=True,
    help="Path to persistent stats file (tracks total requests across restarts)",
)
@click.option(
    "--error-rate",
    default=0.0, type=float,
    help="Probability (0.0-1.0) of returning HTTP 500 per request. Echo backend only. (default: 0.0)",
)
@click.option(
    "--latency-ms",
    default=0, type=int,
    help="Artificial delay in milliseconds added to every response. Echo backend only. (default: 0)",
)
@click.option(
    "--timeout-after",
    default=0, type=int,
    help="After N requests, return 504 for all subsequent requests. Echo backend only. (default: 0 = disabled)",
)
@click.option(
    "--rate-limit-after",
    default=0, type=int,
    help="After N requests, return 429 for all subsequent requests. Echo backend only. (default: 0 = disabled)",
)
@click.version_option(version=__version__, prog_name="llm-katan")
def main(
    model: str,
    served_model_name: str | None,
    port: int,
    host: str,
    backend: str,
    max_tokens: int,
    temperature: float,
    device: str,
    log_level: str,
    quantize: bool,
    max_concurrent: int,
    providers: str,
    tls: bool,
    tls_cert: str,
    tls_key: str,
    validate_keys: bool,
    api_keys: str,
    stats_file: str,
    error_rate: float,
    latency_ms: int,
    timeout_after: int,
    rate_limit_after: int,
):
    """LLM Katan - One tiny model, every LLM API.

    Start a lightweight LLM server using real tiny models for testing.

    Examples:

        llm-katan --model Qwen/Qwen3-0.6B

        llm-katan --model Qwen/Qwen3-0.6B --port 8001 --name "gpt-4o"

        llm-katan --model Qwen/Qwen3-0.6B --backend vllm
    """
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

    from .providers import available_providers

    provider_list = [p.strip() for p in providers.split(",")]
    available = available_providers()
    for p in provider_list:
        if p not in available:
            click.echo(f"Error: Unknown provider {p!r}. Available: {', '.join(available)}", err=True)
            sys.exit(1)

    # Parse api-keys overrides
    key_overrides = {}
    if api_keys:
        for pair in api_keys.split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                key_overrides[k.strip()] = v.strip()

    # --tls-cert/--tls-key implies TLS
    if tls_cert or tls_key:
        if not (tls_cert and tls_key):
            click.echo("Error: --tls-cert and --tls-key must be used together", err=True)
            sys.exit(1)
        tls = True

    config = ServerConfig(
        model_name=model,
        served_model_name=served_model_name,
        port=port,
        host=host,
        backend=backend.lower(),
        max_tokens=max_tokens,
        temperature=temperature,
        device=device.lower(),
        quantize=quantize,
        max_concurrent=max_concurrent,
        providers=provider_list,
        tls=tls,
        tls_cert=tls_cert or None,
        tls_key=tls_key or None,
        validate_keys=validate_keys,
        api_keys=key_overrides,
        stats_file=stats_file,
        error_rate=error_rate,
        latency_ms=latency_ms,
        timeout_after=timeout_after,
        rate_limit_after=rate_limit_after,
    )

    protocol = "https" if config.tls else "http"
    click.echo(f"LLM Katan v{__version__}")
    click.echo(f"  Model:     {config.model_name}")
    click.echo(f"  Served:    {config.served_model_name}")
    click.echo(f"  Backend:   {config.backend}")
    click.echo(f"  Device:    {config.device_auto}")
    if config.device_auto == "cpu":
        click.echo(f"  Quantize:  {'enabled' if config.quantize else 'disabled'}")
    click.echo(f"  Providers: {', '.join(config.providers)}")
    if config.tls:
        if config.tls_cert:
            click.echo(f"  TLS:       enabled ({config.tls_cert})")
        else:
            click.echo("  TLS:       enabled (self-signed)")
    if config.validate_keys:
        click.echo("  Keys:      validating (use --api-keys to override defaults)")
    has_failures = config.error_rate > 0 or config.latency_ms > 0 or config.timeout_after > 0 or config.rate_limit_after > 0
    if has_failures:
        parts = []
        if config.error_rate > 0:
            parts.append(f"error_rate={config.error_rate:.0%}")
        if config.latency_ms > 0:
            parts.append(f"latency={config.latency_ms}ms")
        if config.timeout_after > 0:
            parts.append(f"timeout_after={config.timeout_after}")
        if config.rate_limit_after > 0:
            parts.append(f"rate_limit_after={config.rate_limit_after}")
        click.echo(f"  Failures:  {', '.join(parts)}")
    click.echo(f"  Stats:     {config.stats_file}")
    click.echo(f"  Server:    {protocol}://{config.host}:{config.port}")
    click.echo()

    if config.backend != "echo":
        if config.backend == "vllm":
            try:
                import vllm  # noqa: F401
            except ImportError:
                click.echo("Error: vLLM not installed. pip install vllm", err=True)
                sys.exit(1)

        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            click.echo("Error: Missing deps. pip install transformers torch", err=True)
            sys.exit(1)

    try:
        asyncio.run(run_server(config))
    except KeyboardInterrupt:
        click.echo("\nServer stopped")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
