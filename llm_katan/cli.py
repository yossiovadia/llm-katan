"""
Command Line Interface for LLM Katan.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import asyncio
import logging
import sys

import click

from .config import ServerConfig
from .server import run_server

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("llm-katan")
except PackageNotFoundError:
    __version__ = "0.3.0"

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
    type=click.Choice(["transformers", "vllm"], case_sensitive=False),
    default="transformers",
    help="Backend (default: transformers)",
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
):
    """LLM Katan - One tiny model, every LLM API.

    Start a lightweight LLM server using real tiny models for testing.

    Examples:

        llm-katan --model Qwen/Qwen3-0.6B

        llm-katan --model Qwen/Qwen3-0.6B --port 8001 --name "gpt-4o"

        llm-katan --model Qwen/Qwen3-0.6B --backend vllm
    """
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

    provider_list = [p.strip() for p in providers.split(",")]

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
    )

    click.echo(f"LLM Katan v{__version__}")
    click.echo(f"  Model:     {config.model_name}")
    click.echo(f"  Served:    {config.served_model_name}")
    click.echo(f"  Backend:   {config.backend}")
    click.echo(f"  Device:    {config.device_auto}")
    if config.device_auto == "cpu":
        click.echo(f"  Quantize:  {'enabled' if config.quantize else 'disabled'}")
    click.echo(f"  Providers: {', '.join(config.providers)}")
    click.echo(f"  Server:    http://{config.host}:{config.port}")
    click.echo()

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
