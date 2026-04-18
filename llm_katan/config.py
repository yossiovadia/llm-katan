"""
Configuration management for LLM Katan.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import os
from dataclasses import dataclass, field

# Default API keys per provider — used when --validate-keys is enabled
# without explicit overrides. These are test keys, not secrets.
DEFAULT_API_KEYS = {
    "openai": "llm-katan-openai-key",
    "anthropic": "llm-katan-anthropic-key",
    "vertexai": "llm-katan-vertexai-key",
    "bedrock": "llm-katan-bedrock-key",
    "azure_openai": "llm-katan-azure-key",
}


@dataclass
class ServerConfig:
    """Configuration for the LLM Katan server."""

    model_name: str
    served_model_name: str | None = None
    port: int = 8000
    host: str = "0.0.0.0"
    backend: str = "transformers"  # "transformers" or "vllm"
    max_tokens: int = 512
    temperature: float = 0.7
    device: str = "auto"  # "auto", "cpu", "cuda"
    quantize: bool = True
    max_concurrent: int = 1
    providers: list[str] = field(default_factory=lambda: ["openai"])
    tls: bool = False
    tls_cert: str | None = None
    tls_key: str | None = None
    validate_keys: bool = False
    api_keys: dict[str, str] = field(default_factory=dict)
    stats_file: str | None = None

    def __post_init__(self):
        if self.served_model_name is None:
            self.served_model_name = self.model_name

        # Environment variable overrides
        self.model_name = os.getenv("LLM_KATAN_MODEL", self.model_name)
        self.port = int(os.getenv("LLM_KATAN_PORT", str(self.port)))
        self.backend = os.getenv("LLM_KATAN_BACKEND", self.backend)
        self.host = os.getenv("LLM_KATAN_HOST", self.host)

        if self.backend not in ("transformers", "vllm", "echo"):
            raise ValueError(f"Invalid backend: {self.backend}. Must be 'transformers', 'vllm', or 'echo'")

        # Build effective key map: defaults + overrides
        if self.validate_keys:
            effective = dict(DEFAULT_API_KEYS)
            effective.update(self.api_keys)
            self.api_keys = effective

    def get_expected_key(self, provider: str) -> str | None:
        """Get the expected API key for a provider. Returns None if validation is disabled."""
        if not self.validate_keys:
            return None
        return self.api_keys.get(provider)

    @property
    def device_auto(self) -> str:
        """Auto-detect the best device."""
        if self.device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device
