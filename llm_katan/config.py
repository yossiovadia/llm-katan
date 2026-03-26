"""
Configuration management for LLM Katan.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import os
from dataclasses import dataclass, field


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
