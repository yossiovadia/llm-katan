"""
Model backend implementations for LLM Katan.

Supports HuggingFace transformers and optionally vLLM for inference.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod

from .config import ServerConfig

logger = logging.getLogger(__name__)


class ModelBackend(ABC):
    """Abstract base class for model backends."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

    @abstractmethod
    async def load_model(self) -> None:
        """Load the model."""

    @abstractmethod
    async def _generate_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        """Generate text from messages. Subclasses implement this.

        Returns:
            (generated_text, prompt_tokens, completion_tokens)
        """

    async def generate_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        """Generate text with concurrency control. Called by providers."""
        async with self._semaphore:
            return await self._generate_text(messages, max_tokens, temperature)

    def get_model_info(self) -> dict[str, any]:
        """Get model information."""
        return {
            "id": self.config.served_model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "llm-katan",
        }

    def messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert chat messages to a prompt string.

        Uses the tokenizer's chat template if available, otherwise falls back
        to a simple format.
        """
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning("Chat template failed, using fallback: %s", e)

        parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)


class TransformersBackend(ModelBackend):
    """HuggingFace Transformers backend."""

    def __init__(self, config: ServerConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None

    async def load_model(self) -> None:
        logger.info("Loading model %s with transformers backend", self.config.model_name)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device = self.config.device_auto
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        if device == "cpu":
            self.model = self.model.to("cpu")
            if self.config.quantize:
                logger.info("Applying int8 quantization for CPU optimization")
                try:
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("Quantization applied successfully")
                except RuntimeError as e:
                    if "NoQEngine" in str(e):
                        logger.warning(
                            "Quantization not supported on this platform, "
                            "continuing with full precision"
                        )
                    else:
                        raise

        logger.info("Model loaded on %s", device)

    async def _generate_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        import torch

        prompt = self.messages_to_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        if self.config.device_auto == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        prompt_tokens = len(inputs["input_ids"][0])

        def _run():
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 1e-7),
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            return output[0]

        output_ids = await asyncio.to_thread(_run)

        # Decode only the newly generated tokens, not the prompt
        new_token_ids = output_ids[prompt_tokens:]
        completion_tokens = len(new_token_ids)
        generated_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        return generated_text, prompt_tokens, completion_tokens


class VLLMBackend(ModelBackend):
    """vLLM backend for efficient inference."""

    def __init__(self, config: ServerConfig):
        super().__init__(config)
        self.engine = None

    async def load_model(self) -> None:
        logger.info("Loading model %s with vLLM backend", self.config.model_name)

        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "vLLM is required for VLLMBackend. Install with: pip install vllm"
            ) from e

        self.engine = LLM(
            model=self.config.model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
        )
        logger.info("vLLM model loaded")

    async def _generate_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        from vllm.sampling_params import SamplingParams

        prompt = self.messages_to_prompt(messages)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["User:", "System:"],
        )

        outputs = await asyncio.to_thread(self.engine.generate, [prompt], sampling_params)
        output = outputs[0]
        generated_text = output.outputs[0].text.strip()
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)

        return generated_text, prompt_tokens, completion_tokens


class EchoBackend(ModelBackend):
    """Echo backend — returns request metadata without loading any model.

    No torch/transformers dependency. Instant startup, ~10MB memory.
    """

    def __init__(self, config: ServerConfig):
        super().__init__(config)
        import socket

        self._hostname = socket.gethostname()

    async def load_model(self) -> None:
        logger.info("Echo backend ready (no model loaded)")

    async def _generate_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        from datetime import datetime, timezone

        user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines = [
            f"[echo] model={self.config.served_model_name} host={self._hostname}:{self.config.port} "
            f"time={now} messages={len(messages)} max_tokens={max_tokens} temperature={temperature}",
            f"User: {user_msg}",
        ]
        generated = "\n".join(lines)

        # Fake token counts based on word count
        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
        completion_tokens = len(generated.split())

        return generated, prompt_tokens, completion_tokens


def create_backend(config: ServerConfig) -> ModelBackend:
    """Factory function to create the appropriate backend."""
    if config.backend == "echo":
        return EchoBackend(config)
    if config.backend == "vllm":
        return VLLMBackend(config)
    if config.backend == "transformers":
        return TransformersBackend(config)
    raise ValueError(f"Unknown backend: {config.backend}")
