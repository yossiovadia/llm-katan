FROM python:3.12-slim

COPY . /app/llm-katan
WORKDIR /app/llm-katan
RUN pip install --no-cache-dir torch transformers numpy && \
    pip install --no-cache-dir .

# HuggingFace cache in a directory writable by OpenShift's random UID
ENV HF_HOME=/app/hf_cache
RUN mkdir -p /app/hf_cache && chmod -R 777 /app

# Pre-download model weights into the image
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')"

EXPOSE 8000

ENTRYPOINT ["llm-katan"]
CMD ["--model", "Qwen/Qwen3-0.6B", \
     "--providers", "openai,anthropic,vertexai,bedrock,azure_openai", \
     "--validate-keys", "--max-concurrent", "16", \
     "--enable-conversations"]
