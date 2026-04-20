"""Multi-turn test for all 5 providers — runs inside the llm-katan pod."""
import json
import urllib.request

BASE = "http://localhost:8000"
max_tokens = 32

def post(path, body, headers):
    req = urllib.request.Request(
        f"{BASE}{path}", method="POST",
        headers={**headers, "Content-Type": "application/json"},
        data=json.dumps(body).encode(),
    )
    return json.loads(urllib.request.urlopen(req).read())


def get(path):
    return json.loads(urllib.request.urlopen(f"{BASE}{path}").read())


def test_openai():
    print("=" * 60)
    print("OPENAI")
    print("=" * 60)
    h = {"Authorization": "Bearer llm-katan-openai-key"}

    r1 = post("/v1/chat/completions", {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "What is Kubernetes?"}],
    }, h)
    conv_id = r1["conversation_id"]
    print(f"Turn 1 | conv={conv_id}")
    print(f"  {r1['choices'][0]['message']['content'][:120]}")

    r2 = post("/v1/chat/completions", {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "How is it different from Docker?"}],
        "conversation_id": conv_id,
    }, h)
    print(f"Turn 2 | conv={r2['conversation_id']}")
    print(f"  {r2['choices'][0]['message']['content'][:120]}")

    r3 = post("/v1/chat/completions", {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "Give me a simple example"}],
        "conversation_id": conv_id,
    }, h)
    print(f"Turn 3 | conv={r3['conversation_id']}")
    print(f"  {r3['choices'][0]['message']['content'][:120]}")
    return conv_id


def test_anthropic():
    print("\n" + "=" * 60)
    print("ANTHROPIC")
    print("=" * 60)
    h = {"x-api-key": "llm-katan-anthropic-key", "anthropic-version": "2023-06-01"}

    r1 = post("/v1/messages", {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "What is Kubernetes?"}],
    }, h)
    conv_id = r1["metadata"]["conversation_id"]
    print(f"Turn 1 | conv={conv_id}")
    print(f"  {r1['content'][0]['text'][:120]}")

    r2 = post("/v1/messages", {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "How is it different from Docker?"}],
        "metadata": {"conversation_id": conv_id},
    }, h)
    print(f"Turn 2 | conv={r2['metadata']['conversation_id']}")
    print(f"  {r2['content'][0]['text'][:120]}")

    r3 = post("/v1/messages", {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "Give me a simple example"}],
        "metadata": {"conversation_id": conv_id},
    }, h)
    print(f"Turn 3 | conv={r3['metadata']['conversation_id']}")
    print(f"  {r3['content'][0]['text'][:120]}")
    return conv_id


def test_vertexai():
    print("\n" + "=" * 60)
    print("VERTEX AI")
    print("=" * 60)
    h = {"Authorization": "Bearer llm-katan-vertexai-key"}

    r1 = post("/v1beta/models/gemini-pro:generateContent", {
        "contents": [{"role": "user", "parts": [{"text": "What is Kubernetes?"}]}],
        "max_tokens": max_tokens,
    }, h)
    conv_id = r1["conversationId"]
    print(f"Turn 1 | conv={conv_id}")
    print(f"  {r1['candidates'][0]['content']['parts'][0]['text'][:120]}")

    r2 = post("/v1beta/models/gemini-pro:generateContent", {
        "contents": [{"role": "user", "parts": [{"text": "How is it different from Docker?"}]}],
        "conversation_id": conv_id,
        "max_tokens": max_tokens,
    }, h)
    print(f"Turn 2 | conv={r2['conversationId']}")
    print(f"  {r2['candidates'][0]['content']['parts'][0]['text'][:120]}")

    r3 = post("/v1beta/models/gemini-pro:generateContent", {
        "contents": [{"role": "user", "parts": [{"text": "Give me a simple example"}]}],
        "conversation_id": conv_id,
        "max_tokens": max_tokens,
    }, h)
    print(f"Turn 3 | conv={r3['conversationId']}")
    print(f"  {r3['candidates'][0]['content']['parts'][0]['text'][:120]}")
    return conv_id


def test_bedrock():
    print("\n" + "=" * 60)
    print("BEDROCK (Converse)")
    print("=" * 60)
    h = {
        "Authorization": (
            "AWS4-HMAC-SHA256 Credential=llm-katan-bedrock-key/20240101/us-east-1"
            "/bedrock/aws4_request, SignedHeaders=host, Signature=abc"
        ),
        "x-amz-date": "20240101T000000Z",
    }

    r1 = post("/model/anthropic.claude-v2/converse", {
        "messages": [{"role": "user", "content": [{"text": "What is Kubernetes?"}]}],
        "max_tokens": max_tokens,
    }, h)
    conv_id = r1["sessionId"]
    print(f"Turn 1 | session={conv_id}")
    print(f"  {r1['output']['message']['content'][0]['text'][:120]}")

    r2 = post("/model/anthropic.claude-v2/converse", {
        "messages": [{"role": "user", "content": [{"text": "How is it different from Docker?"}]}],
        "sessionId": conv_id,
        "max_tokens": max_tokens,
    }, h)
    print(f"Turn 2 | session={r2['sessionId']}")
    print(f"  {r2['output']['message']['content'][0]['text'][:120]}")

    r3 = post("/model/anthropic.claude-v2/converse", {
        "messages": [{"role": "user", "content": [{"text": "Give me a simple example"}]}],
        "sessionId": conv_id,
        "max_tokens": max_tokens,
    }, h)
    print(f"Turn 3 | session={r3['sessionId']}")
    print(f"  {r3['output']['message']['content'][0]['text'][:120]}")
    return conv_id


def test_azure():
    print("\n" + "=" * 60)
    print("AZURE OPENAI")
    print("=" * 60)
    h = {"api-key": "llm-katan-azure-key"}

    r1 = post("/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01", {
        "messages": [{"role": "user", "content": "What is Kubernetes?"}],
        "max_tokens": max_tokens,
    }, h)
    conv_id = r1["conversation_id"]
    print(f"Turn 1 | conv={conv_id}")
    print(f"  {r1['choices'][0]['message']['content'][:120]}")

    r2 = post("/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01", {
        "messages": [{"role": "user", "content": "How is it different from Docker?"}],
        "conversation_id": conv_id,
        "max_tokens": max_tokens,
    }, h)
    print(f"Turn 2 | conv={r2['conversation_id']}")
    print(f"  {r2['choices'][0]['message']['content'][:120]}")

    r3 = post("/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01", {
        "messages": [{"role": "user", "content": "Give me a simple example"}],
        "conversation_id": conv_id,
        "max_tokens": max_tokens,
    }, h)
    print(f"Turn 3 | conv={r3['conversation_id']}")
    print(f"  {r3['choices'][0]['message']['content'][:120]}")
    return conv_id


convos = []
convos.append(("openai", test_openai()))
convos.append(("anthropic", test_anthropic()))
convos.append(("vertexai", test_vertexai()))
convos.append(("bedrock", test_bedrock()))
convos.append(("azure", test_azure()))

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
store = get("/conversations")
print(f"Active conversations: {store['count']}")
for provider, cid in convos:
    conv = get(f"/conversations/{cid}")
    print(f"  {provider}: {conv['turn_count']} turns, {len(conv['messages'])} messages")

print("\nAll 5 providers passed!")
