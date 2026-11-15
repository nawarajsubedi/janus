import httpx
import pytest

from app import hooks
from tests.conftest import AUTH, frame

CHAT = {"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]}


def test_health_needs_no_auth(make_client):
    assert make_client(lambda r: httpx.Response(500)).get("/health").json() == {"status": "ok"}


def test_auth_required(make_client):
    client = make_client(lambda r: httpx.Response(500))
    assert client.get("/v1/models").status_code == 401
    assert client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_models(make_client):
    data = make_client(lambda r: httpx.Response(500)).get("/v1/models", headers=AUTH).json()
    assert data["data"][0]["id"].startswith("us.anthropic.claude")


def test_non_streaming(make_client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["auth"], seen["body"] = request.url.raw_path.decode(), request.headers["authorization"], request.read()
        return httpx.Response(200, json={
            "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
            "stopReason": "end_turn", "usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4},
        })

    resp = make_client(handler).post("/v1/chat/completions", json=CHAT, headers=AUTH)
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 4
    assert resp.headers["x-request-id"]
    assert seen["path"].endswith("/converse") and "%3A" in seen["path"]
    assert seen["auth"] == "Bearer bedrock-secret"
    assert b'"system":[{"text":"be brief"}]' in seen["body"]


def test_streaming(make_client):
    stream = b"".join([
        frame("messageStart", {"role": "assistant"}),
        frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hel"}}),
        frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "lo"}}),
        frame("messageStop", {"stopReason": "max_tokens"}),
        frame("metadata", {"usage": {}}),
    ])
    client = make_client(lambda r: httpx.Response(200, content=stream))
    with client.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}, headers=AUTH) as resp:
        lines = [l for l in resp.iter_lines() if l]
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert lines[-1] == "data: [DONE]"
    import json
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_hooks_apply_to_messages_and_stream_chunks(make_client, monkeypatch):
    monkeypatch.setattr(hooks, "pre_request", lambda m: [{**x, "content": x["content"].upper()} for x in m])
    monkeypatch.setattr(hooks, "post_response", lambda t: t.replace("a", "A"))
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, content=frame("contentBlockDelta", {"delta": {"text": "banana"}}))

    client = make_client(handler)
    with client.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}, headers=AUTH) as resp:
        assert "bAnAnA" in resp.read().decode()
    assert b'"HI"' in seen["body"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("exc, status", [(httpx.ConnectError("down"), 502), (httpx.ReadTimeout("slow"), 504)])
def test_error_mapping(make_client, exc, status, stream):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    resp = make_client(handler).post("/v1/chat/completions", json={**CHAT, "stream": stream}, headers=AUTH)
    assert resp.status_code == status
    assert "error" in resp.json()


@pytest.mark.parametrize("upstream, expected", [(429, 429), (400, 400), (403, 502), (500, 502)])
def test_upstream_status_mapping(make_client, upstream, expected):
    client = make_client(lambda r: httpx.Response(upstream, json={"message": "nope"}))
    resp = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    assert resp.status_code == expected and resp.json()["error"]["message"] == "nope"


def test_invalid_body(make_client):
    resp = make_client(lambda r: httpx.Response(500)).post("/v1/chat/completions", json={}, headers=AUTH)
    assert resp.status_code == 400
