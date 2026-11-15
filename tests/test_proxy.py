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


class FakeDetector:
    """Stands in for GLiNER: flags fixed strings."""

    FOUND = {"Alice Smith": "person", "a@b.co": "email", "Pokhara Nepal": "address", "1990-01-01": "date of birth"}

    def detect(self, text):
        return [
            {"start": i, "end": i + len(v), "label": label, "text": v}
            for v, label in self.FOUND.items()
            if (i := text.find(v)) >= 0
        ]


@pytest.fixture
def pii(monkeypatch):
    hooks.configure(FakeDetector())
    yield
    hooks.configure(None)


def _stream_text(client, content):
    import json

    stream_body = {"stream": True, "messages": [{"role": "user", "content": content}]}
    with client.stream("POST", "/v1/chat/completions", json=stream_body, headers=AUTH) as resp:
        text = resp.read().decode()
    return "".join(
        json.loads(l[6:])["choices"][0]["delta"].get("content", "")
        for l in text.splitlines() if l.startswith("data: {")
    )


def test_pii_names_become_fake_names_and_roundtrip(make_client, pii):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        reply = "Hello James Wilson! James's plan is ready, thanks Wilson."
        return httpx.Response(200, json={"output": {"message": {"content": [{"text": reply}]}}})

    body = {"messages": [{"role": "user", "content": "I am Alice Smith"}]}
    resp = make_client(handler).post("/v1/chat/completions", json=body, headers=AUTH)
    assert "Alice" not in seen["body"] and "James Wilson" in seen["body"] and "[PERSON" not in seen["body"]
    assert resp.json()["choices"][0]["message"]["content"] == "Hello Alice Smith! Alice's plan is ready, thanks Smith."


def test_pii_non_name_values_keep_bracket_tokens(make_client, pii):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"output": {"message": {"content": [{"text": "Born [DATE_OF_BIRTH_1] ok"}]}}})

    body = {"messages": [{"role": "user", "content": "I was born 1990-01-01"}]}
    resp = make_client(handler).post("/v1/chat/completions", json=body, headers=AUTH)
    assert "1990" not in seen["body"] and "[DATE_OF_BIRTH_1]" in seen["body"]
    assert resp.json()["choices"][0]["message"]["content"] == "Born 1990-01-01 ok"


def test_pii_fake_name_avoids_names_already_in_text(make_client, pii):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"output": {"message": {"content": [{"text": "ok"}]}}})

    body = {"messages": [{"role": "user", "content": "James asked about Alice Smith"}]}
    make_client(handler).post("/v1/chat/completions", json=body, headers=AUTH)
    assert "Alice" not in seen["body"] and "Emma Turner" in seen["body"]  # "James" was taken


def test_pii_restores_placeholders_split_across_stream_chunks(make_client, pii):
    chunks = ["Hi [DATE_OF", "_BIRTH_", "1], Ja", "mes Wil", "son and James", "'s dog, bye [x"]
    stream = b"".join(frame("contentBlockDelta", {"delta": {"text": t}}) for t in chunks)
    client = make_client(lambda r: httpx.Response(200, content=stream))
    out = _stream_text(client, "I am Alice Smith, born 1990-01-01")
    assert out == "Hi 1990-01-01, Alice Smith and Alice's dog, bye [x"


def test_pii_address_becomes_fake_address_and_roundtrips(make_client, pii):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"output": {"message": {"content": [{"text": "Springfield Ohio is lovely; Springfield too."}]}}})

    body = {"messages": [{"role": "user", "content": "I live in Pokhara Nepal"}]}
    resp = make_client(handler).post("/v1/chat/completions", json=body, headers=AUTH)
    assert "Pokhara" not in seen["body"] and "Springfield Ohio" in seen["body"] and "[ADDRESS" not in seen["body"]
    assert resp.json()["choices"][0]["message"]["content"] == "Pokhara Nepal is lovely; Pokhara too."


class SplitDetector:
    """Mimics GLiNER splitting one thing into several spans."""

    def __init__(self, spans):
        self.spans = spans

    def detect(self, text):
        out = []
        for value, label in self.spans:
            i = text.find(value)
            if i >= 0:
                out.append({"start": i, "end": i + len(value), "label": label, "text": value})
        return out


def _roundtrip(spans, text, reply=None):
    from app.pii import Vault

    v = Vault(SplitDetector(spans))
    sent = v.redact(text)
    return sent, v.restore(reply if reply is not None else sent)


def test_split_name_is_merged_into_one_english_name():
    sent, back = _roundtrip([("Emily", "person"), ("Johnson", "person")], "Hi, I'm Emily Johnson from here")
    assert sent == "Hi, I'm James Wilson from here" and back == "Hi, I'm Emily Johnson from here"


def test_city_and_state_merge_and_keep_the_comma():
    sent, back = _roundtrip([("Austin", "address"), ("Texas", "address")], "I live in Austin, Texas today")
    assert sent == "I live in Springfield, Ohio today" and back == "I live in Austin, Texas today"


def test_street_gets_a_street_shaped_fake():
    sent, back = _roundtrip([("245 Maple Street", "address")], "I live at 245 Maple Street, Denver")
    assert sent == "I live at 412 Oak Avenue, Denver" and back == "I live at 245 Maple Street, Denver"


def test_lone_us_state_gets_a_fake_state_not_a_city():
    sent, _ = _roundtrip([("Texas", "address")], "from Texas")
    assert sent == "from Ohio"


def test_two_separate_people_are_not_merged():
    sent, back = _roundtrip([("Alice", "person"), ("Bob", "person")], "Alice, Bob and I")
    assert sent == "James, Emma and I" and back == "Alice, Bob and I"
