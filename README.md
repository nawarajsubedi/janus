# Bedrock OpenAI-compatible proxy

FastAPI proxy that exposes `/v1/chat/completions` (JSON or SSE), `/v1/models` and `/health`
in OpenAI format, and forwards to **Claude on Amazon Bedrock** via the Converse API.

```
app/
  main.py          create_app(): wiring only
  config.py        env settings
  hooks.py         pre_request / post_response (PII plug-in points)
  security.py      bearer-token auth for the proxy
  errors.py        ProxyError + 502/504/400 mapping, OpenAI error shape
  middleware.py    request-ID logging (never logs content)
  schemas.py       request model
  api/             routes: health, models, chat
  bedrock/         client.py (HTTP), translate.py (OpenAI<->Converse), eventstream.py (stream decoder)
tests/             pytest + httpx.MockTransport
```

## Setup

```bash
cp .env.example .env     # set PROXY_API_KEY and AWS_BEARER_TOKEN_BEDROCK (Bedrock console -> API keys)
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/uvicorn app.main:create_app --factory --port 8001
.venv/bin/pytest
```

Docker: `docker compose up --build`. Enable model access for the Claude model in your Bedrock region first;
`BEDROCK_MODEL` takes a model ID or inference-profile ID.

## curl

```bash
export H='Authorization: Bearer change-me'
curl localhost:8001/health
curl -H "$H" localhost:8001/v1/models
curl -H "$H" -H 'Content-Type: application/json' localhost:8001/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Say hi"}]}'
curl -N -H "$H" -H 'Content-Type: application/json' localhost:8001/v1/chat/completions \
  -d '{"stream":true,"messages":[{"role":"user","content":"Count to 5"}]}'
```

## Notes

- The `model` field in requests is ignored; the proxy always uses `BEDROCK_MODEL`.
- Errors: upstream unreachable -> 502, timeout -> 504, upstream 400/429 passed through, other upstream errors -> 502.
- `hooks.post_response` runs per streamed delta; tokens split across deltas need per-request state in your hook.
- Not supported yet: tool calls, images, `n>1`, streamed usage.

## PII de-identification (optional)

`pip install -e '.[pii]'` then set `PII_ENABLED=true`. The first start downloads `nvidia/gliner-PII`.
Detected spans in every message are replaced with placeholders (`[PERSON_1]`, `[EMAIL_1]`, ...) before the
request goes to Bedrock; placeholders in the reply (full or streamed, even when split across chunks) are
swapped back to the original values. The mapping lives only in memory for that one request.
See `app/pii.py` (detector + vault) and `app/hooks.py` (wiring).
