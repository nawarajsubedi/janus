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

Enable model access for the Claude model in your Bedrock region first; `BEDROCK_MODEL` takes a model ID or
inference-profile ID.

## Docker

```bash
scripts/fetch_wheels.sh        # one-time: pre-download the Linux wheels into ./wheels (git-ignored, ~220 MB)
docker compose up -d --build   # or docker-compose
curl localhost:8001/health
```

- The image installs **offline** from `./wheels` (CPU-only torch + gliner + fastapi ...), so the build does not
  depend on a slow PyPI. If PyPI is slow, use a mirror: `PIP_INDEX_URL=<mirror>/simple scripts/fetch_wheels.sh`.
  On an Intel/AMD Docker host run `ARCH=x86_64 scripts/fetch_wheels.sh`.
- `docker-compose.yml` sets `PII_ENABLED=true`. The first start downloads `nvidia/gliner-PII` into the `models`
  volume (health check `start_period` is 120 s); later starts reuse it.
- Memory: one server process holds about 4.2 GiB idle (about 2.5 GiB during a request, measured). With Docker's
  default 8 GB do not load a second copy of the model in the same container (for example a second Python process
  that builds `GlinerDetector`): the container is OOM-killed (exit 137).
- `restart: unless-stopped` brings the container back after a crash or OOM kill; the model reloads, so allow
  roughly 20-60 s before `/health` answers again.

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
Detected spans in every message are replaced **before** the request goes to Bedrock, and swapped back in the
reply (full or streamed, even when a value is split across chunks). The mapping lives only in memory for that
one request.

| Detected | Sent to Claude (example) |
|---|---|
| person | realistic English name: `Emily Johnson` -> `James Wilson` (first/last alone also restored, plus `'s`) |
| address | city/state `Austin, Texas` -> `Springfield, Ohio`; street `245 Maple Street` -> `412 Oak Avenue`; lone US state -> fake state |
| email | `name@example.com`-style fake on the reserved `example.com` domain |
| phone number | same punctuation, fictional digits: `(415) 555-0132` -> `(202) 555-0100` |
| anything else | bracket token such as `[DATE_OF_BIRTH_1]` |

Adjacent spans of one kind are merged first (GLiNER often returns "Emily" and "Johnson" separately), so each
person or place gets one fake value. Each pool holds 12 values per kind (100 for phones); past that, or for
labels with no fake, the bracket token is used.

Limits: values the detector misses are sent as written (for example a country or a state it did not tag); a fake
that is a real place name can lead the model to state facts about that real place; the fake email is not derived
from the fake name.
See `app/pii.py` (detector + vault) and `app/hooks.py` (wiring).
