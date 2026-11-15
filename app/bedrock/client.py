"""Async Bedrock Converse client: HTTP, error mapping, hook calls."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote

import httpx

from .. import hooks
from ..config import Settings
from ..errors import ProxyError
from . import translate
from .eventstream import parse_frames

IGNORED_EVENTS = {"messageStart", "contentBlockStart", "contentBlockStop", "metadata"}


class BedrockClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self.model = settings.bedrock_model
        self.http = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.bedrock_api_key.get_secret_value()}"},
            timeout=httpx.Timeout(settings.read_timeout, connect=settings.connect_timeout),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _request(self, op: str, body: dict[str, Any]) -> httpx.Request:
        messages = await asyncio.to_thread(hooks.pre_request, body["messages"])  # PII model is CPU-bound
        payload = translate.to_converse(messages, body, self.settings.default_max_tokens)
        return self.http.build_request("POST", f"/model/{quote(self.model, safe='')}/{op}", json=payload)

    @staticmethod
    def _raise_for_status(resp: httpx.Response, content: bytes) -> None:
        if resp.status_code < 400:
            return
        try:
            message = json.loads(content).get("message", "upstream error")
        except ValueError:
            message = "upstream error"
        # 400/429 are the caller's to fix; auth failures and 5xx are bad-gateway from the client's view
        raise ProxyError(resp.status_code if resp.status_code in (400, 429) else 502, message)

    async def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        resp = await self.http.send(await self._request("converse", body))
        self._raise_for_status(resp, resp.content)
        data = resp.json()
        text = "".join(b.get("text", "") for b in data["output"]["message"]["content"])
        return translate.completion(f"chatcmpl-{uuid.uuid4().hex}", self.model, hooks.post_response(text) + hooks.finish(), data)

    async def open_stream(self, body: dict[str, Any]) -> httpx.Response:
        resp = await self.http.send(await self._request("converse-stream", body), stream=True)
        if resp.status_code >= 400:
            content = await resp.aread()
            await resp.aclose()
            self._raise_for_status(resp, content)
        return resp

    async def sse(self, resp: httpx.Response) -> AsyncIterator[str]:
        cid, buf, finish = f"chatcmpl-{uuid.uuid4().hex}", bytearray(), "stop"
        try:
            yield translate.sse_chunk(cid, self.model, {"role": "assistant", "content": ""})
            async for raw in resp.aiter_bytes():
                buf += raw
                for kind, data in parse_frames(buf):
                    if kind == "contentBlockDelta" and "text" in data["delta"]:
                        text = hooks.post_response(data["delta"]["text"])
                        yield translate.sse_chunk(cid, self.model, {"content": text})
                    elif kind == "messageStop":
                        finish = translate.STOP_REASONS.get(data.get("stopReason", ""), "stop")
                    elif kind not in IGNORED_EVENTS and kind != "contentBlockDelta":
                        yield translate.sse_error(data.get("message", "upstream stream error"))
                        return
            if tail := hooks.finish():
                yield translate.sse_chunk(cid, self.model, {"content": tail})
            yield translate.sse_chunk(cid, self.model, {}, finish)
            yield "data: [DONE]\n\n"
        except httpx.HTTPError:  # headers already sent, so signal in-band
            yield translate.sse_error("upstream connection lost")
        finally:
            await resp.aclose()
