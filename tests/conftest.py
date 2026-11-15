import json
import struct
from collections.abc import Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

AUTH = {"Authorization": "Bearer proxy-secret"}


def frame(event: str, payload: dict) -> bytes:
    """Encode one AWS event-stream frame (CRCs are not checked by the decoder)."""
    headers = b""
    for name, value in ((":event-type", event), (":message-type", "event")):
        headers += bytes([len(name)]) + name.encode() + b"\x07" + struct.pack(">H", len(value)) + value.encode()
    body = json.dumps(payload).encode()
    return struct.pack(">III", 16 + len(headers) + len(body), len(headers), 0) + headers + body + b"\0\0\0\0"


@pytest.fixture
def make_client() -> Callable[[Callable[[httpx.Request], httpx.Response]], TestClient]:
    def build(handler: Callable[[httpx.Request], httpx.Response]) -> TestClient:
        settings = Settings(proxy_api_key="proxy-secret", bedrock_api_key="bedrock-secret", _env_file=None)
        return TestClient(create_app(settings, httpx.MockTransport(handler)))

    return build
