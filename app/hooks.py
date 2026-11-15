"""Request/response hooks. With PII enabled they de-identify prompts and restore replies.

Without `configure()` all hooks are no-ops. State lives in a per-request Vault held in a
ContextVar; the chat route calls `start_request()` before anything else.
"""

from contextvars import ContextVar
from typing import Any

from .pii import Detector, Vault

Message = dict[str, Any]

_detector: Detector | None = None
_vault: ContextVar[Vault | None] = ContextVar("pii_vault", default=None)


def configure(detector: Detector | None) -> None:
    global _detector
    _detector = detector


def start_request() -> None:
    _vault.set(Vault(_detector) if _detector else None)


def pre_request(messages: list[Message]) -> list[Message]:
    """Called once per request with OpenAI-style messages, before they go upstream."""
    vault = _vault.get()
    if vault is None:
        return messages
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            content = vault.redact(content)
        elif isinstance(content, list):
            content = [{**p, "text": vault.redact(p["text"])} if "text" in p else p for p in content]
        out.append({**m, "content": content})
    return out


def post_response(text: str) -> str:
    """Called on the full completion text, or on each streamed delta."""
    vault = _vault.get()
    return vault.feed(text) if vault else text


def finish() -> str:
    """Called once after the last delta; returns any text still held back."""
    vault = _vault.get()
    return vault.flush() if vault else ""
