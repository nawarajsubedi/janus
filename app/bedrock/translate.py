"""Pure OpenAI <-> Converse shape conversions (no I/O)."""

import json
import time
from typing import Any

STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}
SYSTEM_ROLES = ("system", "developer")


def _text(content: object) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "")


def to_converse(messages: list[dict[str, Any]], body: dict[str, Any], default_max_tokens: int) -> dict[str, Any]:
    system = [{"text": _text(m["content"])} for m in messages if m["role"] in SYSTEM_ROLES]
    turns: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] in SYSTEM_ROLES:
            continue
        role = "assistant" if m["role"] == "assistant" else "user"
        block = {"text": _text(m["content"])}
        if turns and turns[-1]["role"] == role:  # Converse requires alternating roles
            turns[-1]["content"].append(block)
        else:
            turns.append({"role": role, "content": [block]})
    stop = body.get("stop")
    cfg = {
        "maxTokens": body.get("max_tokens") or body.get("max_completion_tokens") or default_max_tokens,
        "temperature": body.get("temperature"),
        "topP": body.get("top_p"),
        "stopSequences": [stop] if isinstance(stop, str) else stop,
    }
    payload: dict[str, Any] = {"messages": turns, "inferenceConfig": {k: v for k, v in cfg.items() if v is not None}}
    if system:
        payload["system"] = system
    return payload


def completion(cid: str, model: str, text: str, data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("usage", {})
    return {
        "id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": STOP_REASONS.get(data.get("stopReason", ""), "stop"),
        }],
        "usage": {
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
        },
    }


def sse_chunk(cid: str, model: str, delta: dict[str, Any], finish: str | None = None) -> str:
    obj = {
        "id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(obj)}\n\n"


def sse_error(message: str) -> str:
    return f'data: {json.dumps({"error": {"message": message, "type": "upstream_error"}})}\n\ndata: [DONE]\n\n'
