from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # unknown OpenAI params are ignored, not rejected

    messages: list[dict[str, Any]]
    stream: bool = False
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
