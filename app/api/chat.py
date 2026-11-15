from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from .. import hooks
from ..bedrock.client import BedrockClient
from ..schemas import ChatRequest
from ..security import require_auth

router = APIRouter(prefix="/v1", dependencies=[Depends(require_auth)])


@router.post("/chat/completions", response_model=None)
async def chat_completions(req: ChatRequest, request: Request) -> Any:
    bedrock: BedrockClient = request.app.state.bedrock
    body = req.model_dump()
    hooks.start_request()
    if req.stream:
        resp = await bedrock.open_stream(body)  # errors surface as real HTTP codes before streaming starts
        return StreamingResponse(bedrock.sse(resp), media_type="text/event-stream")
    return await bedrock.complete(body)
