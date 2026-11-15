from typing import Any

from fastapi import APIRouter, Depends, Request

from ..security import require_auth

router = APIRouter(prefix="/v1", dependencies=[Depends(require_auth)])


@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    model = request.app.state.settings.bedrock_model
    return {"object": "list", "data": [{"id": model, "object": "model", "created": 0, "owned_by": "bedrock"}]}
