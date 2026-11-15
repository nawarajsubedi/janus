import hmac
from typing import Annotated

from fastapi import Header, Request

from .errors import ProxyError


def require_auth(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    token = (authorization or "").removeprefix("Bearer ")
    expected = request.app.state.settings.proxy_api_key.get_secret_value()
    if not hmac.compare_digest(token.encode(), expected.encode()):
        raise ProxyError(401, "invalid or missing API key", "auth_error")
