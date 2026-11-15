import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ProxyError(Exception):
    """An error with a known HTTP status, rendered in OpenAI's error format."""

    def __init__(self, status: int, message: str, kind: str = "upstream_error") -> None:
        super().__init__(message)
        self.status, self.message, self.kind = status, message, kind


def _json(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def register_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProxyError)
    async def _proxy(_: Request, exc: ProxyError) -> JSONResponse:
        return _json(exc.status, exc.message, exc.kind)

    @app.exception_handler(httpx.TimeoutException)
    async def _timeout(_: Request, __: httpx.TimeoutException) -> JSONResponse:
        return _json(504, "upstream timed out", "upstream_timeout")

    @app.exception_handler(httpx.HTTPError)
    async def _down(_: Request, __: httpx.HTTPError) -> JSONResponse:
        return _json(502, "upstream unavailable", "upstream_error")

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(f"{e['loc'][-1]} {e['msg']}" for e in exc.errors())
        return _json(400, f"invalid request: {detail}", "invalid_request_error")
