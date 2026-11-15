import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

log = logging.getLogger("llm_proxy")


async def request_id_logging(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    # metadata only: message content is never logged
    log.info("rid=%s %s %s -> %s %.0fms", rid, request.method, request.url.path, response.status_code,
             (time.perf_counter() - start) * 1000)
    return response
