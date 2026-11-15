from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from .api import chat, health, models
from .bedrock.client import BedrockClient
from . import hooks
from .config import Settings
from .errors import register_handlers
from .middleware import request_id_logging


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings()  # type: ignore[call-arg]  # values come from env / .env
    bedrock = BedrockClient(settings, transport)
    if settings.pii_enabled:
        from .pii import GlinerDetector

        hooks.configure(GlinerDetector(settings.pii_model, settings.pii_threshold))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await bedrock.aclose()

    app = FastAPI(title="Bedrock OpenAI proxy", lifespan=lifespan)
    app.state.settings, app.state.bedrock = settings, bedrock
    app.middleware("http")(request_id_logging)
    register_handlers(app)
    for module in (health, models, chat):
        app.include_router(module.router)
    return app
