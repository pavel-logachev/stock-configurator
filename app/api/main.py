from fastapi import FastAPI

from app.api.routes import health, match
from app.core.config import get_settings
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title="Stock Configurator API",
        version="0.1.0",
        description=(
            "AI-first service that turns a request and available inventory into a "
            "reviewable configuration draft. Engineering approval remains mandatory."
        ),
    )
    app.include_router(health.router)
    app.include_router(match.router)
    return app


app = create_app()
