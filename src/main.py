from fastapi import FastAPI

from src.api.router import api_router
from src.core.logging import configure_logging
from src.core.middleware import register_middleware


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Adult Income LLM Capstone API")
    register_middleware(app)
    app.include_router(api_router)
    return app


app = create_app()
