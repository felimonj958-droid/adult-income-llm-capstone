from fastapi import APIRouter

from src.api.routes.chat import router as chat_router
from src.api.routes.health import router as health_router
from src.api.routes.prediction import router as prediction_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(chat_router)
api_router.include_router(prediction_router)
