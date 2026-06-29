from fastapi import APIRouter, status
from pydantic import BaseModel


router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str


@router.get("/", status_code=status.HTTP_200_OK)
def read_root() -> dict[str, str]:
    return {"message": "Adult Income Prediction API is running."}


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness health check",
)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="adult_income_api",
    )
