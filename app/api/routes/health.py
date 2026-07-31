import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import get_session

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)

SessionDep = Annotated[AsyncSession, Depends(get_session)]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    environment: str


class ReadinessResponse(BaseModel):
    status: Literal["ready"]
    service: str
    environment: str
    database: Literal["ok"]


@router.get("/health", response_model=HealthResponse, summary="Проверка состояния сервиса")
async def healthcheck(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        environment=settings.environment,
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Проверка готовности сервиса")
async def readiness(
    session: SessionDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReadinessResponse:
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("Readiness database probe failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service database is not ready.",
        ) from exc

    return ReadinessResponse(
        status="ready",
        service=settings.service_name,
        environment=settings.environment,
        database="ok",
    )
