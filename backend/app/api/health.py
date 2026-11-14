from fastapi import APIRouter
from app.core.config import settings
from app.schemas.preset import HealthResponse
from app.core.database import SQLALCHEMY_AVAILABLE

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return health status of the API and connected database mode."""
    db_mode = "sqlalchemy" if SQLALCHEMY_AVAILABLE else "sqlite_fallback"
    return HealthResponse(
        status="healthy",
        service="lumora-backend",
        version=settings.VERSION,
        database=db_mode,
    )
