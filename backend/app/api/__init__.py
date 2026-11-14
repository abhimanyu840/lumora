from fastapi import APIRouter
from app.api.auth import router as auth_router
from app.api.health import router as health_router
from app.api.models_dist import router as models_router
from app.api.presets import router as presets_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(presets_router)
api_router.include_router(models_router)

__all__ = ["api_router", "auth_router", "health_router", "models_router", "presets_router"]
