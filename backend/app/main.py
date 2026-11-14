from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend root is on sys.path so 'import backend.app.main' or 'import app.main' both resolve cleanly
_backend_root = str(Path(__file__).resolve().parent.parent)
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from app.core.config import settings  # noqa: E402
from app.core.database import init_db  # noqa: E402
from app.api.auth import router as auth_router  # noqa: E402
from app.api.health import router as health_router  # noqa: E402
from app.api.models_dist import router as models_router  # noqa: E402
from app.api.presets import router as presets_router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables
    try:
        init_db()
    except Exception:
        # Avoid crashing startup if DB is initializing
        pass
    yield
    # Shutdown logic if needed


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root-level health endpoint (e.g. for container healthchecks & monitoring)
app.include_router(health_router)

# API v1 routes
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(presets_router, prefix=settings.API_V1_STR)
app.include_router(models_router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
    }
