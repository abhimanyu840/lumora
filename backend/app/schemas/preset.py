"""Pydantic schemas for Lumora presets and marketplace."""

from datetime import datetime
from typing import Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    database: str


class PresetBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    author: str = Field(..., min_length=1, max_length=100)
    category: str = Field(
        default="portrait",
        description="Category: portrait, landscape, vintage, cinematic, black_and_white, cyberpunk, etc.",
        max_length=50,
    )
    description: Optional[str] = Field(None, max_length=255)
    data: str = Field(..., description="JSON payload of .lumora or .xmp adjustments")
    tags: Optional[str] = Field(default="", description="Comma-separated tags e.g. 'moody,film,warm'")


class PresetCreate(PresetBase):
    author_id: Optional[int] = None


class PresetResponse(PresetBase):
    id: int
    author_id: Optional[int] = None
    downloads: int = 0
    downloads_count: int = 0
    upvotes: int = 0
    rating: float = 5.0
    created_at: Optional[Union[datetime, str]] = None

    model_config = ConfigDict(from_attributes=True)
