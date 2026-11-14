from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    database: str


class PresetBase(BaseModel):
    name: str = Field(..., max_length=100)
    author: str = Field(..., max_length=100)
    category: str = Field(default="General", max_length=50)
    description: Optional[str] = Field(None, max_length=255)
    data: str  # JSON payload of .lumora adjustments


class PresetCreate(PresetBase):
    pass


class PresetResponse(PresetBase):
    id: int
    downloads: int = 0
    rating: float = 5.0

    model_config = ConfigDict(from_attributes=True)

