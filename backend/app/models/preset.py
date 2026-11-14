"""SQLAlchemy database models for Lumora."""
from app.core.database import Base, SQLALCHEMY_AVAILABLE

if SQLALCHEMY_AVAILABLE:
    from sqlalchemy import Column, Integer, String, Float, Text, DateTime
    from datetime import datetime, timezone

    class PresetModel(Base):
        __tablename__ = "presets"

        id = Column(Integer, primary_key=True, index=True)
        name = Column(String(100), nullable=False, index=True)
        author = Column(String(100), nullable=False)
        category = Column(String(50), nullable=False, default="General")
        description = Column(String(255), nullable=True)
        data = Column(Text, nullable=False)  # JSON serialized .lumora or .xmp settings
        downloads = Column(Integer, default=0)
        rating = Column(Float, default=5.0)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
else:
    # Fallback model representation
    class PresetModel:
        def __init__(self, id=None, name="", author="", category="General", description="", data="{}", downloads=0, rating=5.0):
            self.id = id
            self.name = name
            self.author = author
            self.category = category
            self.description = description
            self.data = data
            self.downloads = downloads
            self.rating = rating
