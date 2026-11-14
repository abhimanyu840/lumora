"""SQLAlchemy database models for Lumora Presets."""

from datetime import datetime, timezone
from app.core.database import Base, SQLALCHEMY_AVAILABLE

if SQLALCHEMY_AVAILABLE:
    from sqlalchemy import Column, DateTime, Float, Integer, String, Text

    class PresetModel(Base):
        __tablename__ = "presets"

        id = Column(Integer, primary_key=True, index=True)
        name = Column(String(100), nullable=False, index=True)
        author = Column(String(100), nullable=False)
        author_id = Column(Integer, nullable=True, index=True)
        category = Column(String(50), nullable=False, default="portrait")
        description = Column(String(255), nullable=True)
        data = Column(Text, nullable=False)  # JSON serialized .lumora or .xmp settings
        tags = Column(String(255), default="")  # Comma-separated tags
        downloads = Column(Integer, default=0)
        upvotes = Column(Integer, default=0)
        rating = Column(Float, default=5.0)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

        @property
        def downloads_count(self) -> int:
            return self.downloads or 0
else:
    # Fallback model representation for host SQLite mode
    class PresetModel:
        def __init__(
            self,
            id=None,
            name="",
            author="",
            author_id=None,
            category="portrait",
            description="",
            data="{}",
            tags="",
            downloads=0,
            upvotes=0,
            rating=5.0,
            created_at=None,
        ):
            self.id = id
            self.name = name
            self.author = author
            self.author_id = author_id
            self.category = category
            self.description = description
            self.data = data
            self.tags = tags
            self.downloads = downloads
            self.upvotes = upvotes
            self.rating = rating
            self.created_at = created_at or datetime.now(timezone.utc).isoformat()

        @property
        def downloads_count(self) -> int:
            return self.downloads or 0


# Public model alias
Preset = PresetModel
