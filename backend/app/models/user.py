"""SQLAlchemy database model for Lumora Users."""

from datetime import datetime, timezone
from app.core.database import Base, SQLALCHEMY_AVAILABLE

if SQLALCHEMY_AVAILABLE:
    from sqlalchemy import Boolean, Column, DateTime, Integer, String

    class UserModel(Base):
        __tablename__ = "users"

        id = Column(Integer, primary_key=True, index=True)
        username = Column(String(50), unique=True, index=True, nullable=False)
        email = Column(String(100), unique=True, index=True, nullable=False)
        hashed_password = Column(String(255), nullable=False)
        is_active = Column(Boolean, default=True)
        created_at = Column(
            DateTime,
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        )
else:
    # Fallback model representation for non-SQLAlchemy SQLite mode
    class UserModel:
        def __init__(
            self,
            id=None,
            username="",
            email="",
            hashed_password="",
            is_active=True,
            created_at=None,
        ):
            self.id = id
            self.username = username
            self.email = email
            self.hashed_password = hashed_password
            self.is_active = is_active
            self.created_at = created_at or datetime.now(timezone.utc).isoformat()


# Public model alias
User = UserModel
