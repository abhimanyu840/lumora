import logging
import sqlite3
from typing import Generator, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import declarative_base, sessionmaker
    
    # Configure SQLite or PostgreSQL connect args
    connect_args = {}
    if settings.DATABASE_URL.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
        
    engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    # Graceful fallback for local development without SQLAlchemy installed
    logger.warning("SQLAlchemy not installed on host. Running in fallback SQLite mode.")
    SQLALCHEMY_AVAILABLE = False
    
    class FallbackBase:
        pass
        
    Base = FallbackBase
    engine = None
    
    class FallbackSession:
        def __init__(self, db_path: str = "lumora.db"):
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            
        def close(self):
            self.conn.close()
            
        def commit(self):
            self.conn.commit()
            
        def rollback(self):
            self.conn.rollback()

        def execute(self, sql: str, params=()):
            return self.conn.execute(sql, params)
            
    SessionLocal = FallbackSession


def get_db() -> Generator[Any, None, None]:
    """Dependency for providing database sessions to endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables."""
    if SQLALCHEMY_AVAILABLE and engine is not None:
        import app.models.preset  # noqa: F401
        Base.metadata.create_all(bind=engine)
    else:
        # Create minimal fallback SQLite tables
        conn = sqlite3.connect("lumora.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS presets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                author TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                data TEXT NOT NULL,
                downloads INTEGER DEFAULT 0,
                rating REAL DEFAULT 5.0
            )
        """)
        # Ensure description column exists if table was created previously without it
        columns = [col[1] for col in cursor.execute("PRAGMA table_info(presets)").fetchall()]
        if "description" not in columns:
            cursor.execute("ALTER TABLE presets ADD COLUMN description TEXT")
        conn.commit()
        conn.close()
