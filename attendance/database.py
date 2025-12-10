"""
Database Connection and Session Management
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from config.settings import settings
from config.logging_config import logger
from attendance.models import Base


# Create engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # Verify connections before use
    echo=False  # Set True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)


def init_database() -> None:
    """
    Initialize database - create all tables.
    Call this once at application startup.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


def drop_database() -> None:
    """
    Drop all tables. USE WITH CAUTION!
    """
    try:
        Base.metadata.drop_all(bind=engine)
        logger.warning("All database tables dropped")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise


def get_session() -> Generator[Session, None, None]:
    """
    Dependency for getting database session.
    Use with FastAPI or as a generator.
    
    Usage:
        for session in get_session():
            # use session
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database session.
    
    Usage:
        with get_db_session() as session:
            # use session
            session.commit()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def check_database_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


# Optional: Log slow queries (for debugging)
@event.listens_for(engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault("query_start_time", []).append(time.time())


@event.listens_for(engine, "after_cursor_execute")
def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    import time
    total = time.time() - conn.info["query_start_time"].pop(-1)
    if total > 0.5:  # Log queries taking more than 500ms
        logger.warning(f"Slow query ({total:.2f}s): {statement[:100]}...")


# Need to import time for slow query logging
import time