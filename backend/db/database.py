"""
Database configuration and session management.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

logger = logging.getLogger(__name__)

# Database URL - using SQLite for simplicity
# In production, use PostgreSQL: postgresql://user:pass@host:port/dbname
SQLALCHEMY_DATABASE_URL = "sqlite:///./health_agent.db"

# Create engine with appropriate settings
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False,  # Set to True for SQL query logging
    pool_pre_ping=True  # Verify connections before using
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)


def get_db():
    """
    Dependency for FastAPI endpoints.
    Yields a database session and ensures cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """
    Initialize database: create tables and seed initial data.
    Call this on application startup.
    """
    from db.models import Base, create_all_tables, seed_medical_protocols
    
    # Create all tables
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    # Seed medical protocols
    logger.info("Seeding medical protocols...")
    db = SessionLocal()
    try:
        seeded_count = seed_medical_protocols(db)
        if seeded_count > 0:
            logger.info(f"Seeded {seeded_count} medical protocols")
        else:
            logger.info("Medical protocols already exist, skipping seed")
    finally:
        db.close()
    
    logger.info("Database initialization complete")


def check_database_health() -> bool:
    """
    Check if database is accessible.
    Returns True if healthy, False otherwise.
    """
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
