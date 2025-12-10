"""
Database Initialization Script
Run this once to create all tables.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging, logger
from attendance.database import init_database, check_database_connection


def main():
    """Initialize the database."""
    setup_logging(log_level="INFO", log_to_file=False)
    
    logger.info("=" * 50)
    logger.info("Face Re-ID Attendance System - Database Setup")
    logger.info("=" * 50)
    
    # Test connection
    logger.info("Testing database connection...")
    if not check_database_connection():
        logger.error("Cannot connect to database. Check your .env file.")
        logger.error("Make sure PostgreSQL is running and credentials are correct.")
        sys.exit(1)
    
    # Create tables
    logger.info("Creating database tables...")
    init_database()
    
    logger.info("=" * 50)
    logger.info("Database initialization complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()