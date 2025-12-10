"""
Logging Configuration using Loguru
"""

import sys
from pathlib import Path
from loguru import logger

from config.settings import settings


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_filename: str = "app.log"
) -> None:
    """
    Configure application logging.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to write logs to file
        log_filename: Name of the log file
    """
    # Remove default handler
    logger.remove()
    
    # Console handler with color
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )
    
    # File handler
    if log_to_file:
        log_path = settings.logs_dir / log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
    
    logger.info(f"Logging configured - Level: {log_level}, File: {log_to_file}")


# Export logger for use in other modules
__all__ = ["logger", "setup_logging"]