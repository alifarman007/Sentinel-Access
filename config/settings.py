"""
Application Settings

Centralized configuration using Pydantic Settings.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).parent.parent
    
    # Database settings
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "face_attendance"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    
    # Model settings
    DETECTION_MODEL: str = "det_10g.onnx"
    RECOGNITION_MODEL: str = "w600k_r50.onnx"
    
    # Recognition settings
    RECOGNITION_THRESHOLD: float = 0.4
    DETECTION_CONFIDENCE: float = 0.5
    USE_GPU: bool = True
    
    # Camera settings
    MAX_CAMERAS: int = 4
    TARGET_FPS: int = 15
    FRAME_BUFFER_SIZE: int = 3
    RTSP_LATENCY: int = 100
    
    # Attendance settings
    DEDUP_INTERVAL_MINUTES: int = 60
    ENABLE_EXIT_TRACKING: bool = False
    
    # FAISS settings
    FAISS_INDEX_TYPE: str = "Flat"
    EMBEDDING_DIMENSION: int = 512
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def models_dir(self) -> Path:
        """Get models directory path."""
        return self.BASE_DIR / "models"
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self.BASE_DIR / "data"
    
    @property
    def faces_dir(self) -> Path:
        """Get faces storage directory."""
        path = self.data_dir / "faces"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def embeddings_dir(self) -> Path:
        """Get embeddings storage directory."""
        path = self.data_dir / "embeddings"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        path = self.data_dir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def detection_model_path(self) -> Path:
        """Get detection model path."""
        return self.models_dir / self.DETECTION_MODEL
    
    @property
    def recognition_model_path(self) -> Path:
        """Get recognition model path."""
        return self.models_dir / self.RECOGNITION_MODEL


# Global settings instance
settings = Settings()