"""
Application Settings - Centralized Configuration
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODELS_DIR: Path = Field(default=Path("models"))
    DATA_DIR: Path = Field(default=Path("data"))
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/face_attendance"
    )
    
    # Model paths (relative to MODELS_DIR)
    DETECTION_MODEL: str = Field(default="det_10g.onnx")
    RECOGNITION_MODEL: str = Field(default="w600k_r50.onnx")
    
    # Recognition settings
    DETECTION_CONFIDENCE: float = Field(default=0.5, ge=0.1, le=1.0)
    RECOGNITION_THRESHOLD: float = Field(default=0.4, ge=0.1, le=1.0)
    USE_GPU: bool = Field(default=True)
    
    # Face detection settings
    DETECTION_INPUT_SIZE: tuple = (640, 640)
    NMS_THRESHOLD: float = Field(default=0.4)
    
    # Attendance settings
    DEDUP_INTERVAL_MINUTES: int = Field(default=60, ge=1)
    ENABLE_EXIT_TRACKING: bool = Field(default=False)
    
    # Camera settings
    MAX_CAMERAS: int = Field(default=8)
    DEFAULT_CAMERA_FPS: int = Field(default=15)
    FRAME_BUFFER_SIZE: int = Field(default=4)
    RTSP_LATENCY_MS: int = Field(default=100)
    
    # Display settings
    DISPLAY_WIDTH: int = Field(default=640)
    DISPLAY_HEIGHT: int = Field(default=480)
    
    # FAISS settings
    FAISS_INDEX_TYPE: str = Field(default="Flat")  # Flat, IVF256
    EMBEDDING_DIMENSION: int = Field(default=512)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    @property
    def detection_model_path(self) -> Path:
        """Full path to detection model."""
        return self.BASE_DIR / self.MODELS_DIR / self.DETECTION_MODEL
    
    @property
    def recognition_model_path(self) -> Path:
        """Full path to recognition model."""
        return self.BASE_DIR / self.MODELS_DIR / self.RECOGNITION_MODEL
    
    @property
    def faces_dir(self) -> Path:
        """Directory for registered face images."""
        return self.BASE_DIR / self.DATA_DIR / "faces"
    
    @property
    def embeddings_dir(self) -> Path:
        """Directory for FAISS index files."""
        return self.BASE_DIR / self.DATA_DIR / "embeddings"
    
    @property
    def logs_dir(self) -> Path:
        """Directory for log files."""
        return self.BASE_DIR / self.DATA_DIR / "logs"


# Global settings instance
settings = Settings()