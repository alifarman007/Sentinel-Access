"""
SQLAlchemy Database Models
"""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String, Boolean, Float, Integer, DateTime, 
    ForeignKey, Text, Index, func
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship
)
from sqlalchemy.dialects.postgresql import UUID


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Person(Base):
    """Registered person/employee."""
    
    __tablename__ = "persons"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    person_id: Mapped[str] = mapped_column(
        String(50), 
        unique=True, 
        nullable=False,
        index=True,
        comment="Employee/Student ID"
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    department: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now(), 
        onupdate=func.now()
    )
    
    # Relationships
    face_embeddings: Mapped[list["FaceEmbedding"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan"
    )
    attendance_records: Mapped[list["AttendanceRecord"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Person(id={self.person_id}, name={self.name})>"


class FaceEmbedding(Base):
    """Face embedding reference (actual vectors stored in FAISS)."""
    
    __tablename__ = "face_embeddings"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    person_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="CASCADE"),
        nullable=False
    )
    faiss_index_id: Mapped[int] = mapped_column(
        Integer, 
        nullable=False,
        comment="FAISS internal vector ID"
    )
    image_path: Mapped[Optional[str]] = mapped_column(
        String(500), 
        nullable=True,
        comment="Path to original face image"
    )
    quality_score: Mapped[Optional[float]] = mapped_column(
        Float, 
        nullable=True,
        comment="Face quality score at registration"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now()
    )
    
    # Relationships
    person: Mapped["Person"] = relationship(back_populates="face_embeddings")
    
    def __repr__(self) -> str:
        return f"<FaceEmbedding(person_id={self.person_id}, faiss_id={self.faiss_index_id})>"


class Camera(Base):
    """Camera configuration."""
    
    __tablename__ = "cameras"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    camera_id: Mapped[str] = mapped_column(
        String(50), 
        unique=True, 
        nullable=False,
        index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    rtsp_url: Mapped[str] = mapped_column(String(500), nullable=False)
    camera_type: Mapped[str] = mapped_column(
        String(20), 
        default="entry",
        comment="'entry' or 'exit'"
    )
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now()
    )
    
    # Relationships
    attendance_records: Mapped[list["AttendanceRecord"]] = relationship(
        back_populates="camera"
    )
    
    def __repr__(self) -> str:
        return f"<Camera(id={self.camera_id}, name={self.name}, type={self.camera_type})>"


class AttendanceRecord(Base):
    """Attendance record entry."""
    
    __tablename__ = "attendance_records"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    person_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="CASCADE"),
        nullable=False
    )
    camera_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cameras.id", ondelete="SET NULL"),
        nullable=True
    )
    
    record_type: Mapped[str] = mapped_column(
        String(20), 
        nullable=False,
        comment="'entry' or 'exit'"
    )
    confidence: Mapped[float] = mapped_column(
        Float, 
        nullable=False,
        comment="Recognition confidence score"
    )
    snapshot_path: Mapped[Optional[str]] = mapped_column(
        String(500), 
        nullable=True,
        comment="Face snapshot at detection"
    )
    
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now(),
        index=True
    )
    
    # Relationships
    person: Mapped["Person"] = relationship(back_populates="attendance_records")
    camera: Mapped[Optional["Camera"]] = relationship(back_populates="attendance_records")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_attendance_person_date', person_id, func.date(recorded_at)),
        Index('idx_attendance_type_date', record_type, recorded_at),
    )
    
    def __repr__(self) -> str:
        return f"<AttendanceRecord(person={self.person_id}, type={self.record_type}, at={self.recorded_at})>"