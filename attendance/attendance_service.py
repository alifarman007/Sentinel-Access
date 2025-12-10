"""
Attendance Service

Handles attendance recording with deduplication.
"""

import uuid
from datetime import datetime, date, timedelta
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc

from config.settings import settings
from config.logging_config import logger
from attendance.models import Person, AttendanceRecord, Camera, FaceEmbedding
from attendance.database import get_db_session


class AttendanceService:
    """
    Attendance management service.
    
    Features:
    - Record attendance with deduplication
    - Entry/Exit tracking
    - Query attendance records
    """
    
    def __init__(
        self,
        dedup_interval_minutes: int = None,
        enable_exit_tracking: bool = None
    ):
        """
        Initialize attendance service.
        
        Args:
            dedup_interval_minutes: Minutes before same person can be recorded again
            enable_exit_tracking: Whether to track exit times
        """
        self.dedup_interval = dedup_interval_minutes or settings.DEDUP_INTERVAL_MINUTES
        self.enable_exit_tracking = enable_exit_tracking if enable_exit_tracking is not None else settings.ENABLE_EXIT_TRACKING
        
        logger.info(
            f"AttendanceService initialized: dedup={self.dedup_interval}min, "
            f"exit_tracking={self.enable_exit_tracking}"
        )
    
    def _should_record(
        self,
        session: Session,
        person_db_id: uuid.UUID,
        record_type: str
    ) -> bool:
        """
        Check if attendance should be recorded (deduplication).
        
        Returns True if enough time has passed since last record.
        """
        cutoff_time = datetime.now() - timedelta(minutes=self.dedup_interval)
        
        # Check for recent record of same type
        recent = session.query(AttendanceRecord).filter(
            and_(
                AttendanceRecord.person_id == person_db_id,
                AttendanceRecord.record_type == record_type,
                AttendanceRecord.recorded_at >= cutoff_time
            )
        ).first()
        
        return recent is None
    
    def record_attendance(
        self,
        person_id: str,
        camera_id: str,
        confidence: float,
        record_type: str = "entry",
        snapshot_path: Optional[str] = None
    ) -> Tuple[bool, Optional[AttendanceRecord], str]:
        """
        Record attendance for a person.
        
        Args:
            person_id: Person's ID (e.g., employee ID)
            camera_id: Camera identifier
            confidence: Recognition confidence score
            record_type: "entry" or "exit"
            snapshot_path: Path to face snapshot image
        
        Returns:
            (success, record, message)
        """
        with get_db_session() as session:
            # Find person
            person = session.query(Person).filter(
                Person.person_id == person_id
            ).first()
            
            if not person:
                return False, None, f"Person not found: {person_id}"
            
            # Find camera (optional)
            camera = None
            if camera_id:
                camera = session.query(Camera).filter(
                    Camera.camera_id == camera_id
                ).first()
            
            # Check deduplication
            if not self._should_record(session, person.id, record_type):
                return False, None, f"Duplicate: {person.name} already recorded within {self.dedup_interval} minutes"
            
            # Create attendance record
            record = AttendanceRecord(
                person_id=person.id,
                camera_id=camera.id if camera else None,
                record_type=record_type,
                confidence=confidence,
                snapshot_path=snapshot_path
            )
            
            session.add(record)
            session.commit()
            session.refresh(record)
            
            logger.info(
                f"Attendance recorded: {person.name} ({person_id}) - "
                f"{record_type} at {record.recorded_at.strftime('%H:%M:%S')}"
            )
            
            return True, record, f"Recorded {record_type} for {person.name}"
    
    def get_today_attendance(self) -> List[dict]:
        """Get all attendance records for today."""
        today = date.today()
        return self.get_attendance_by_date(today, today)
    
    def get_attendance_by_date(
        self,
        start_date: date,
        end_date: date = None
    ) -> List[dict]:
        """
        Get attendance records for a date range.
        
        Args:
            start_date: Start date
            end_date: End date (defaults to start_date)
        
        Returns:
            List of attendance record dicts
        """
        if end_date is None:
            end_date = start_date
        
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        
        with get_db_session() as session:
            records = session.query(AttendanceRecord).join(Person).filter(
                and_(
                    AttendanceRecord.recorded_at >= start_dt,
                    AttendanceRecord.recorded_at <= end_dt
                )
            ).order_by(desc(AttendanceRecord.recorded_at)).all()
            
            result = []
            for r in records:
                result.append({
                    'id': str(r.id),
                    'person_id': r.person.person_id,
                    'name': r.person.name,
                    'record_type': r.record_type,
                    'confidence': r.confidence,
                    'recorded_at': r.recorded_at,
                    'time': r.recorded_at.strftime('%H:%M:%S'),
                    'date': r.recorded_at.strftime('%Y-%m-%d')
                })
            
            return result
    
    def get_person_attendance(
        self,
        person_id: str,
        start_date: date = None,
        end_date: date = None
    ) -> List[dict]:
        """Get attendance history for a specific person."""
        with get_db_session() as session:
            person = session.query(Person).filter(
                Person.person_id == person_id
            ).first()
            
            if not person:
                return []
            
            query = session.query(AttendanceRecord).filter(
                AttendanceRecord.person_id == person.id
            )
            
            if start_date:
                query = query.filter(
                    AttendanceRecord.recorded_at >= datetime.combine(start_date, datetime.min.time())
                )
            if end_date:
                query = query.filter(
                    AttendanceRecord.recorded_at <= datetime.combine(end_date, datetime.max.time())
                )
            
            records = query.order_by(desc(AttendanceRecord.recorded_at)).all()
            
            return [{
                'record_type': r.record_type,
                'confidence': r.confidence,
                'recorded_at': r.recorded_at,
                'time': r.recorded_at.strftime('%H:%M:%S'),
                'date': r.recorded_at.strftime('%Y-%m-%d')
            } for r in records]


class PersonService:
    """Service for managing registered persons."""
    
    @staticmethod
    def create_person(
        person_id: str,
        name: str,
        department: str = None,
        email: str = None,
        phone: str = None
    ) -> Tuple[bool, Optional[Person], str]:
        """
        Create a new person in the database.
        
        Returns:
            (success, person, message)
        """
        with get_db_session() as session:
            # Check if exists
            existing = session.query(Person).filter(
                Person.person_id == person_id
            ).first()
            
            if existing:
                return False, None, f"Person with ID '{person_id}' already exists"
            
            person = Person(
                person_id=person_id,
                name=name,
                department=department,
                email=email,
                phone=phone
            )
            
            session.add(person)
            session.commit()
            session.refresh(person)
            
            logger.info(f"Created person: {name} (ID: {person_id})")
            
            return True, person, f"Created person: {name}"
    
    @staticmethod
    def get_person(person_id: str) -> Optional[dict]:
        """Get person by ID."""
        with get_db_session() as session:
            person = session.query(Person).filter(
                Person.person_id == person_id
            ).first()
            
            if not person:
                return None
            
            return {
                'id': str(person.id),
                'person_id': person.person_id,
                'name': person.name,
                'department': person.department,
                'email': person.email,
                'phone': person.phone,
                'is_active': person.is_active,
                'created_at': person.created_at
            }
    
    @staticmethod
    def get_all_persons(active_only: bool = True) -> List[dict]:
        """Get all registered persons."""
        with get_db_session() as session:
            query = session.query(Person)
            
            if active_only:
                query = query.filter(Person.is_active == True)
            
            persons = query.order_by(Person.name).all()
            
            return [{
                'id': str(p.id),
                'person_id': p.person_id,
                'name': p.name,
                'department': p.department,
                'is_active': p.is_active
            } for p in persons]
    
    @staticmethod
    def link_face_embedding(
        person_id: str,
        faiss_index_id: int,
        image_path: str = None
    ) -> bool:
        """Link a FAISS embedding to a person."""
        with get_db_session() as session:
            person = session.query(Person).filter(
                Person.person_id == person_id
            ).first()
            
            if not person:
                logger.error(f"Person not found: {person_id}")
                return False
            
            embedding = FaceEmbedding(
                person_id=person.id,
                faiss_index_id=faiss_index_id,
                image_path=image_path
            )
            
            session.add(embedding)
            session.commit()
            
            logger.info(f"Linked embedding {faiss_index_id} to {person.name}")
            return True


class CameraService:
    """Service for managing cameras."""
    
    @staticmethod
    def add_camera(
        camera_id: str,
        name: str,
        rtsp_url: str,
        camera_type: str = "entry",
        location: str = None
    ) -> Tuple[bool, Optional[Camera], str]:
        """Add a new camera."""
        with get_db_session() as session:
            existing = session.query(Camera).filter(
                Camera.camera_id == camera_id
            ).first()
            
            if existing:
                return False, None, f"Camera '{camera_id}' already exists"
            
            camera = Camera(
                camera_id=camera_id,
                name=name,
                rtsp_url=rtsp_url,
                camera_type=camera_type,
                location=location
            )
            
            session.add(camera)
            session.commit()
            session.refresh(camera)
            
            logger.info(f"Added camera: {name} (ID: {camera_id}, Type: {camera_type})")
            
            return True, camera, f"Added camera: {name}"
    
    @staticmethod
    def get_all_cameras(active_only: bool = True) -> List[dict]:
        """Get all cameras."""
        with get_db_session() as session:
            query = session.query(Camera)
            
            if active_only:
                query = query.filter(Camera.is_active == True)
            
            cameras = query.all()
            
            return [{
                'id': str(c.id),
                'camera_id': c.camera_id,
                'name': c.name,
                'rtsp_url': c.rtsp_url,
                'camera_type': c.camera_type,
                'location': c.location,
                'is_active': c.is_active
            } for c in cameras]
    
    @staticmethod
    def get_camera(camera_id: str) -> Optional[dict]:
        """Get camera by ID."""
        with get_db_session() as session:
            camera = session.query(Camera).filter(
                Camera.camera_id == camera_id
            ).first()
            
            if not camera:
                return None
            
            return {
                'id': str(camera.id),
                'camera_id': camera.camera_id,
                'name': camera.name,
                'rtsp_url': camera.rtsp_url,
                'camera_type': camera.camera_type,
                'location': camera.location,
                'is_active': camera.is_active
            }