"""
Test Attendance System

Tests the complete flow: detection → recognition → attendance recording
"""

import sys
import cv2
import time
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging, logger
from core.recognition_pipeline import RecognitionPipeline
from core.utils import draw_face_box
from attendance.attendance_service import (
    AttendanceService, PersonService, CameraService
)
from attendance.database import check_database_connection, init_database


def test_attendance_system():
    """Test the complete attendance system."""
    setup_logging(log_level="INFO", log_to_file=False)
    
    logger.info("=" * 60)
    logger.info("Attendance System Integration Test")
    logger.info("=" * 60)
    
    # Check database connection
    logger.info("Checking database connection...")
    if not check_database_connection():
        logger.error("Database connection failed! Check .env file.")
        return
    
    # Initialize database tables
    logger.info("Initializing database tables...")
    init_database()
    
    # Initialize services
    logger.info("Initializing services...")
    pipeline = RecognitionPipeline(use_gpu=True)
    attendance_service = AttendanceService(dedup_interval_minutes=1)  # 1 min for testing
    
    # Add a test camera to database (optional)
    CameraService.add_camera(
        camera_id="webcam",
        name="Test Webcam",
        rtsp_url="0",  # Webcam
        camera_type="entry"
    )
    
    logger.info(f"FAISS Database: {pipeline.database.get_count()} persons")
    logger.info(f"PostgreSQL Persons: {len(PersonService.get_all_persons())}")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONTROLS:")
    logger.info("  'r' - Register person (to both FAISS and PostgreSQL)")
    logger.info("  'a' - Show today's attendance")
    logger.info("  'p' - List all registered persons")
    logger.info("  'q' - Quit")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Recognized faces will automatically be logged to attendance!")
    logger.info("")
    
    frame_count = 0
    start_time = time.time()
    last_recognition_time = {}  # Track last recognition per person
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        results = pipeline.process_frame(frame, identify=True)
        
        # Draw results and record attendance
        for result in results:
            color = (0, 255, 0) if result.is_known else (0, 0, 255)
            
            # Record attendance for known faces
            if result.is_known:
                person_id = result.person_id
                now = time.time()
                
                # Simple client-side rate limiting (1 per second per person)
                if person_id not in last_recognition_time or \
                   (now - last_recognition_time[person_id]) > 1.0:
                    
                    success, record, msg = attendance_service.record_attendance(
                        person_id=person_id,
                        camera_id="webcam",
                        confidence=result.confidence,
                        record_type="entry"
                    )
                    
                    if success:
                        logger.info(f"✓ ATTENDANCE: {msg}")
                    
                    last_recognition_time[person_id] = now
            
            draw_face_box(
                frame,
                result.detection.bbox.tolist(),
                name=result.name,
                confidence=result.confidence,
                color=color
            )
        
        # Show stats
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(results)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Attendance System Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('r'):
            # Register person to both FAISS and PostgreSQL
            if len(results) == 1:
                person_id = input("Enter person ID (e.g., EMP001): ").strip()
                name = input("Enter full name: ").strip()
                department = input("Enter department (optional): ").strip() or None
                
                if person_id and name:
                    # Add to PostgreSQL
                    success, person, msg = PersonService.create_person(
                        person_id=person_id,
                        name=name,
                        department=department
                    )
                    
                    if success:
                        # Add to FAISS
                        faiss_id = pipeline.database.add_person(
                            person_id=person_id,
                            name=name,
                            embedding=results[0].embedding
                        )
                        
                        # Link embedding to person
                        PersonService.link_face_embedding(person_id, faiss_id)
                        
                        pipeline.database.save()
                        logger.info(f"✓ Registered {name} to both databases")
                    else:
                        logger.warning(msg)
            else:
                logger.warning("Need exactly 1 face for registration")
        
        elif key == ord('a'):
            # Show today's attendance
            records = attendance_service.get_today_attendance()
            logger.info(f"\n=== Today's Attendance ({len(records)} records) ===")
            for r in records[:10]:  # Show first 10
                logger.info(f"  {r['time']} | {r['name']} | {r['record_type']} | {r['confidence']:.2f}")
            if len(records) > 10:
                logger.info(f"  ... and {len(records) - 10} more")
        
        elif key == ord('p'):
            # List persons
            persons = PersonService.get_all_persons()
            logger.info(f"\n=== Registered Persons ({len(persons)}) ===")
            for p in persons:
                logger.info(f"  {p['person_id']} | {p['name']} | {p['department'] or 'N/A'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    logger.info("=" * 60)
    logger.info("Test complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_attendance_system()