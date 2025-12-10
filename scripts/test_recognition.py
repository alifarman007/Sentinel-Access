"""
Test Full Recognition Pipeline
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging, logger
from core.recognition_pipeline import RecognitionPipeline
from core.utils import draw_face_box


def test_recognition_pipeline():
    """Test the complete recognition pipeline."""
    setup_logging(log_level="INFO", log_to_file=False)
    
    logger.info("=" * 50)
    logger.info("Face Recognition Pipeline Test")
    logger.info("=" * 50)
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = RecognitionPipeline(use_gpu=True)
    
    logger.info(f"Database has {pipeline.database.get_count()} persons")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("CONTROLS:")
    logger.info("  'r' - Register current face")
    logger.info("  's' - Save database")
    logger.info("  'l' - List registered persons")
    logger.info("  'c' - Clear database")
    logger.info("  'q' - Quit")
    logger.info("=" * 50)
    logger.info("")
    
    import time
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        results = pipeline.process_frame(frame, identify=True)
        
        # Draw results
        for result in results:
            # Choose color: green for known, red for unknown
            color = (0, 255, 0) if result.is_known else (0, 0, 255)
            
            draw_face_box(
                frame,
                result.detection.bbox.tolist(),
                name=result.name,
                confidence=result.confidence,
                color=color
            )
            
            # Draw landmarks
            for point in result.detection.landmarks:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1)
        
        # Show FPS and stats
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(results)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Registered: {pipeline.database.get_count()}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Face Recognition Test", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('r'):
            # Register current face
            if len(results) == 1:
                person_id = input("Enter person ID: ").strip()
                name = input("Enter name: ").strip()
                
                if person_id and name:
                    success, msg = pipeline.register_from_embedding(
                        results[0].embedding,
                        person_id,
                        name
                    )
                    logger.info(msg)
            elif len(results) == 0:
                logger.warning("No face detected for registration")
            else:
                logger.warning("Multiple faces detected. Show only one face.")
        
        elif key == ord('s'):
            pipeline.database.save()
            logger.info("Database saved")
        
        elif key == ord('l'):
            persons = pipeline.database.get_all_persons()
            logger.info(f"Registered persons ({len(persons)}):")
            for p in persons:
                logger.info(f"  - {p['name']} (ID: {p['person_id']})")
        
        elif key == ord('c'):
            confirm = input("Clear database? (yes/no): ")
            if confirm.lower() == 'yes':
                pipeline.database.clear()
                logger.info("Database cleared")
    
    cap.release()
    cv2.destroyAllWindows()
    
    logger.info("=" * 50)
    logger.info("Test complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    test_recognition_pipeline()