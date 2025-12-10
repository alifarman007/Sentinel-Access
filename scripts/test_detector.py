"""
Test Face Detector
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging, logger
from core.face_detector import FaceDetector
from core.utils import draw_face_box


def test_detector_initialization():
    """Test detector loads correctly."""
    logger.info("Testing detector initialization...")
    
    detector = FaceDetector(use_gpu=True)
    logger.info(f"Detector using GPU: {detector._using_gpu}")
    logger.info("✓ Detector initialized successfully")
    
    return detector


def test_detection_on_webcam(detector: FaceDetector, duration_sec: int = 10):
    """Test detection on webcam feed."""
    logger.info(f"Testing webcam detection for {duration_sec} seconds...")
    logger.info("Press 'q' to quit early")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.warning("No webcam available, skipping webcam test")
        return
    
    import time
    start_time = time.time()
    frame_count = 0
    
    while (time.time() - start_time) < duration_sec:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        detections = detector.detect(frame)
        frame_count += 1
        
        # Draw detections
        for det in detections:
            draw_face_box(
                frame,
                det.bbox.tolist(),
                name=f"Face",
                confidence=det.confidence
            )
            
            # Draw landmarks
            for point in det.landmarks:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
        
        # Show FPS
        fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(detections)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Face Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    
    cap.release()
    cv2.destroyAllWindows()
    
    logger.info(f"✓ Webcam test complete: {frame_count} frames, {avg_fps:.1f} FPS average")


def main():
    setup_logging(log_level="INFO", log_to_file=False)
    
    logger.info("=" * 50)
    logger.info("Face Detector Test")
    logger.info("=" * 50)
    
    # Test 1: Initialize detector
    detector = test_detector_initialization()
    
    # Test 2: Webcam detection
    test_detection_on_webcam(detector, duration_sec=15)
    
    logger.info("=" * 50)
    logger.info("All tests complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()