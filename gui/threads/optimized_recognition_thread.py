"""
Optimized Recognition Thread

Memory-efficient face recognition processing.
"""

import numpy as np
import time
from typing import Optional, List
from dataclasses import dataclass
from collections import deque
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

from config.logging_config import logger
from core.recognition_pipeline import RecognitionPipeline, RecognitionResult


@dataclass
class RecognitionOutput:
    """Recognition result for GUI."""
    camera_id: str
    results: List[RecognitionResult]
    process_time: float


class OptimizedRecognitionThread(QThread):
    """
    Optimized recognition thread with frame dropping.
    
    Features:
    - Single frame buffer per camera (latest only)
    - Automatic frame dropping when busy
    - Memory-efficient processing
    """
    
    # Signals
    recognition_complete = Signal(str, object)  # camera_id, RecognitionOutput
    person_recognized = Signal(str, str, str, float)  # camera_id, person_id, name, confidence
    stats_updated = Signal(float, int)  # avg_time_ms, pending_count
    
    def __init__(
        self,
        pipeline: Optional[RecognitionPipeline] = None,
        parent=None
    ):
        super().__init__(parent)
        
        self.pipeline = pipeline
        
        self._running = False
        self._mutex = QMutex()
        
        # Single frame buffer per camera (always latest)
        self._pending_frames = {}  # camera_id -> (frame, timestamp)
        
        # Stats
        self._process_times = deque(maxlen=30)
    
    def set_pipeline(self, pipeline: RecognitionPipeline):
        """Set recognition pipeline."""
        with QMutexLocker(self._mutex):
            self.pipeline = pipeline
    
    def submit_frame(self, camera_id: str, frame: np.ndarray):
        """
        Submit frame for recognition.
        Always keeps only the latest frame per camera.
        """
        with QMutexLocker(self._mutex):
            # Replace any existing frame for this camera
            self._pending_frames[camera_id] = (frame, time.time())
    
    def run(self):
        """Main processing loop."""
        self._running = True
        
        if self.pipeline is None:
            logger.info("Initializing recognition pipeline...")
            self.pipeline = RecognitionPipeline(use_gpu=True)
        
        logger.info("Optimized recognition thread started")
        
        while self._running:
            # Get a frame to process
            camera_id, frame, timestamp = self._get_next_frame()
            
            if frame is None:
                self.msleep(10)
                continue
            
            # Skip old frames (> 500ms)
            if time.time() - timestamp > 0.5:
                continue
            
            # Process
            start_time = time.time()
            
            try:
                results = self.pipeline.process_frame(frame, identify=True)
                process_time = time.time() - start_time
                
                # Track stats
                self._process_times.append(process_time)
                avg_time = sum(self._process_times) / len(self._process_times)
                
                with QMutexLocker(self._mutex):
                    pending_count = len(self._pending_frames)
                
                self.stats_updated.emit(avg_time * 1000, pending_count)
                
                # Emit results
                output = RecognitionOutput(
                    camera_id=camera_id,
                    results=results,
                    process_time=process_time
                )
                self.recognition_complete.emit(camera_id, output)
                
                # Emit individual recognitions
                for result in results:
                    if result.is_known:
                        self.person_recognized.emit(
                            camera_id,
                            result.person_id,
                            result.name,
                            result.confidence
                        )
                
            except Exception as e:
                logger.error(f"Recognition error: {e}")
            
            # Free frame memory
            del frame
        
        logger.info("Optimized recognition thread stopped")
    
    def _get_next_frame(self):
        """Get next frame to process (oldest camera first for fairness)."""
        with QMutexLocker(self._mutex):
            if not self._pending_frames:
                return None, None, 0
            
            # Get oldest frame
            oldest_camera = None
            oldest_time = float('inf')
            
            for cam_id, (frame, ts) in self._pending_frames.items():
                if ts < oldest_time:
                    oldest_time = ts
                    oldest_camera = cam_id
            
            if oldest_camera:
                frame, timestamp = self._pending_frames.pop(oldest_camera)
                return oldest_camera, frame, timestamp
            
            return None, None, 0
    
    def stop(self):
        """Stop processing thread."""
        self._running = False
        self.wait(3000)
    
    def clear(self):
        """Clear pending frames."""
        with QMutexLocker(self._mutex):
            self._pending_frames.clear()