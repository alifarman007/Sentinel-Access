"""
Recognition Thread for GUI

Processes frames for face recognition in background.
"""

import numpy as np
import queue
import time
from typing import Optional, List
from dataclasses import dataclass
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

from config.logging_config import logger
from core.recognition_pipeline import RecognitionPipeline, RecognitionResult


@dataclass
class FrameTask:
    """Frame to be processed."""
    camera_id: str
    frame: np.ndarray
    timestamp: float


@dataclass 
class RecognitionOutput:
    """Recognition result for GUI."""
    camera_id: str
    frame: np.ndarray
    results: List[RecognitionResult]
    process_time: float


class RecognitionThread(QThread):
    """
    Background thread for face recognition.
    
    Processes frames from a queue and emits results.
    """
    
    # Signals
    recognition_complete = Signal(object)  # RecognitionOutput
    person_recognized = Signal(str, str, str, float)  # camera_id, person_id, name, confidence
    unknown_face_detected = Signal(str)  # camera_id
    stats_updated = Signal(float, int)  # process_time_ms, queue_size
    
    def __init__(
        self,
        pipeline: Optional[RecognitionPipeline] = None,
        max_queue_size: int = 10,
        parent=None
    ):
        super().__init__(parent)
        
        self.pipeline = pipeline
        self.max_queue_size = max_queue_size
        
        self._running = False
        self._frame_queue = queue.Queue(maxsize=max_queue_size)
        self._mutex = QMutex()
        
        # Stats
        self._process_times = []
    
    def set_pipeline(self, pipeline: RecognitionPipeline):
        """Set or update the recognition pipeline."""
        with QMutexLocker(self._mutex):
            self.pipeline = pipeline
    
    def submit_frame(self, camera_id: str, frame: np.ndarray) -> bool:
        """
        Submit a frame for processing.
        
        Returns:
            True if frame was queued, False if queue is full
        """
        try:
            task = FrameTask(
                camera_id=camera_id,
                frame=frame.copy(),
                timestamp=time.time()
            )
            self._frame_queue.put_nowait(task)
            return True
        except queue.Full:
            return False
    
    def run(self):
        """Main processing loop."""
        self._running = True
        
        # Initialize pipeline if not set
        if self.pipeline is None:
            logger.info("Initializing recognition pipeline in thread...")
            self.pipeline = RecognitionPipeline(use_gpu=True)
        
        logger.info("Recognition thread started")
        
        while self._running:
            try:
                # Get frame from queue (with timeout)
                task = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Check frame age (skip old frames)
            frame_age = time.time() - task.timestamp
            if frame_age > 0.5:  # Skip frames older than 500ms
                continue
            
            # Process frame
            start_time = time.time()
            
            with QMutexLocker(self._mutex):
                if self.pipeline is None:
                    continue
                results = self.pipeline.process_frame(task.frame, identify=True)
            
            process_time = time.time() - start_time
            
            # Track stats
            self._process_times.append(process_time)
            if len(self._process_times) > 30:
                self._process_times.pop(0)
            
            avg_time = sum(self._process_times) / len(self._process_times)
            self.stats_updated.emit(avg_time * 1000, self._frame_queue.qsize())
            
            # Emit recognition results
            output = RecognitionOutput(
                camera_id=task.camera_id,
                frame=task.frame,
                results=results,
                process_time=process_time
            )
            self.recognition_complete.emit(output)
            
            # Emit individual recognitions
            for result in results:
                if result.is_known:
                    self.person_recognized.emit(
                        task.camera_id,
                        result.person_id,
                        result.name,
                        result.confidence
                    )
                else:
                    self.unknown_face_detected.emit(task.camera_id)
        
        logger.info("Recognition thread stopped")
    
    def stop(self):
        """Stop the processing thread."""
        self._running = False
        self.wait(3000)
    
    def clear_queue(self):
        """Clear pending frames."""
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
    
    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._frame_queue.qsize()