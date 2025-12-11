"""
Simple Reliable Camera Thread

Direct OpenCV approach - proven to work.
"""

import cv2
import numpy as np
import time
from typing import Optional
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

from config.logging_config import logger


class SimpleCameraThread(QThread):
    """
    Simple and reliable camera thread.
    
    Uses direct OpenCV - no fancy pipelines that might fail.
    """
    
    # Output resolution
    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 360
    
    # Signals
    frame_ready = Signal(str, np.ndarray)
    frame_for_recognition = Signal(str, np.ndarray)
    status_changed = Signal(str, str)
    fps_updated = Signal(str, float)
    error_occurred = Signal(str, str)
    
    def __init__(
        self,
        camera_id: str,
        source: str,
        target_fps: int = 12,
        recognition_fps: int = 4,
        parent=None
    ):
        super().__init__(parent)
        
        self.camera_id = camera_id
        self.source = source
        self.target_fps = target_fps
        self.recognition_fps = recognition_fps
        
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._mutex = QMutex()
        
        self._display_interval = 1.0 / target_fps
        self._recognition_interval = 1.0 / recognition_fps
        self._last_display_time = 0
        self._last_recognition_time = 0
    
    def _connect(self) -> bool:
        """Connect to camera - simple direct method."""
        self.status_changed.emit(self.camera_id, "connecting")
        
        logger.info(f"[{self.camera_id}] Connecting to: {self.source[:50]}...")
        
        try:
            # For webcam
            if self.source.isdigit():
                self._cap = cv2.VideoCapture(int(self.source))
                if self._cap.isOpened():
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                # For RTSP - use direct OpenCV
                # This is the most compatible method
                self._cap = cv2.VideoCapture(self.source)
            
            if self._cap is None or not self._cap.isOpened():
                logger.error(f"[{self.camera_id}] Failed to open capture")
                self.status_changed.emit(self.camera_id, "error")
                self.error_occurred.emit(self.camera_id, "Cannot open stream")
                return False
            
            # Set minimal buffer
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test read
            ret, frame = self._cap.read()
            
            if not ret or frame is None:
                logger.error(f"[{self.camera_id}] Cannot read frame")
                self._cap.release()
                self._cap = None
                self.status_changed.emit(self.camera_id, "error")
                self.error_occurred.emit(self.camera_id, "Cannot read frames")
                return False
            
            logger.info(f"[{self.camera_id}] Connected! Frame: {frame.shape[1]}x{frame.shape[0]}")
            self.status_changed.emit(self.camera_id, "connected")
            return True
            
        except Exception as e:
            logger.error(f"[{self.camera_id}] Connection error: {e}")
            self.status_changed.emit(self.camera_id, "error")
            self.error_occurred.emit(self.camera_id, str(e))
            return False
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize to output resolution."""
        h, w = frame.shape[:2]
        
        if w <= self.OUTPUT_WIDTH and h <= self.OUTPUT_HEIGHT:
            return frame
        
        # Calculate scale preserving aspect ratio
        scale = min(self.OUTPUT_WIDTH / w, self.OUTPUT_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _grab_and_skip(self, skip_count: int = 5):
        """Grab and skip frames to clear buffer."""
        for _ in range(skip_count):
            self._cap.grab()
    
    def run(self):
        """Main capture loop."""
        self._running = True
        
        if not self._connect():
            return
        
        frame_count = 0
        fps_start = time.time()
        reconnect_delay = 2.0
        error_count = 0
        
        # Skip initial buffered frames
        self._grab_and_skip(10)
        
        while self._running:
            current_time = time.time()
            
            # Check if we need to display this frame
            time_since_display = current_time - self._last_display_time
            if time_since_display < self._display_interval:
                # Skip frame but keep grabbing to prevent buffer buildup
                with QMutexLocker(self._mutex):
                    if self._cap:
                        self._cap.grab()
                self.msleep(5)
                continue
            
            # Read frame
            with QMutexLocker(self._mutex):
                if self._cap is None:
                    break
                ret, frame = self._cap.read()
            
            if not ret or frame is None:
                error_count += 1
                logger.warning(f"[{self.camera_id}] Read failed ({error_count})")
                
                if error_count >= 10:
                    logger.error(f"[{self.camera_id}] Too many errors, reconnecting...")
                    self.status_changed.emit(self.camera_id, "disconnected")
                    
                    with QMutexLocker(self._mutex):
                        if self._cap:
                            self._cap.release()
                            self._cap = None
                    
                    self.msleep(int(reconnect_delay * 1000))
                    
                    if self._running:
                        if self._connect():
                            error_count = 0
                            reconnect_delay = 2.0
                            self._grab_and_skip(10)
                        else:
                            reconnect_delay = min(reconnect_delay * 1.5, 30)
                
                continue
            
            error_count = 0
            self._last_display_time = current_time
            
            # Resize frame
            small_frame = self._resize_frame(frame)
            
            # Emit for display
            self.frame_ready.emit(self.camera_id, small_frame)
            
            # FPS calculation
            frame_count += 1
            elapsed = current_time - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                self.fps_updated.emit(self.camera_id, fps)
                frame_count = 0
                fps_start = current_time
            
            # Emit for recognition (less frequently)
            if current_time - self._last_recognition_time >= self._recognition_interval:
                self._last_recognition_time = current_time
                self.frame_for_recognition.emit(self.camera_id, small_frame.copy())
        
        # Cleanup
        with QMutexLocker(self._mutex):
            if self._cap:
                self._cap.release()
                self._cap = None
        
        self.status_changed.emit(self.camera_id, "stopped")
        logger.info(f"[{self.camera_id}] Thread stopped")
    
    def stop(self):
        """Stop the thread."""
        self._running = False
        self.wait(5000)