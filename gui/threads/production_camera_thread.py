"""
Production Camera Thread

Reliable camera thread for RTSP and webcam sources.
"""

import cv2
import numpy as np
import time
import os
from typing import Optional
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

from config.logging_config import logger


class ProductionCameraThread(QThread):
    """
    Production-grade camera thread.
    
    Features:
    - TCP transport for RTSP (prevents packet loss)
    - Automatic quality adjustment
    - Corruption detection
    - Smooth frame delivery
    """
    
    # Output resolution (480p for smooth performance)
    OUTPUT_WIDTH = 854
    OUTPUT_HEIGHT = 480
    
    # Signals
    frame_ready = Signal(str, np.ndarray)           # For display
    frame_for_recognition = Signal(str, np.ndarray) # For recognition
    status_changed = Signal(str, str)
    fps_updated = Signal(str, float)
    error_occurred = Signal(str, str)
    
    def __init__(
        self,
        camera_id: str,
        source: str,
        target_fps: int = 15,
        recognition_fps: int = 5,
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
    
    def _is_webcam(self) -> bool:
        """Check if source is webcam."""
        return self.source.isdigit()
    
    def _is_rtsp(self) -> bool:
        """Check if source is RTSP."""
        return self.source.lower().startswith("rtsp://")
    
    def _connect_webcam(self) -> bool:
        """Connect to webcam."""
        try:
            self._cap = cv2.VideoCapture(int(self.source))
            
            if self._cap.isOpened():
                # Set to 720p or lower
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    logger.info(f"[{self.camera_id}] Webcam: {frame.shape[1]}x{frame.shape[0]}")
                    return True
                
            if self._cap:
                self._cap.release()
                
        except Exception as e:
            logger.error(f"[{self.camera_id}] Webcam error: {e}")
        
        return False
    
    def _connect_rtsp_opencv(self) -> bool:
        """Connect to RTSP using OpenCV with TCP transport."""
        try:
            # Force TCP transport via FFmpeg options
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;524288|max_delay;500000"
            
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Read and validate frame
                ret, frame = self._cap.read()
                
                if ret and frame is not None and not self._is_frame_corrupted(frame):
                    logger.info(
                        f"[{self.camera_id}] RTSP OpenCV: {frame.shape[1]}x{frame.shape[0]}"
                    )
                    return True
                
            if self._cap:
                self._cap.release()
                
        except Exception as e:
            logger.error(f"[{self.camera_id}] OpenCV RTSP error: {e}")
        
        return False
    
    def _connect_rtsp_gstreamer(self) -> bool:
        """Connect to RTSP using GStreamer with TCP."""
        try:
            # Simple, reliable GStreamer pipeline with TCP
            pipeline = (
                f"rtspsrc location={self.source} protocols=tcp latency=200 ! "
                "queue max-size-buffers=1 leaky=downstream ! "
                "rtph264depay ! h264parse ! "
                "avdec_h264 ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true sync=false max-buffers=1"
            )
            
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if self._cap.isOpened():
                ret, frame = self._cap.read()
                
                if ret and frame is not None and not self._is_frame_corrupted(frame):
                    logger.info(
                        f"[{self.camera_id}] RTSP GStreamer: {frame.shape[1]}x{frame.shape[0]}"
                    )
                    return True
                
            if self._cap:
                self._cap.release()
                
        except Exception as e:
            logger.error(f"[{self.camera_id}] GStreamer RTSP error: {e}")
        
        return False
    
    def _is_frame_corrupted(self, frame: np.ndarray) -> bool:
        """Detect corrupted frames."""
        if frame is None or frame.size == 0:
            return True
        
        h, w = frame.shape[:2]
        if w < 100 or h < 100:
            return True
        
        # Check for vertical line corruption pattern
        # Sample vertical slices and check for repetition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check variance - corrupted frames often have very low or very uniform variance
        variance = np.var(gray)
        if variance < 50:  # Nearly uniform = likely corrupted
            return True
        
        # Check for vertical stripes (common corruption)
        # Sample every 10th column and compare
        cols = gray[:, ::10]
        col_vars = np.var(cols, axis=0)
        
        # If most columns have very similar variance = likely OK
        # If variance of variances is very high = likely corrupted
        if len(col_vars) > 5:
            var_of_vars = np.var(col_vars)
            mean_var = np.mean(col_vars)
            
            # Corrupted frames often have extreme patterns
            if mean_var < 10 or var_of_vars > 10000:
                return True
        
        return False
    
    def _connect(self) -> bool:
        """Connect to camera source."""
        self.status_changed.emit(self.camera_id, "connecting")
        
        if self._is_webcam():
            if self._connect_webcam():
                self.status_changed.emit(self.camera_id, "connected")
                return True
        
        elif self._is_rtsp():
            # Try OpenCV FFmpeg first (most reliable with TCP)
            logger.info(f"[{self.camera_id}] Trying OpenCV FFmpeg with TCP...")
            if self._connect_rtsp_opencv():
                self.status_changed.emit(self.camera_id, "connected")
                return True
            
            # Fallback to GStreamer
            logger.info(f"[{self.camera_id}] Trying GStreamer with TCP...")
            if self._connect_rtsp_gstreamer():
                self.status_changed.emit(self.camera_id, "connected")
                return True
        
        self.status_changed.emit(self.camera_id, "error")
        self.error_occurred.emit(self.camera_id, "Connection failed")
        return False
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to output resolution."""
        h, w = frame.shape[:2]
        
        if w <= self.OUTPUT_WIDTH and h <= self.OUTPUT_HEIGHT:
            return frame
        
        # Calculate aspect-preserving scale
        scale = min(self.OUTPUT_WIDTH / w, self.OUTPUT_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def run(self):
        """Main capture loop."""
        self._running = True
        
        if not self._connect():
            return
        
        frame_count = 0
        fps_start = time.time()
        reconnect_delay = 2.0
        consecutive_errors = 0
        
        while self._running:
            current_time = time.time()
            
            # Read frame
            with QMutexLocker(self._mutex):
                if self._cap is None:
                    break
                ret, frame = self._cap.read()
            
            # Handle read failure
            if not ret or frame is None:
                consecutive_errors += 1
                
                if consecutive_errors >= 5:
                    logger.warning(f"[{self.camera_id}] Reconnecting after errors...")
                    self.status_changed.emit(self.camera_id, "disconnected")
                    
                    with QMutexLocker(self._mutex):
                        if self._cap:
                            self._cap.release()
                            self._cap = None
                    
                    self.msleep(int(reconnect_delay * 1000))
                    
                    if self._running and self._connect():
                        consecutive_errors = 0
                        reconnect_delay = 2.0
                    else:
                        reconnect_delay = min(reconnect_delay * 1.5, 30)
                
                continue
            
            # Check for corruption
            if self._is_frame_corrupted(frame):
                consecutive_errors += 1
                continue
            
            consecutive_errors = 0
            
            # Resize frame
            frame = self._resize_frame(frame)
            
            # Emit for display
            if current_time - self._last_display_time >= self._display_interval:
                self._last_display_time = current_time
                self.frame_ready.emit(self.camera_id, frame)
                
                # FPS
                frame_count += 1
                elapsed = current_time - fps_start
                if elapsed >= 1.0:
                    self.fps_updated.emit(self.camera_id, frame_count / elapsed)
                    frame_count = 0
                    fps_start = current_time
            
            # Emit for recognition (less frequently)
            if current_time - self._last_recognition_time >= self._recognition_interval:
                self._last_recognition_time = current_time
                self.frame_for_recognition.emit(self.camera_id, frame.copy())
            
            # Prevent CPU spinning
            self.msleep(1)
        
        # Cleanup
        with QMutexLocker(self._mutex):
            if self._cap:
                self._cap.release()
                self._cap = None
        
        self.status_changed.emit(self.camera_id, "stopped")
    
    def stop(self):
        """Stop thread."""
        self._running = False
        self.wait(5000)
        logger.info(f"[{self.camera_id}] Stopped")