"""
Camera Thread for GUI

Handles camera capture in background thread with Qt signals.
"""

import cv2
import numpy as np
from typing import Optional
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

from config.logging_config import logger


class CameraThread(QThread):
    """
    Background thread for camera capture.
    
    Emits frames via Qt signal for thread-safe GUI updates.
    """
    
    # Signals
    frame_ready = Signal(str, np.ndarray)  # camera_id, frame
    status_changed = Signal(str, str)       # camera_id, status
    fps_updated = Signal(str, float)        # camera_id, fps
    error_occurred = Signal(str, str)       # camera_id, error_message
    
    def __init__(
        self,
        camera_id: str,
        source: str,  # RTSP URL or camera index
        use_gstreamer: bool = True,
        use_nvidia: bool = True,
        target_fps: int = 15,
        parent=None
    ):
        super().__init__(parent)
        
        self.camera_id = camera_id
        self.source = source
        self.use_gstreamer = use_gstreamer
        self.use_nvidia = use_nvidia
        self.target_fps = target_fps
        
        self._running = False
        self._mutex = QMutex()
        self._cap: Optional[cv2.VideoCapture] = None
        
        # Frame interval for target FPS
        self._frame_interval = 1.0 / target_fps
    
    def _build_pipeline(self) -> str:
        """Build GStreamer pipeline string."""
        # Check if source is webcam (integer) or RTSP URL
        if self.source.isdigit():
            return int(self.source)
        
        if not self.use_gstreamer:
            return self.source
        
        if self.use_nvidia:
            pipeline = (
                f"rtspsrc location={self.source} latency=100 ! "
                "queue max-size-buffers=4 leaky=downstream ! "
                "rtph264depay ! h264parse ! "
                "nvh264dec ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink drop=true sync=false max-buffers=2"
            )
        else:
            pipeline = (
                f"rtspsrc location={self.source} latency=200 ! "
                "queue max-size-buffers=4 leaky=downstream ! "
                "rtph264depay ! h264parse ! "
                "avdec_h264 ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink drop=true sync=false max-buffers=2"
            )
        
        return pipeline
    
    def _connect(self) -> bool:
        """Connect to camera source."""
        self.status_changed.emit(self.camera_id, "connecting")
        
        pipeline = self._build_pipeline()
        
        try:
            if isinstance(pipeline, int):
                # Webcam
                self._cap = cv2.VideoCapture(pipeline)
            elif self.use_gstreamer:
                # GStreamer pipeline
                self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                # Direct RTSP
                self._cap = cv2.VideoCapture(self.source)
            
            if not self._cap.isOpened():
                # Fallback attempts
                if self.use_nvidia and self.use_gstreamer:
                    logger.warning(f"[{self.camera_id}] NVIDIA failed, trying CPU...")
                    self.use_nvidia = False
                    return self._connect()
                
                if self.use_gstreamer:
                    logger.warning(f"[{self.camera_id}] GStreamer failed, trying direct...")
                    self.use_gstreamer = False
                    return self._connect()
                
                self.status_changed.emit(self.camera_id, "error")
                self.error_occurred.emit(self.camera_id, "Failed to connect")
                return False
            
            # Test read
            ret, frame = self._cap.read()
            if not ret:
                self.status_changed.emit(self.camera_id, "error")
                self.error_occurred.emit(self.camera_id, "Cannot read frames")
                return False
            
            self.status_changed.emit(self.camera_id, "connected")
            logger.info(f"[{self.camera_id}] Connected: {frame.shape[1]}x{frame.shape[0]}")
            return True
            
        except Exception as e:
            self.status_changed.emit(self.camera_id, "error")
            self.error_occurred.emit(self.camera_id, str(e))
            return False
    
    def run(self):
        """Main capture loop."""
        self._running = True
        
        if not self._connect():
            return
        
        import time
        frame_count = 0
        fps_start = time.time()
        last_frame_time = 0
        
        while self._running:
            current_time = time.time()
            
            # Frame rate limiting
            if current_time - last_frame_time < self._frame_interval:
                self.msleep(1)
                continue
            
            with QMutexLocker(self._mutex):
                if self._cap is None:
                    break
                
                ret, frame = self._cap.read()
            
            if not ret or frame is None:
                self.status_changed.emit(self.camera_id, "disconnected")
                self.msleep(1000)
                
                # Try reconnect
                if self._running:
                    self._connect()
                continue
            
            last_frame_time = current_time
            
            # Emit frame
            self.frame_ready.emit(self.camera_id, frame)
            
            # Calculate FPS
            frame_count += 1
            elapsed = current_time - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                self.fps_updated.emit(self.camera_id, fps)
                frame_count = 0
                fps_start = current_time
        
        # Cleanup
        with QMutexLocker(self._mutex):
            if self._cap:
                self._cap.release()
                self._cap = None
        
        self.status_changed.emit(self.camera_id, "stopped")
    
    def stop(self):
        """Stop the capture thread."""
        self._running = False
        self.wait(3000)  # Wait up to 3 seconds
        logger.info(f"[{self.camera_id}] Thread stopped")