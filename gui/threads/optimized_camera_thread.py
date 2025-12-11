"""
Optimized Camera Thread for GUI

Memory-efficient camera capture with frame skipping.
"""

import cv2
import numpy as np
import time
from typing import Optional
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

from config.logging_config import logger


class OptimizedCameraThread(QThread):
    """
    Optimized camera thread with hardware scaling.
    
    Features:
    - Hardware-accelerated decode/scale
    - Frame skipping for recognition
    - Memory-efficient buffering
    - Automatic resolution adjustment
    """
    
    # Target resolution (720p optimal for face detection)
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720
    
    # Signals
    frame_ready = Signal(str, np.ndarray)      # camera_id, frame (for display)
    frame_for_recognition = Signal(str, np.ndarray)  # camera_id, frame (for recognition)
    status_changed = Signal(str, str)           # camera_id, status
    fps_updated = Signal(str, float)            # camera_id, fps
    error_occurred = Signal(str, str)           # camera_id, error
    
    def __init__(
        self,
        camera_id: str,
        source: str,
        target_fps: int = 15,
        recognition_fps: int = 5,  # Recognition at lower FPS
        use_gpu: bool = True,
        parent=None
    ):
        super().__init__(parent)
        
        self.camera_id = camera_id
        self.source = source
        self.target_fps = target_fps
        self.recognition_fps = recognition_fps
        self.use_gpu = use_gpu
        
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._mutex = QMutex()
        
        # Frame timing
        self._display_interval = 1.0 / target_fps
        self._recognition_interval = 1.0 / recognition_fps
        self._last_display_time = 0
        self._last_recognition_time = 0
    
    def _build_gstreamer_pipeline(self) -> str:
        """Build optimized GStreamer pipeline."""
        if self.source.isdigit():
            # Webcam - no GStreamer needed
            return None
        
        if self.use_gpu:
            # NVIDIA hardware decode + scale
            pipeline = (
                f"rtspsrc location={self.source} latency=100 buffer-mode=auto ! "
                "queue max-size-buffers=2 leaky=downstream ! "
                "rtph264depay ! h264parse ! "
                "nvh264dec ! "
                "nvvideoconvert ! "
                f"video/x-raw,format=BGRx,width={self.TARGET_WIDTH},height={self.TARGET_HEIGHT} ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true sync=false max-buffers=1"
            )
        else:
            # CPU decode + scale
            pipeline = (
                f"rtspsrc location={self.source} latency=200 buffer-mode=auto ! "
                "queue max-size-buffers=2 leaky=downstream ! "
                "rtph264depay ! h264parse ! "
                "avdec_h264 ! "
                "videoscale ! "
                f"video/x-raw,width={self.TARGET_WIDTH},height={self.TARGET_HEIGHT} ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true sync=false max-buffers=1"
            )
        
        return pipeline
    
    def _connect(self) -> bool:
        """Connect to camera source."""
        self.status_changed.emit(self.camera_id, "connecting")
        
        # Webcam
        if self.source.isdigit():
            try:
                self._cap = cv2.VideoCapture(int(self.source))
                if self._cap.isOpened():
                    # Set webcam to 720p if supported
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.TARGET_WIDTH)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.TARGET_HEIGHT)
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    ret, frame = self._cap.read()
                    if ret:
                        logger.info(f"[{self.camera_id}] Webcam connected: {frame.shape[1]}x{frame.shape[0]}")
                        self.status_changed.emit(self.camera_id, "connected")
                        return True
            except Exception as e:
                logger.error(f"[{self.camera_id}] Webcam error: {e}")
            
            self.status_changed.emit(self.camera_id, "error")
            self.error_occurred.emit(self.camera_id, "Failed to open webcam")
            return False
        
        # RTSP - Try GStreamer pipeline
        pipeline = self._build_gstreamer_pipeline()
        
        if pipeline:
            try:
                logger.info(f"[{self.camera_id}] Trying GStreamer pipeline...")
                self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                
                if self._cap.isOpened():
                    ret, frame = self._cap.read()
                    if ret and frame is not None:
                        logger.info(
                            f"[{self.camera_id}] GStreamer connected: "
                            f"{frame.shape[1]}x{frame.shape[0]}"
                        )
                        self.status_changed.emit(self.camera_id, "connected")
                        return True
                    self._cap.release()
            except Exception as e:
                logger.warning(f"[{self.camera_id}] GStreamer failed: {e}")
        
        # Fallback: Direct OpenCV with manual resize
        try:
            logger.info(f"[{self.camera_id}] Trying direct OpenCV...")
            self._cap = cv2.VideoCapture(self.source)
            
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, frame = self._cap.read()
                
                if ret and frame is not None:
                    logger.info(
                        f"[{self.camera_id}] Direct OpenCV connected: "
                        f"{frame.shape[1]}x{frame.shape[0]} (will resize)"
                    )
                    self.status_changed.emit(self.camera_id, "connected")
                    return True
        except Exception as e:
            logger.error(f"[{self.camera_id}] Direct OpenCV failed: {e}")
        
        self.status_changed.emit(self.camera_id, "error")
        self.error_occurred.emit(self.camera_id, "Failed to connect to camera")
        return False
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target resolution."""
        h, w = frame.shape[:2]
        
        if w <= self.TARGET_WIDTH and h <= self.TARGET_HEIGHT:
            return frame
        
        # Aspect-preserving resize
        scale = min(self.TARGET_WIDTH / w, self.TARGET_HEIGHT / h)
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
        
        while self._running:
            current_time = time.time()
            
            # Read frame
            with QMutexLocker(self._mutex):
                if self._cap is None:
                    break
                ret, frame = self._cap.read()
            
            if not ret or frame is None:
                self.status_changed.emit(self.camera_id, "disconnected")
                self.msleep(int(reconnect_delay * 1000))
                
                if self._running:
                    if self._connect():
                        reconnect_delay = 2.0
                    else:
                        reconnect_delay = min(reconnect_delay * 1.5, 30)
                continue
            
            # Resize if needed
            if frame.shape[1] > self.TARGET_WIDTH or frame.shape[0] > self.TARGET_HEIGHT:
                frame = self._resize_frame(frame)
            
            # Emit for display (at display FPS)
            if current_time - self._last_display_time >= self._display_interval:
                self._last_display_time = current_time
                self.frame_ready.emit(self.camera_id, frame)
                
                # FPS calculation
                frame_count += 1
                elapsed = current_time - fps_start
                if elapsed >= 1.0:
                    self.fps_updated.emit(self.camera_id, frame_count / elapsed)
                    frame_count = 0
                    fps_start = current_time
            
            # Emit for recognition (at lower FPS)
            if current_time - self._last_recognition_time >= self._recognition_interval:
                self._last_recognition_time = current_time
                # Send copy only for recognition
                self.frame_for_recognition.emit(self.camera_id, frame.copy())
            
            # Small sleep to prevent CPU spinning
            self.msleep(1)
        
        # Cleanup
        with QMutexLocker(self._mutex):
            if self._cap:
                self._cap.release()
                self._cap = None
        
        self.status_changed.emit(self.camera_id, "stopped")
    
    def stop(self):
        """Stop the capture thread."""
        self._running = False
        self.wait(3000)
        logger.info(f"[{self.camera_id}] Thread stopped")