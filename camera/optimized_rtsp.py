"""
Optimized RTSP Stream Handler

Hardware-accelerated decoding and scaling for 4K streams.
"""

import cv2
import threading
import time
import numpy as np
from typing import Optional, Tuple, Callable
from enum import Enum
from collections import deque

from config.settings import settings
from config.logging_config import logger


class StreamStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class OptimizedRTSPStream:
    """
    Optimized RTSP stream handler with hardware scaling.
    
    Features:
    - Hardware decoding via NVDEC
    - Hardware scaling to target resolution
    - Frame skipping for performance
    - Memory-efficient buffering
    """
    
    # Target resolution for processing (720p is optimal for face detection)
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720
    
    def __init__(
        self,
        rtsp_url: str,
        camera_id: str,
        target_fps: int = 15,
        use_gpu: bool = True,
        on_frame: Optional[Callable] = None,
        on_status: Optional[Callable] = None
    ):
        """
        Initialize optimized RTSP stream.
        
        Args:
            rtsp_url: RTSP URL
            camera_id: Unique camera ID
            target_fps: Target frame rate
            use_gpu: Use NVIDIA hardware acceleration
            on_frame: Callback for new frames (camera_id, frame, original_size)
            on_status: Callback for status changes (camera_id, status)
        """
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.use_gpu = use_gpu
        self.on_frame = on_frame
        self.on_status = on_status
        
        # State
        self._cap: Optional[cv2.VideoCapture] = None
        self._status = StreamStatus.DISCONNECTED
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Frame buffer (small, latest only)
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        
        # Stats
        self._fps = 0.0
        self._frame_count = 0
        self._original_size = (0, 0)
        
        # Frame timing
        self._frame_interval = 1.0 / target_fps
        self._last_frame_time = 0
    
    def _build_gstreamer_pipeline(self) -> str:
        """Build optimized GStreamer pipeline with hardware scaling."""
        
        if self.use_gpu:
            # NVIDIA hardware decode + scale pipeline
            # nvh264dec decodes, nvvideoconvert scales on GPU
            pipeline = (
                f"rtspsrc location={self.rtsp_url} latency=100 buffer-mode=auto ! "
                "queue max-size-buffers=2 leaky=downstream ! "
                "rtph264depay ! h264parse ! "
                "nvh264dec ! "
                f"nvvideoconvert ! "
                f"video/x-raw,format=BGRx,width={self.TARGET_WIDTH},height={self.TARGET_HEIGHT} ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink drop=true sync=false max-buffers=1"
            )
        else:
            # CPU decode + scale pipeline
            pipeline = (
                f"rtspsrc location={self.rtsp_url} latency=200 buffer-mode=auto ! "
                "queue max-size-buffers=2 leaky=downstream ! "
                "rtph264depay ! h264parse ! "
                "avdec_h264 ! "
                f"videoscale ! "
                f"video/x-raw,width={self.TARGET_WIDTH},height={self.TARGET_HEIGHT} ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink drop=true sync=false max-buffers=1"
            )
        
        return pipeline
    
    def _build_fallback_pipeline(self) -> str:
        """Build simpler fallback pipeline."""
        # Simpler pipeline without hardware scaling
        pipeline = (
            f"rtspsrc location={self.rtsp_url} latency=300 ! "
            "queue max-size-buffers=2 leaky=downstream ! "
            "rtph264depay ! h264parse ! "
            "avdec_h264 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        )
        return pipeline
    
    def _connect(self) -> bool:
        """Connect to RTSP stream."""
        self._set_status(StreamStatus.CONNECTING)
        
        # Try GPU pipeline first
        if self.use_gpu:
            pipeline = self._build_gstreamer_pipeline()
            logger.info(f"[{self.camera_id}] Trying GPU pipeline...")
            
            try:
                self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if self._cap.isOpened():
                    ret, frame = self._cap.read()
                    if ret and frame is not None:
                        self._original_size = (frame.shape[1], frame.shape[0])
                        logger.info(
                            f"[{self.camera_id}] GPU pipeline connected! "
                            f"Frame: {frame.shape[1]}x{frame.shape[0]}"
                        )
                        self._set_status(StreamStatus.CONNECTED)
                        return True
                    self._cap.release()
            except Exception as e:
                logger.warning(f"[{self.camera_id}] GPU pipeline failed: {e}")
        
        # Try CPU pipeline
        logger.info(f"[{self.camera_id}] Trying CPU pipeline...")
        pipeline = self._build_fallback_pipeline()
        
        try:
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    self._original_size = (frame.shape[1], frame.shape[0])
                    logger.info(
                        f"[{self.camera_id}] CPU pipeline connected! "
                        f"Frame: {frame.shape[1]}x{frame.shape[0]} (will resize)"
                    )
                    self._set_status(StreamStatus.CONNECTED)
                    return True
                self._cap.release()
        except Exception as e:
            logger.warning(f"[{self.camera_id}] CPU pipeline failed: {e}")
        
        # Last resort: Direct OpenCV
        logger.info(f"[{self.camera_id}] Trying direct OpenCV...")
        try:
            self._cap = cv2.VideoCapture(self.rtsp_url)
            if self._cap.isOpened():
                # Set buffer size
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    self._original_size = (frame.shape[1], frame.shape[0])
                    logger.info(
                        f"[{self.camera_id}] Direct OpenCV connected! "
                        f"Frame: {frame.shape[1]}x{frame.shape[0]} (will resize)"
                    )
                    self._set_status(StreamStatus.CONNECTED)
                    return True
        except Exception as e:
            logger.error(f"[{self.camera_id}] Direct OpenCV failed: {e}")
        
        self._set_status(StreamStatus.ERROR)
        return False
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target resolution if needed."""
        h, w = frame.shape[:2]
        
        if w <= self.TARGET_WIDTH and h <= self.TARGET_HEIGHT:
            return frame
        
        # Calculate aspect-preserving resize
        scale = min(self.TARGET_WIDTH / w, self.TARGET_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Use INTER_AREA for downscaling (best quality)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _capture_loop(self):
        """Main capture loop."""
        reconnect_delay = 2.0
        fps_update_time = time.time()
        frame_count = 0
        
        while self._running:
            # Connect if needed
            if self._status != StreamStatus.CONNECTED:
                if not self._connect():
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, 30)  # Backoff
                    continue
                reconnect_delay = 2.0
            
            # Rate limiting
            current_time = time.time()
            elapsed = current_time - self._last_frame_time
            if elapsed < self._frame_interval:
                time.sleep(0.001)
                continue
            
            # Read frame
            try:
                ret, frame = self._cap.read()
                
                if not ret or frame is None:
                    logger.warning(f"[{self.camera_id}] Frame read failed")
                    self._set_status(StreamStatus.DISCONNECTED)
                    if self._cap:
                        self._cap.release()
                    time.sleep(0.5)
                    continue
                
                # Resize if needed (for non-GStreamer scaled streams)
                if frame.shape[1] > self.TARGET_WIDTH or frame.shape[0] > self.TARGET_HEIGHT:
                    frame = self._resize_frame(frame)
                
                self._last_frame_time = current_time
                
                # Update latest frame (no copy needed here)
                with self._frame_lock:
                    self._latest_frame = frame
                
                # FPS calculation
                frame_count += 1
                if current_time - fps_update_time >= 1.0:
                    self._fps = frame_count / (current_time - fps_update_time)
                    frame_count = 0
                    fps_update_time = current_time
                
                # Callback
                if self.on_frame:
                    self.on_frame(self.camera_id, frame, self._original_size)
                
            except Exception as e:
                logger.error(f"[{self.camera_id}] Capture error: {e}")
                self._set_status(StreamStatus.ERROR)
                time.sleep(1)
        
        # Cleanup
        if self._cap:
            self._cap.release()
            self._cap = None
        self._set_status(StreamStatus.DISCONNECTED)
    
    def _set_status(self, status: StreamStatus):
        """Update status and notify callback."""
        self._status = status
        if self.on_status:
            self.on_status(self.camera_id, status.value)
    
    def start(self):
        """Start capture thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"[{self.camera_id}] Stream started")
    
    def stop(self):
        """Stop capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        logger.info(f"[{self.camera_id}] Stream stopped")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get latest frame (no copy for efficiency)."""
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame
    
    def read_copy(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get copy of latest frame."""
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def status(self) -> StreamStatus:
        return self._status
    
    @property
    def is_connected(self) -> bool:
        return self._status == StreamStatus.CONNECTED
    
    @property
    def original_resolution(self) -> Tuple[int, int]:
        return self._original_size