"""
RTSP Stream Handler using OpenCV with GStreamer backend.

Supports hardware-accelerated decoding via NVDEC.
"""

import cv2
import threading
import queue
import time
import numpy as np
from typing import Optional, Tuple, Callable
from enum import Enum

from config.settings import settings
from config.logging_config import logger


class StreamStatus(Enum):
    """Stream status codes."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class RTSPStream:
    """
    RTSP Stream handler with GStreamer backend.
    
    Features:
    - Hardware decoding via NVDEC (when available)
    - Automatic reconnection
    - Frame buffering
    - Thread-safe operation
    """
    
    # GStreamer pipeline for NVIDIA hardware decoding
    GSTREAMER_PIPELINE_NVIDIA = (
        "rtspsrc location={url} latency={latency} ! "
        "queue max-size-buffers=4 leaky=downstream ! "
        "rtph264depay ! h264parse ! "
        "nvh264dec ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true sync=false max-buffers=2"
    )
    
    # GStreamer pipeline for CPU decoding (fallback)
    GSTREAMER_PIPELINE_CPU = (
        "rtspsrc location={url} latency={latency} ! "
        "queue max-size-buffers=4 leaky=downstream ! "
        "rtph264depay ! h264parse ! "
        "avdec_h264 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true sync=false max-buffers=2"
    )
    
    # Simple OpenCV RTSP (no GStreamer)
    OPENCV_DIRECT = "{url}"
    
    def __init__(
        self,
        rtsp_url: str,
        camera_id: str,
        use_gstreamer: bool = True,
        use_nvidia: bool = True,
        latency_ms: int = 100,
        reconnect_delay: float = 5.0,
        frame_callback: Optional[Callable] = None
    ):
        """
        Initialize RTSP stream.
        
        Args:
            rtsp_url: RTSP URL (rtsp://user:pass@ip:port/path)
            camera_id: Unique camera identifier
            use_gstreamer: Use GStreamer backend
            use_nvidia: Use NVIDIA hardware decoding
            latency_ms: Stream latency buffer
            reconnect_delay: Seconds to wait before reconnect
            frame_callback: Optional callback for new frames
        """
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.use_gstreamer = use_gstreamer
        self.use_nvidia = use_nvidia
        self.latency_ms = latency_ms
        self.reconnect_delay = reconnect_delay
        self.frame_callback = frame_callback
        
        # State
        self._cap: Optional[cv2.VideoCapture] = None
        self._status = StreamStatus.DISCONNECTED
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Frame buffer
        self._frame_queue = queue.Queue(maxsize=settings.FRAME_BUFFER_SIZE)
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        
        # Stats
        self._fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()
        
        logger.info(f"RTSPStream created: {camera_id} -> {rtsp_url}")
    
    def _build_pipeline(self) -> str:
        """Build GStreamer pipeline string."""
        if not self.use_gstreamer:
            return self.rtsp_url
        
        if self.use_nvidia:
            pipeline = self.GSTREAMER_PIPELINE_NVIDIA.format(
                url=self.rtsp_url,
                latency=self.latency_ms
            )
        else:
            pipeline = self.GSTREAMER_PIPELINE_CPU.format(
                url=self.rtsp_url,
                latency=self.latency_ms
            )
        
        return pipeline
    
    def _connect(self) -> bool:
        """Establish connection to RTSP stream."""
        self._status = StreamStatus.CONNECTING
        
        pipeline = self._build_pipeline()
        logger.info(f"[{self.camera_id}] Connecting with pipeline: {pipeline[:80]}...")
        
        try:
            if self.use_gstreamer:
                self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                self._cap = cv2.VideoCapture(self.rtsp_url)
            
            if not self._cap.isOpened():
                # Fallback to CPU pipeline
                if self.use_nvidia and self.use_gstreamer:
                    logger.warning(f"[{self.camera_id}] NVIDIA decode failed, trying CPU...")
                    self.use_nvidia = False
                    return self._connect()
                
                # Fallback to direct OpenCV
                if self.use_gstreamer:
                    logger.warning(f"[{self.camera_id}] GStreamer failed, trying direct OpenCV...")
                    self.use_gstreamer = False
                    return self._connect()
                
                logger.error(f"[{self.camera_id}] Failed to connect")
                self._status = StreamStatus.ERROR
                return False
            
            # Read a test frame
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.error(f"[{self.camera_id}] Connected but cannot read frames")
                self._status = StreamStatus.ERROR
                return False
            
            self._status = StreamStatus.CONNECTED
            logger.info(
                f"[{self.camera_id}] Connected! Frame size: {frame.shape[1]}x{frame.shape[0]}, "
                f"GStreamer: {self.use_gstreamer}, NVIDIA: {self.use_nvidia}"
            )
            return True
            
        except Exception as e:
            logger.error(f"[{self.camera_id}] Connection error: {e}")
            self._status = StreamStatus.ERROR
            return False
    
    def _capture_loop(self):
        """Main capture loop running in thread."""
        while self._running:
            # Connect if not connected
            if self._status != StreamStatus.CONNECTED:
                if not self._connect():
                    time.sleep(self.reconnect_delay)
                    continue
            
            # Read frame
            try:
                ret, frame = self._cap.read()
                
                if not ret or frame is None:
                    logger.warning(f"[{self.camera_id}] Frame read failed, reconnecting...")
                    self._status = StreamStatus.DISCONNECTED
                    if self._cap:
                        self._cap.release()
                    time.sleep(1)
                    continue
                
                # Update latest frame
                with self._frame_lock:
                    self._latest_frame = frame
                
                # Update stats
                self._frame_count += 1
                elapsed = time.time() - self._last_fps_time
                if elapsed >= 1.0:
                    self._fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._last_fps_time = time.time()
                
                # Call callback if set
                if self.frame_callback:
                    self.frame_callback(self.camera_id, frame)
                
            except Exception as e:
                logger.error(f"[{self.camera_id}] Capture error: {e}")
                self._status = StreamStatus.ERROR
                time.sleep(1)
        
        # Cleanup
        if self._cap:
            self._cap.release()
            self._cap = None
        self._status = StreamStatus.DISCONNECTED
    
    def start(self):
        """Start the capture thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"[{self.camera_id}] Stream started")
    
    def stop(self):
        """Stop the capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info(f"[{self.camera_id}] Stream stopped")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the latest frame.
        
        Returns:
            (success, frame)
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()
    
    @property
    def status(self) -> StreamStatus:
        """Get current stream status."""
        return self._status
    
    @property
    def fps(self) -> float:
        """Get current FPS."""
        return self._fps
    
    @property
    def is_connected(self) -> bool:
        """Check if stream is connected."""
        return self._status == StreamStatus.CONNECTED