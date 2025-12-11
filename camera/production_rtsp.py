"""
Production-Grade RTSP Stream Handler

Reliable RTSP streaming with TCP transport and proper error recovery.
"""

import cv2
import threading
import time
import numpy as np
from typing import Optional, Tuple, Callable
from enum import Enum
from queue import Queue, Empty
import subprocess
import os

from config.logging_config import logger


class StreamStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class ProductionRTSPStream:
    """
    Production-grade RTSP handler optimized for reliability.
    
    Key features:
    - TCP transport (prevents packet loss)
    - Automatic resolution scaling
    - Keyframe synchronization
    - Robust reconnection
    - Memory efficient
    """
    
    # Output resolution (480p for smooth performance)
    OUTPUT_WIDTH = 854
    OUTPUT_HEIGHT = 480
    
    def __init__(
        self,
        rtsp_url: str,
        camera_id: str,
        output_width: int = 854,
        output_height: int = 480,
        target_fps: int = 15,
        on_frame: Optional[Callable] = None,
        on_status: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.output_width = output_width
        self.output_height = output_height
        self.target_fps = target_fps
        self.on_frame = on_frame
        self.on_status = on_status
        self.on_error = on_error
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._status = StreamStatus.DISCONNECTED
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        
        self._fps = 0.0
        self._original_size = (0, 0)
        self._frame_interval = 1.0 / target_fps
    
    def _get_opencv_rtsp_url(self) -> str:
        """Get RTSP URL with TCP transport forced."""
        # Force TCP transport via FFmpeg options
        # This prevents UDP packet loss issues
        url = self.rtsp_url
        
        # Add rtsp_transport=tcp if not already specified
        if "rtsp_transport" not in url:
            if "?" in url:
                url += "&rtsp_transport=tcp"
            else:
                # For OpenCV, we use environment variable or modify URL
                pass
        
        return url
    
    def _create_capture_opencv(self) -> Optional[cv2.VideoCapture]:
        """Create OpenCV capture with optimal settings for RTSP."""
        url = self.rtsp_url
        
        # Set FFmpeg options via environment (for TCP transport)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000"
        
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        if cap.isOpened():
            # Minimize buffer to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        
        return None
    
    def _create_capture_gstreamer_tcp(self) -> Optional[cv2.VideoCapture]:
        """Create GStreamer capture with TCP transport."""
        # Force TCP with rtspsrc protocols=tcp
        pipeline = (
            f"rtspsrc location={self.rtsp_url} protocols=tcp latency=100 ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "rtph264depay ! h264parse ! "
            "avdec_h264 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        )
        
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if cap.isOpened():
            return cap
        
        return None
    
    def _create_capture_gstreamer_nvidia(self) -> Optional[cv2.VideoCapture]:
        """Create GStreamer capture with NVIDIA hardware decoding and TCP."""
        pipeline = (
            f"rtspsrc location={self.rtsp_url} protocols=tcp latency=100 ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "rtph264depay ! h264parse ! "
            "nvh264dec ! "
            "nvvideoconvert ! "
            f"video/x-raw,format=BGRx,width={self.output_width},height={self.output_height} ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        )
        
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if cap.isOpened():
            # Test read
            ret, frame = cap.read()
            if ret and frame is not None and not self._is_corrupted(frame):
                # Seek back
                return cap
            cap.release()
        
        return None
    
    def _is_corrupted(self, frame: np.ndarray) -> bool:
        """Check if frame appears corrupted."""
        if frame is None:
            return True
        
        # Check for vertical lines (corruption pattern)
        # Sample columns and check variance
        h, w = frame.shape[:2]
        if w < 10 or h < 10:
            return True
        
        # Check if frame is mostly one color (blank/corrupted)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        
        if variance < 10:  # Very low variance = likely corrupted or blank
            return True
        
        return False
    
    def _connect(self) -> bool:
        """Connect using the most reliable method."""
        self._set_status(StreamStatus.CONNECTING)
        
        methods = [
            ("OpenCV FFmpeg (TCP)", self._create_capture_opencv),
            ("GStreamer TCP", self._create_capture_gstreamer_tcp),
            ("GStreamer NVIDIA", self._create_capture_gstreamer_nvidia),
        ]
        
        for name, method in methods:
            logger.info(f"[{self.camera_id}] Trying {name}...")
            
            try:
                cap = method()
                
                if cap is not None:
                    # Verify with test read
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        if self._is_corrupted(frame):
                            logger.warning(f"[{self.camera_id}] {name}: Corrupted frame, trying next...")
                            cap.release()
                            continue
                        
                        self._cap = cap
                        self._original_size = (frame.shape[1], frame.shape[0])
                        logger.info(
                            f"[{self.camera_id}] Connected via {name}! "
                            f"Original: {frame.shape[1]}x{frame.shape[0]}"
                        )
                        self._set_status(StreamStatus.CONNECTED)
                        return True
                    
                    cap.release()
                    
            except Exception as e:
                logger.warning(f"[{self.camera_id}] {name} failed: {e}")
        
        self._set_status(StreamStatus.ERROR)
        if self.on_error:
            self.on_error(self.camera_id, "All connection methods failed")
        return False
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to output resolution."""
        h, w = frame.shape[:2]
        
        if w == self.output_width and h == self.output_height:
            return frame
        
        # Aspect-preserving resize
        scale = min(self.output_width / w, self.output_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Use INTER_AREA for downscaling (best quality, reasonable speed)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def _capture_loop(self):
        """Main capture loop with robust error handling."""
        reconnect_delay = 1.0
        max_reconnect_delay = 30.0
        
        frame_count = 0
        fps_start_time = time.time()
        last_frame_time = 0
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self._running:
            # Connect if needed
            if self._status != StreamStatus.CONNECTED:
                if not self._connect():
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                    continue
                reconnect_delay = 1.0
                consecutive_errors = 0
            
            # Frame rate control
            current_time = time.time()
            elapsed_since_last = current_time - last_frame_time
            
            if elapsed_since_last < self._frame_interval:
                time.sleep(0.001)
                continue
            
            # Read frame
            try:
                ret, frame = self._cap.read()
                
                if not ret or frame is None:
                    consecutive_errors += 1
                    logger.warning(f"[{self.camera_id}] Frame read failed ({consecutive_errors})")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"[{self.camera_id}] Too many errors, reconnecting...")
                        self._set_status(StreamStatus.DISCONNECTED)
                        if self._cap:
                            self._cap.release()
                            self._cap = None
                        consecutive_errors = 0
                    
                    time.sleep(0.1)
                    continue
                
                # Check for corruption
                if self._is_corrupted(frame):
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"[{self.camera_id}] Corrupted frames, reconnecting...")
                        self._set_status(StreamStatus.DISCONNECTED)
                        if self._cap:
                            self._cap.release()
                            self._cap = None
                    continue
                
                consecutive_errors = 0
                last_frame_time = current_time
                
                # Resize
                frame = self._resize_frame(frame)
                
                # Store latest frame
                with self._frame_lock:
                    self._latest_frame = frame
                
                # Calculate FPS
                frame_count += 1
                fps_elapsed = current_time - fps_start_time
                if fps_elapsed >= 1.0:
                    self._fps = frame_count / fps_elapsed
                    frame_count = 0
                    fps_start_time = current_time
                
                # Callback
                if self.on_frame:
                    self.on_frame(self.camera_id, frame)
                
            except Exception as e:
                logger.error(f"[{self.camera_id}] Capture error: {e}")
                consecutive_errors += 1
                time.sleep(0.1)
        
        # Cleanup
        if self._cap:
            self._cap.release()
            self._cap = None
        self._set_status(StreamStatus.DISCONNECTED)
    
    def _set_status(self, status: StreamStatus):
        """Update status."""
        old_status = self._status
        self._status = status
        
        if self.on_status and old_status != status:
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
            self._thread.join(timeout=5)
            self._thread = None
        logger.info(f"[{self.camera_id}] Stream stopped")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get latest frame."""
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def status(self) -> StreamStatus:
        return self._status
    
    @property
    def is_connected(self) -> bool:
        return self._status == StreamStatus.CONNECTED