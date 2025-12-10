"""
Multi-Camera Manager

Manages multiple RTSP streams and distributes frames for processing.
"""

import threading
import time
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass

from config.settings import settings
from config.logging_config import logger
from camera.rtsp_stream import RTSPStream, StreamStatus


@dataclass
class CameraInfo:
    """Camera information."""
    camera_id: str
    name: str
    rtsp_url: str
    camera_type: str  # "entry" or "exit"
    status: StreamStatus
    fps: float


class CameraManager:
    """
    Multi-camera orchestration.
    
    Features:
    - Manage multiple RTSP streams
    - Thread-safe camera add/remove
    - Frame distribution to callbacks
    """
    
    def __init__(
        self,
        on_frame_callback: Optional[Callable] = None,
        max_cameras: int = None
    ):
        """
        Initialize camera manager.
        
        Args:
            on_frame_callback: Called when new frame arrives (camera_id, frame)
            max_cameras: Maximum number of cameras
        """
        self.on_frame_callback = on_frame_callback
        self.max_cameras = max_cameras or settings.MAX_CAMERAS
        
        # Camera storage
        self._cameras: Dict[str, RTSPStream] = {}
        self._camera_info: Dict[str, dict] = {}
        self._lock = threading.Lock()
        
        logger.info(f"CameraManager initialized: max_cameras={self.max_cameras}")
    
    def add_camera(
        self,
        camera_id: str,
        name: str,
        rtsp_url: str,
        camera_type: str = "entry",
        auto_start: bool = True
    ) -> Tuple[bool, str]:
        """
        Add a new camera.
        
        Args:
            camera_id: Unique camera ID
            name: Display name
            rtsp_url: RTSP URL
            camera_type: "entry" or "exit"
            auto_start: Start stream immediately
        
        Returns:
            (success, message)
        """
        with self._lock:
            if camera_id in self._cameras:
                return False, f"Camera '{camera_id}' already exists"
            
            if len(self._cameras) >= self.max_cameras:
                return False, f"Maximum cameras ({self.max_cameras}) reached"
            
            # Create stream
            stream = RTSPStream(
                rtsp_url=rtsp_url,
                camera_id=camera_id,
                use_gstreamer=True,
                use_nvidia=settings.USE_GPU,
                frame_callback=self._handle_frame
            )
            
            self._cameras[camera_id] = stream
            self._camera_info[camera_id] = {
                'camera_id': camera_id,
                'name': name,
                'rtsp_url': rtsp_url,
                'camera_type': camera_type
            }
            
            if auto_start:
                stream.start()
            
            logger.info(f"Added camera: {name} ({camera_id})")
            return True, f"Added camera: {name}"
    
    def remove_camera(self, camera_id: str) -> Tuple[bool, str]:
        """Remove a camera."""
        with self._lock:
            if camera_id not in self._cameras:
                return False, f"Camera '{camera_id}' not found"
            
            stream = self._cameras.pop(camera_id)
            info = self._camera_info.pop(camera_id, {})
            stream.stop()
            
            logger.info(f"Removed camera: {info.get('name', camera_id)}")
            return True, f"Removed camera: {camera_id}"
    
    def _handle_frame(self, camera_id: str, frame: np.ndarray):
        """Internal frame handler."""
        if self.on_frame_callback:
            self.on_frame_callback(camera_id, frame)
    
    def get_frame(self, camera_id: str) -> Tuple[bool, Optional[np.ndarray]]:
        """Get latest frame from a specific camera."""
        with self._lock:
            stream = self._cameras.get(camera_id)
            if not stream:
                return False, None
            return stream.read()
    
    def get_all_frames(self) -> Dict[str, np.ndarray]:
        """Get latest frames from all cameras."""
        frames = {}
        with self._lock:
            for camera_id, stream in self._cameras.items():
                ret, frame = stream.read()
                if ret:
                    frames[camera_id] = frame
        return frames
    
    def get_camera_info(self, camera_id: str) -> Optional[CameraInfo]:
        """Get info for a specific camera."""
        with self._lock:
            stream = self._cameras.get(camera_id)
            info = self._camera_info.get(camera_id)
            
            if not stream or not info:
                return None
            
            return CameraInfo(
                camera_id=camera_id,
                name=info['name'],
                rtsp_url=info['rtsp_url'],
                camera_type=info['camera_type'],
                status=stream.status,
                fps=stream.fps
            )
    
    def get_all_cameras(self) -> List[CameraInfo]:
        """Get info for all cameras."""
        cameras = []
        with self._lock:
            for camera_id in self._cameras:
                info = self.get_camera_info(camera_id)
                if info:
                    cameras.append(info)
        return cameras
    
    def start_all(self):
        """Start all camera streams."""
        with self._lock:
            for stream in self._cameras.values():
                if not stream._running:
                    stream.start()
        logger.info("Started all cameras")
    
    def stop_all(self):
        """Stop all camera streams."""
        with self._lock:
            for stream in self._cameras.values():
                stream.stop()
        logger.info("Stopped all cameras")
    
    def get_camera_count(self) -> int:
        """Get number of cameras."""
        with self._lock:
            return len(self._cameras)
    
    def get_connected_count(self) -> int:
        """Get number of connected cameras."""
        with self._lock:
            return sum(1 for s in self._cameras.values() if s.is_connected)