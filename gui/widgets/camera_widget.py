"""
Camera Display Widget

Displays camera feed with overlays.
"""

import cv2
import numpy as np
from typing import Optional, List
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
    QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont

from core.recognition_pipeline import RecognitionResult
from core.utils import draw_face_box


class CameraWidget(QWidget):
    """
    Widget for displaying a single camera feed.
    
    Features:
    - Frame display with aspect ratio preservation
    - Face detection overlay
    - Status indicator
    - FPS display
    """
    
    # Signals
    clicked = Signal(str)  # camera_id
    double_clicked = Signal(str)  # camera_id
    
    def __init__(
        self,
        camera_id: str = "",
        camera_name: str = "Camera",
        parent=None
    ):
        super().__init__(parent)
        
        self.camera_id = camera_id
        self.camera_name = camera_name
        self._current_frame: Optional[np.ndarray] = None
        self._results: List[RecognitionResult] = []
        self._fps = 0.0
        self._status = "disconnected"
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Header with camera name
        header = QHBoxLayout()
        
        # Status indicator
        self._status_indicator = QLabel("●")
        self._status_indicator.setFixedWidth(20)
        self._update_status_color()
        header.addWidget(self._status_indicator)
        
        # Camera name
        self._name_label = QLabel(self.camera_name)
        self._name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header.addWidget(self._name_label)
        
        header.addStretch()
        
        # FPS label
        self._fps_label = QLabel("0 FPS")
        self._fps_label.setStyleSheet("color: #888; font-size: 11px;")
        header.addWidget(self._fps_label)
        
        layout.addLayout(header)
        
        # Video frame
        self._frame_label = QLabel()
        self._frame_label.setAlignment(Qt.AlignCenter)
        self._frame_label.setMinimumSize(320, 240)
        self._frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._frame_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        
        # Placeholder text
        self._frame_label.setText("No Signal")
        self._frame_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 4px;
                color: #666;
                font-size: 14px;
            }
        """)
        
        layout.addWidget(self._frame_label)
        
        # Set frame style
        self.setFrameStyle()
    
    def setFrameStyle(self):
        """Set widget frame style."""
        self.setStyleSheet("""
            CameraWidget {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 6px;
            }
        """)
    
    def _update_status_color(self):
        """Update status indicator color."""
        colors = {
            "connected": "#4CAF50",      # Green
            "connecting": "#FFC107",     # Yellow
            "disconnected": "#9E9E9E",   # Gray
            "error": "#F44336",          # Red
            "stopped": "#9E9E9E"         # Gray
        }
        color = colors.get(self._status, "#9E9E9E")
        self._status_indicator.setStyleSheet(f"color: {color}; font-size: 14px;")
    
    @Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray, results: List[RecognitionResult] = None):
        """
        Update displayed frame.
        
        Args:
            frame: BGR image from camera
            results: Optional recognition results to overlay
        """
        if frame is None:
            return
        
        self._current_frame = frame.copy()
        self._results = results or []
        
        # Draw overlays on frame
        display_frame = frame.copy()
        
        for result in self._results:
            color = (0, 255, 0) if result.is_known else (0, 0, 255)
            draw_face_box(
                display_frame,
                result.detection.bbox.tolist(),
                name=result.name,
                confidence=result.confidence,
                color=color
            )
        
        # Convert to QPixmap and display
        self._display_frame(display_frame)
    
    def _display_frame(self, frame: np.ndarray):
        """Convert and display frame."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage
        q_img = QImage(
            rgb_frame.data,
            w, h,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        # Scale to fit label while preserving aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self._frame_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self._frame_label.setPixmap(scaled_pixmap)
    
    @Slot(float)
    def update_fps(self, fps: float):
        """Update FPS display."""
        self._fps = fps
        self._fps_label.setText(f"{fps:.1f} FPS")
    
    @Slot(str)
    def update_status(self, status: str):
        """Update connection status."""
        self._status = status
        self._update_status_color()
        
        if status == "disconnected" or status == "error":
            self._frame_label.setText("No Signal" if status == "disconnected" else "Error")
            self._frame_label.setPixmap(QPixmap())  # Clear pixmap
    
    def set_camera_info(self, camera_id: str, camera_name: str):
        """Update camera info."""
        self.camera_id = camera_id
        self.camera_name = camera_name
        self._name_label.setText(camera_name)
    
    def mousePressEvent(self, event):
        """Handle mouse click."""
        self.clicked.emit(self.camera_id)
        super().mousePressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """Handle double click."""
        self.double_clicked.emit(self.camera_id)
        super().mouseDoubleClickEvent(event)
    
    def sizeHint(self) -> QSize:
        """Preferred size."""
        return QSize(400, 320)