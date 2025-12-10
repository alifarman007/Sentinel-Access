"""
Camera Grid Widget

Displays multiple cameras in a grid layout.
"""

from typing import Dict, List, Optional
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot
import numpy as np

from gui.widgets.camera_widget import CameraWidget
from core.recognition_pipeline import RecognitionResult


class CameraGrid(QWidget):
    """
    Grid layout for multiple camera feeds.
    
    Supports 1x1, 2x2, 3x3, 4x4 layouts.
    """
    
    # Signals
    camera_clicked = Signal(str)  # camera_id
    camera_double_clicked = Signal(str)  # camera_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._cameras: Dict[str, CameraWidget] = {}
        self._grid_size = (2, 2)  # Default 2x2 grid
        self._max_cameras = 4
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the widget UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        # Grid size selector
        toolbar.addWidget(QLabel("Layout:"))
        self._layout_combo = QComboBox()
        self._layout_combo.addItems(["1x1", "2x2", "3x3", "2x1", "1x2"])
        self._layout_combo.setCurrentText("2x2")
        self._layout_combo.currentTextChanged.connect(self._on_layout_changed)
        toolbar.addWidget(self._layout_combo)
        
        toolbar.addStretch()
        
        # Stats label
        self._stats_label = QLabel("Cameras: 0/4")
        toolbar.addWidget(self._stats_label)
        
        main_layout.addLayout(toolbar)
        
        # Camera grid
        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(5)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        
        main_layout.addWidget(self._grid_widget)
        
        # Initialize empty grid
        self._update_grid()
    
    def _on_layout_changed(self, layout_text: str):
        """Handle layout change."""
        layouts = {
            "1x1": (1, 1),
            "2x2": (2, 2),
            "3x3": (3, 3),
            "2x1": (2, 1),
            "1x2": (1, 2)
        }
        self._grid_size = layouts.get(layout_text, (2, 2))
        self._max_cameras = self._grid_size[0] * self._grid_size[1]
        self._update_grid()
    
    def _update_grid(self):
        """Update grid layout."""
        # Clear existing widgets from grid
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            # Don't delete camera widgets, just remove from layout
            if item.widget() and item.widget() not in self._cameras.values():
                item.widget().deleteLater()
        
        rows, cols = self._grid_size
        
        # Add camera widgets or placeholders
        camera_list = list(self._cameras.values())
        idx = 0
        
        for row in range(rows):
            for col in range(cols):
                if idx < len(camera_list):
                    widget = camera_list[idx]
                    idx += 1
                else:
                    # Placeholder
                    widget = self._create_placeholder(row, col)
                
                self._grid_layout.addWidget(widget, row, col)
        
        # Update stats
        self._stats_label.setText(f"Cameras: {len(self._cameras)}/{self._max_cameras}")
    
    def _create_placeholder(self, row: int, col: int) -> QWidget:
        """Create a placeholder widget for empty grid cells."""
        placeholder = QLabel(f"Camera {row * self._grid_size[1] + col + 1}\n(Empty)")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setMinimumSize(320, 240)
        placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        placeholder.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 2px dashed #404040;
                border-radius: 6px;
                color: #555;
                font-size: 14px;
            }
        """)
        return placeholder
    
    def add_camera(self, camera_id: str, camera_name: str) -> Optional[CameraWidget]:
        """
        Add a camera to the grid.
        
        Returns:
            CameraWidget if added, None if grid is full
        """
        if camera_id in self._cameras:
            return self._cameras[camera_id]
        
        if len(self._cameras) >= self._max_cameras:
            return None
        
        # Create camera widget
        widget = CameraWidget(camera_id, camera_name)
        widget.clicked.connect(self.camera_clicked.emit)
        widget.double_clicked.connect(self.camera_double_clicked.emit)
        
        self._cameras[camera_id] = widget
        self._update_grid()
        
        return widget
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera from the grid."""
        if camera_id not in self._cameras:
            return False
        
        widget = self._cameras.pop(camera_id)
        widget.deleteLater()
        self._update_grid()
        
        return True
    
    def get_camera_widget(self, camera_id: str) -> Optional[CameraWidget]:
        """Get camera widget by ID."""
        return self._cameras.get(camera_id)
    
    @Slot(str, np.ndarray, list)
    def update_camera_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        results: List[RecognitionResult] = None
    ):
        """Update frame for a specific camera."""
        widget = self._cameras.get(camera_id)
        if widget:
            widget.update_frame(frame, results)
    
    @Slot(str, float)
    def update_camera_fps(self, camera_id: str, fps: float):
        """Update FPS for a specific camera."""
        widget = self._cameras.get(camera_id)
        if widget:
            widget.update_fps(fps)
    
    @Slot(str, str)
    def update_camera_status(self, camera_id: str, status: str):
        """Update status for a specific camera."""
        widget = self._cameras.get(camera_id)
        if widget:
            widget.update_status(status)
    
    def get_all_camera_ids(self) -> List[str]:
        """Get list of all camera IDs."""
        return list(self._cameras.keys())
    
    def clear_all(self):
        """Remove all cameras."""
        for camera_id in list(self._cameras.keys()):
            self.remove_camera(camera_id)