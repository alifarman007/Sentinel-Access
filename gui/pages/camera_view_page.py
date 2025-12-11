"""
Live Cameras Page - Simplified and Reliable
"""

import cv2
import numpy as np
import time
from typing import Dict, Optional, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QFrame, QListWidget,
    QListWidgetItem, QStackedWidget, QMessageBox,
    QComboBox
)
from PySide6.QtCore import Qt, Signal, Slot

from config.logging_config import logger
from config.settings import settings
from core.recognition_pipeline import RecognitionPipeline
from core.utils import draw_face_box
from gui.widgets.camera_widget import CameraWidget
from gui.widgets.camera_grid import CameraGrid
from gui.threads.simple_camera_thread import SimpleCameraThread
from gui.dialogs.add_camera_dialog import AddCameraDialog
from attendance.attendance_service import AttendanceService


class CameraViewPage(QWidget):
    """Live camera viewing page - simplified for reliability."""
    
    attendance_recorded = Signal(str, str, str)
    
    def __init__(
        self,
        pipeline: RecognitionPipeline,
        recognition_thread,
        parent=None
    ):
        super().__init__(parent)
        
        self.pipeline = pipeline
        
        # Camera threads
        self._camera_threads: Dict[str, SimpleCameraThread] = {}
        self._camera_info: Dict[str, dict] = {}
        
        # Recognition results cache
        self._latest_results: Dict[str, List] = {}
        
        # Attendance
        self.attendance_service = AttendanceService()
        self._last_attendance: Dict[str, float] = {}
        
        # View
        self._single_view_camera: Optional[str] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("Live Cameras")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        header.addWidget(title)
        
        header.addStretch()
        
        # View mode
        header.addWidget(QLabel("View:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Grid View", "Single Camera"])
        self.view_mode_combo.currentIndexChanged.connect(self._on_view_mode_changed)
        self.view_mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 12px;
                color: white;
                min-width: 120px;
            }
        """)
        header.addWidget(self.view_mode_combo)
        
        # Add camera
        add_btn = QPushButton("➕ Add Camera")
        add_btn.clicked.connect(self._show_add_camera_dialog)
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a6fa5;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #5a7fb5; }
        """)
        header.addWidget(add_btn)
        
        layout.addLayout(header)
        layout.addSpacing(15)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([200, 800])
        splitter.setStyleSheet("QSplitter::handle { background-color: #404040; width: 2px; }")
        
        layout.addWidget(splitter)
    
    def _create_left_panel(self) -> QWidget:
        """Create camera list panel."""
        panel = QFrame()
        panel.setStyleSheet("QFrame { background-color: #252525; border-radius: 8px; }")
        panel.setMinimumWidth(180)
        panel.setMaximumWidth(250)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("Cameras")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ccc;")
        layout.addWidget(title)
        
        self.camera_list = QListWidget()
        self.camera_list.setStyleSheet("""
            QListWidget {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 4px;
                color: white;
            }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #404040; }
            QListWidget::item:selected { background-color: #4a6fa5; }
            QListWidget::item:hover { background-color: #3d3d3d; }
        """)
        self.camera_list.itemClicked.connect(self._on_camera_selected)
        self.camera_list.itemDoubleClicked.connect(self._on_camera_double_clicked)
        layout.addWidget(self.camera_list)
        
        self.camera_count_label = QLabel("0 cameras")
        self.camera_count_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self.camera_count_label)
        
        remove_btn = QPushButton("🗑️ Remove Selected")
        remove_btn.clicked.connect(self._remove_selected_camera)
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px;
                color: #ccc;
            }
            QPushButton:hover { background-color: #4d4d4d; border-color: #F44336; color: #F44336; }
        """)
        layout.addWidget(remove_btn)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create view panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 0, 0, 0)
        
        self.view_stack = QStackedWidget()
        
        # Grid view
        self.camera_grid = CameraGrid()
        self.view_stack.addWidget(self.camera_grid)
        
        # Single view
        self.single_view = QWidget()
        single_layout = QVBoxLayout(self.single_view)
        single_layout.setContentsMargins(0, 0, 0, 0)
        
        self.single_camera_widget = CameraWidget("", "Select a camera")
        single_layout.addWidget(self.single_camera_widget)
        
        back_btn = QPushButton("← Back to Grid")
        back_btn.clicked.connect(lambda: self.view_mode_combo.setCurrentIndex(0))
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px 16px;
                color: #ccc;
            }
            QPushButton:hover { background-color: #4d4d4d; }
        """)
        single_layout.addWidget(back_btn, 0, Qt.AlignLeft)
        
        self.view_stack.addWidget(self.single_view)
        layout.addWidget(self.view_stack)
        
        return panel
    
    def _show_add_camera_dialog(self):
        """Show add camera dialog."""
        dialog = AddCameraDialog(
            existing_ids=list(self._camera_info.keys()),
            parent=self
        )
        
        if dialog.exec():
            data = dialog.get_camera_data()
            self._add_camera(data)
    
    def _add_camera(self, data: dict):
        """Add a camera."""
        camera_id = data['camera_id']
        name = data['name']
        source = data['source']
        
        logger.info(f"Adding camera: {name}, source: {source}")
        
        try:
            # Create simple camera thread
            thread = SimpleCameraThread(
                camera_id=camera_id,
                source=source,
                target_fps=12,
                recognition_fps=4
            )
            
            # Connect signals
            thread.frame_ready.connect(self._on_frame_ready)
            thread.frame_for_recognition.connect(self._on_frame_for_recognition)
            thread.status_changed.connect(self._on_status_changed)
            thread.fps_updated.connect(self._on_fps_updated)
            thread.error_occurred.connect(self._on_error)
            
            # Store
            self._camera_threads[camera_id] = thread
            self._camera_info[camera_id] = data
            
            # Add to grid
            self.camera_grid.add_camera(camera_id, name)
            
            # Add to list
            item = QListWidgetItem(f"📹 {name}")
            item.setData(Qt.UserRole, camera_id)
            self.camera_list.addItem(item)
            
            self._update_count()
            
            # Start thread
            thread.start()
            
            logger.info(f"Camera {camera_id} thread started")
            
        except Exception as e:
            logger.error(f"Failed to add camera: {e}")
            QMessageBox.critical(self, "Error", f"Failed to add camera:\n{e}")
    
    def _remove_selected_camera(self):
        """Remove selected camera."""
        item = self.camera_list.currentItem()
        if not item:
            return
        
        camera_id = item.data(Qt.UserRole)
        name = self._camera_info.get(camera_id, {}).get('name', camera_id)
        
        if QMessageBox.question(
            self, "Confirm", f"Remove '{name}'?",
            QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes:
            self._remove_camera(camera_id)
    
    def _remove_camera(self, camera_id: str):
        """Remove a camera."""
        if camera_id in self._camera_threads:
            self._camera_threads[camera_id].stop()
            del self._camera_threads[camera_id]
        
        self._camera_info.pop(camera_id, None)
        self._latest_results.pop(camera_id, None)
        
        self.camera_grid.remove_camera(camera_id)
        
        for i in range(self.camera_list.count()):
            if self.camera_list.item(i).data(Qt.UserRole) == camera_id:
                self.camera_list.takeItem(i)
                break
        
        self._update_count()
    
    def _update_count(self):
        """Update camera count."""
        count = len(self._camera_info)
        self.camera_count_label.setText(f"{count} camera{'s' if count != 1 else ''}")
    
    @Slot(str, np.ndarray)
    def _on_frame_ready(self, camera_id: str, frame: np.ndarray):
        """Handle display frame."""
        # Draw recognition results
        display_frame = frame.copy()
        results = self._latest_results.get(camera_id, [])
        
        for result in results:
            color = (0, 255, 0) if result.is_known else (0, 0, 255)
            draw_face_box(
                display_frame,
                result.detection.bbox.tolist(),
                name=result.name,
                confidence=result.confidence,
                color=color
            )
        
        # Update widgets
        widget = self.camera_grid.get_camera_widget(camera_id)
        if widget:
            widget.update_frame(display_frame, results)
        
        if self._single_view_camera == camera_id:
            self.single_camera_widget.update_frame(display_frame, results)
    
    @Slot(str, np.ndarray)
    def _on_frame_for_recognition(self, camera_id: str, frame: np.ndarray):
        """Handle recognition frame."""
        try:
            results = self.pipeline.process_frame(frame, identify=True)
            self._latest_results[camera_id] = results
            
            # Record attendance
            for result in results:
                if result.is_known:
                    self._record_attendance(camera_id, result)
                    
        except Exception as e:
            logger.error(f"Recognition error: {e}")
    
    def _record_attendance(self, camera_id: str, result):
        """Record attendance."""
        person_id = result.person_id
        now = time.time()
        
        key = f"{camera_id}_{person_id}"
        if now - self._last_attendance.get(key, 0) < 2.0:
            return
        
        self._last_attendance[key] = now
        
        camera_type = self._camera_info.get(camera_id, {}).get('camera_type', 'entry')
        record_type = 'exit' if camera_type == 'exit' else 'entry'
        
        success, _, _ = self.attendance_service.record_attendance(
            person_id=person_id,
            camera_id=camera_id,
            confidence=result.confidence,
            record_type=record_type
        )
        
        if success:
            self.attendance_recorded.emit(person_id, result.name, camera_id)
            logger.info(f"Attendance: {result.name}")
    
    @Slot(str, str)
    def _on_status_changed(self, camera_id: str, status: str):
        """Handle status change."""
        self.camera_grid.update_camera_status(camera_id, status)
        if self._single_view_camera == camera_id:
            self.single_camera_widget.update_status(status)
    
    @Slot(str, float)
    def _on_fps_updated(self, camera_id: str, fps: float):
        """Handle FPS update."""
        self.camera_grid.update_camera_fps(camera_id, fps)
        if self._single_view_camera == camera_id:
            self.single_camera_widget.update_fps(fps)
    
    @Slot(str, str)
    def _on_error(self, camera_id: str, error: str):
        """Handle error."""
        logger.error(f"[{camera_id}] {error}")
    
    def _on_view_mode_changed(self, index: int):
        """Handle view mode change."""
        self.view_stack.setCurrentIndex(index)
        if index == 1 and not self._single_view_camera and self._camera_info:
            self._single_view_camera = list(self._camera_info.keys())[0]
    
    def _on_camera_selected(self, item):
        """Handle camera selection."""
        camera_id = item.data(Qt.UserRole)
        if self.view_mode_combo.currentIndex() == 1:
            self._single_view_camera = camera_id
            info = self._camera_info.get(camera_id, {})
            self.single_camera_widget.set_camera_info(camera_id, info.get('name', camera_id))
    
    def _on_camera_double_clicked(self, item):
        """Handle double-click."""
        camera_id = item.data(Qt.UserRole)
        self._single_view_camera = camera_id
        info = self._camera_info.get(camera_id, {})
        self.single_camera_widget.set_camera_info(camera_id, info.get('name', camera_id))
        self.view_mode_combo.setCurrentIndex(1)
    
    def stop_all_cameras(self):
        """Stop all cameras."""
        for thread in self._camera_threads.values():
            thread.stop()
        self._camera_threads.clear()