"""
Live Cameras Page

Displays live camera feeds with recognition overlays.
"""

import cv2
import numpy as np
from typing import Dict, Optional, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QFrame, QListWidget,
    QListWidgetItem, QStackedWidget, QMessageBox,
    QComboBox, QGroupBox
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QPixmap, QImage

from config.logging_config import logger
from config.settings import settings
from core.recognition_pipeline import RecognitionPipeline, RecognitionResult
from core.utils import draw_face_box
from gui.widgets.camera_widget import CameraWidget
from gui.widgets.camera_grid import CameraGrid
from gui.threads.camera_thread import CameraThread
from gui.threads.recognition_thread import RecognitionThread
from gui.dialogs.add_camera_dialog import AddCameraDialog
from attendance.attendance_service import AttendanceService


class CameraViewPage(QWidget):
    """
    Live camera viewing page.
    
    Features:
    - Grid view (1x1, 2x2, 2x1)
    - Single camera fullscreen view
    - Camera list panel
    - Real-time face recognition
    - Attendance logging
    """
    
    # Signals
    attendance_recorded = Signal(str, str, str)  # person_id, name, camera_id
    
    def __init__(
        self,
        pipeline: RecognitionPipeline,
        recognition_thread: RecognitionThread,
        parent=None
    ):
        super().__init__(parent)
        
        self.pipeline = pipeline
        self.recognition_thread = recognition_thread
        
        # Camera management
        self._camera_threads: Dict[str, CameraThread] = {}
        self._camera_info: Dict[str, dict] = {}
        
        # Attendance service
        self.attendance_service = AttendanceService()
        
        # Last recognition time per person (for rate limiting)
        self._last_recognition: Dict[str, float] = {}
        
        # View mode
        self._single_view_camera: Optional[str] = None
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Setup the page UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("Live Cameras")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        header.addWidget(title)
        
        header.addStretch()
        
        # View mode selector
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
        
        # Add camera button
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
            QPushButton:hover {
                background-color: #5a7fb5;
            }
        """)
        header.addWidget(add_btn)
        
        layout.addLayout(header)
        layout.addSpacing(15)
        
        # Main content with splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Camera list
        left_panel = self._create_camera_list_panel()
        splitter.addWidget(left_panel)
        
        # Right: Camera views
        right_panel = self._create_view_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([200, 800])
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #404040;
                width: 2px;
            }
        """)
        
        layout.addWidget(splitter)
    
    def _create_camera_list_panel(self) -> QWidget:
        """Create the camera list panel."""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-radius: 8px;
            }
        """)
        panel.setMinimumWidth(180)
        panel.setMaximumWidth(250)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Cameras")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ccc;")
        layout.addWidget(title)
        
        # Camera list
        self.camera_list = QListWidget()
        self.camera_list.setStyleSheet("""
            QListWidget {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 4px;
                color: white;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #404040;
            }
            QListWidget::item:selected {
                background-color: #4a6fa5;
            }
            QListWidget::item:hover {
                background-color: #3d3d3d;
            }
        """)
        self.camera_list.itemClicked.connect(self._on_camera_selected)
        self.camera_list.itemDoubleClicked.connect(self._on_camera_double_clicked)
        layout.addWidget(self.camera_list)
        
        # Camera count
        self.camera_count_label = QLabel("0 cameras")
        self.camera_count_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self.camera_count_label)
        
        # Remove button
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
            QPushButton:hover {
                background-color: #4d4d4d;
                border-color: #F44336;
                color: #F44336;
            }
        """)
        layout.addWidget(remove_btn)
        
        return panel
    
    def _create_view_panel(self) -> QWidget:
        """Create the camera view panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 0, 0, 0)
        
        # Stacked widget for grid/single views
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
        
        # Back to grid button
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
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        single_layout.addWidget(back_btn, 0, Qt.AlignLeft)
        
        self.view_stack.addWidget(self.single_view)
        
        layout.addWidget(self.view_stack)
        
        return panel
    
    def _connect_signals(self):
        """Connect signals."""
        self.recognition_thread.recognition_complete.connect(self._on_recognition_complete)
    
    def _show_add_camera_dialog(self):
        """Show add camera dialog."""
        existing_ids = list(self._camera_info.keys())
        
        dialog = AddCameraDialog(existing_ids=existing_ids, parent=self)
        
        if dialog.exec():
            camera_data = dialog.get_camera_data()
            self._add_camera(camera_data)
    
    def _add_camera(self, camera_data: dict):
        """Add a new camera."""
        camera_id = camera_data['camera_id']
        name = camera_data['name']
        source = camera_data['source']
        
        try:
            # Create camera thread
            thread = CameraThread(
                camera_id=camera_id,
                source=source,
                use_gstreamer=source.startswith("rtsp://"),
                use_nvidia=settings.USE_GPU,
                target_fps=15
            )
            
            # Connect signals
            thread.frame_ready.connect(self._on_frame_received)
            thread.status_changed.connect(self._on_camera_status_changed)
            thread.fps_updated.connect(self._on_fps_updated)
            thread.error_occurred.connect(self._on_camera_error)
            
            # Store
            self._camera_threads[camera_id] = thread
            self._camera_info[camera_id] = camera_data
            
            # Add to grid
            self.camera_grid.add_camera(camera_id, name)
            
            # Add to list
            item = QListWidgetItem(f"📹 {name}")
            item.setData(Qt.UserRole, camera_id)
            self.camera_list.addItem(item)
            
            # Update count
            self._update_camera_count()
            
            # Start thread
            thread.start()
            
            logger.info(f"Added camera: {name} ({camera_id})")
            
        except Exception as e:
            logger.error(f"Failed to add camera: {e}")
            QMessageBox.critical(self, "Error", f"Failed to add camera:\n{e}")
    
    def _remove_selected_camera(self):
        """Remove the selected camera."""
        item = self.camera_list.currentItem()
        if not item:
            QMessageBox.information(self, "Info", "Please select a camera to remove")
            return
        
        camera_id = item.data(Qt.UserRole)
        camera_name = self._camera_info.get(camera_id, {}).get('name', camera_id)
        
        reply = QMessageBox.question(
            self,
            "Confirm Remove",
            f"Remove camera '{camera_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._remove_camera(camera_id)
    
    def _remove_camera(self, camera_id: str):
        """Remove a camera."""
        # Stop thread
        if camera_id in self._camera_threads:
            self._camera_threads[camera_id].stop()
            del self._camera_threads[camera_id]
        
        # Remove info
        if camera_id in self._camera_info:
            del self._camera_info[camera_id]
        
        # Remove from grid
        self.camera_grid.remove_camera(camera_id)
        
        # Remove from list
        for i in range(self.camera_list.count()):
            item = self.camera_list.item(i)
            if item.data(Qt.UserRole) == camera_id:
                self.camera_list.takeItem(i)
                break
        
        self._update_camera_count()
        logger.info(f"Removed camera: {camera_id}")
    
    def _update_camera_count(self):
        """Update camera count label."""
        count = len(self._camera_info)
        self.camera_count_label.setText(f"{count} camera{'s' if count != 1 else ''}")
    
    @Slot(str, np.ndarray)
    def _on_frame_received(self, camera_id: str, frame: np.ndarray):
        """Handle frame from camera thread."""
        # Submit to recognition thread
        self.recognition_thread.submit_frame(camera_id, frame)
    
    @Slot(object)
    def _on_recognition_complete(self, output):
        """Handle recognition results."""
        camera_id = output.camera_id
        frame = output.frame
        results = output.results
        
        # Draw results on frame
        display_frame = frame.copy()
        for result in results:
            color = (0, 255, 0) if result.is_known else (0, 0, 255)
            draw_face_box(
                display_frame,
                result.detection.bbox.tolist(),
                name=result.name,
                confidence=result.confidence,
                color=color
            )
            
            # Record attendance for known faces
            if result.is_known:
                self._record_attendance(camera_id, result)
        
        # Update grid view
        widget = self.camera_grid.get_camera_widget(camera_id)
        if widget:
            widget.update_frame(display_frame, results)
        
        # Update single view if this camera is selected
        if self._single_view_camera == camera_id:
            self.single_camera_widget.update_frame(display_frame, results)
    
    def _record_attendance(self, camera_id: str, result: RecognitionResult):
        """Record attendance for recognized person."""
        import time
        
        person_id = result.person_id
        now = time.time()
        
        # Rate limit: 1 second per person
        last_time = self._last_recognition.get(person_id, 0)
        if now - last_time < 1.0:
            return
        
        self._last_recognition[person_id] = now
        
        # Get camera type
        camera_info = self._camera_info.get(camera_id, {})
        camera_type = camera_info.get('camera_type', 'entry')
        record_type = 'exit' if camera_type == 'exit' else 'entry'
        
        # Record attendance
        success, record, msg = self.attendance_service.record_attendance(
            person_id=person_id,
            camera_id=camera_id,
            confidence=result.confidence,
            record_type=record_type
        )
        
        if success:
            self.attendance_recorded.emit(person_id, result.name, camera_id)
            logger.info(f"Attendance: {result.name} ({person_id}) - {record_type}")
    
    @Slot(str, str)
    def _on_camera_status_changed(self, camera_id: str, status: str):
        """Handle camera status change."""
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
    def _on_camera_error(self, camera_id: str, error: str):
        """Handle camera error."""
        logger.error(f"Camera {camera_id} error: {error}")
    
    def _on_view_mode_changed(self, index: int):
        """Handle view mode change."""
        if index == 0:  # Grid view
            self.view_stack.setCurrentIndex(0)
            self._single_view_camera = None
        else:  # Single view
            self.view_stack.setCurrentIndex(1)
            # Select first camera if none selected
            if not self._single_view_camera and self._camera_info:
                self._single_view_camera = list(self._camera_info.keys())[0]
                info = self._camera_info[self._single_view_camera]
                self.single_camera_widget.set_camera_info(
                    self._single_view_camera,
                    info['name']
                )
    
    def _on_camera_selected(self, item: QListWidgetItem):
        """Handle camera selection in list."""
        camera_id = item.data(Qt.UserRole)
        
        if self.view_mode_combo.currentIndex() == 1:  # Single view mode
            self._single_view_camera = camera_id
            info = self._camera_info.get(camera_id, {})
            self.single_camera_widget.set_camera_info(camera_id, info.get('name', camera_id))
    
    def _on_camera_double_clicked(self, item: QListWidgetItem):
        """Handle camera double-click - switch to single view."""
        camera_id = item.data(Qt.UserRole)
        
        self._single_view_camera = camera_id
        info = self._camera_info.get(camera_id, {})
        self.single_camera_widget.set_camera_info(camera_id, info.get('name', camera_id))
        
        # Switch to single view
        self.view_mode_combo.setCurrentIndex(1)
    
    def stop_all_cameras(self):
        """Stop all camera threads."""
        for thread in self._camera_threads.values():
            thread.stop()
        self._camera_threads.clear()
    
    def hideEvent(self, event):
        """Handle page hidden - optionally pause cameras."""
        super().hideEvent(event)
        # Keep cameras running in background for attendance
    
    def closeEvent(self, event):
        """Handle close - stop all cameras."""
        self.stop_all_cameras()
        super().closeEvent(event)