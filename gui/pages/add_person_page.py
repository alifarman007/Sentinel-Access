"""
Add Person Page

Complete page for person registration with camera capture.
"""

import cv2
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QFrame, QMessageBox,
    QFileDialog, QGroupBox
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QPixmap, QImage

from config.settings import settings
from config.logging_config import logger
from core.face_detector import FaceDetector, Detection
from core.face_recognizer import FaceRecognizer
from core.face_database import FaceDatabase
from core.utils import draw_face_box, crop_face
from gui.widgets.person_form import PersonForm
from gui.widgets.persons_table import PersonsTable
from gui.threads.camera_thread import CameraThread
from attendance.attendance_service import PersonService


class AddPersonPage(QWidget):
    """
    Page for registering new persons.
    
    Features:
    - Live camera preview
    - Face detection visualization
    - Photo capture or upload
    - Registration form
    - Registered persons list
    """
    
    # Signals
    person_registered = Signal(str, str)  # person_id, name
    person_deleted = Signal(str)  # person_id
    
    def __init__(
        self,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        database: FaceDatabase,
        parent=None
    ):
        super().__init__(parent)
        
        self.detector = detector
        self.recognizer = recognizer
        self.database = database
        
        # State
        self._current_frame: Optional[np.ndarray] = None
        self._current_detection: Optional[Detection] = None
        self._captured_face: Optional[np.ndarray] = None
        self._captured_embedding: Optional[np.ndarray] = None
        self._camera_active = False
        
        # Camera thread
        self._camera_thread: Optional[CameraThread] = None
        
        self._setup_ui()
        self._load_persons()
    
    def _setup_ui(self):
        """Setup the page UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Add New Person")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        layout.addWidget(title)
        
        layout.addSpacing(15)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Camera and capture
        left_panel = self._create_camera_panel()
        splitter.addWidget(left_panel)
        
        # Right side: Form and table
        right_panel = self._create_form_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter sizes
        splitter.setSizes([500, 600])
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #404040;
                width: 2px;
            }
        """)
        
        layout.addWidget(splitter)
    
    def _create_camera_panel(self) -> QWidget:
        """Create the camera preview panel."""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-radius: 8px;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Camera preview section
        camera_group = QGroupBox("Camera Preview")
        camera_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #ccc;
                border: 1px solid #404040;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        camera_layout = QVBoxLayout(camera_group)
        
        # Preview label
        self.preview_label = QLabel("Click 'Start Camera' to begin")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 2px dashed #404040;
                border-radius: 6px;
                color: #666;
                font-size: 14px;
            }
        """)
        camera_layout.addWidget(self.preview_label)
        
        # Camera controls
        cam_btn_layout = QHBoxLayout()
        
        self.start_cam_btn = QPushButton("📷 Start Camera")
        self.start_cam_btn.clicked.connect(self._toggle_camera)
        self._style_button(self.start_cam_btn)
        cam_btn_layout.addWidget(self.start_cam_btn)
        
        self.capture_btn = QPushButton("📸 Capture Face")
        self.capture_btn.clicked.connect(self._capture_face)
        self.capture_btn.setEnabled(False)
        self._style_button(self.capture_btn)
        cam_btn_layout.addWidget(self.capture_btn)
        
        camera_layout.addLayout(cam_btn_layout)
        
        layout.addWidget(camera_group)
        
        # Captured face section
        captured_group = QGroupBox("Captured Face")
        captured_group.setStyleSheet(camera_group.styleSheet())
        captured_layout = QVBoxLayout(captured_group)
        
        # Captured face preview
        self.captured_label = QLabel("No face captured")
        self.captured_label.setAlignment(Qt.AlignCenter)
        self.captured_label.setFixedSize(150, 150)
        self.captured_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 2px solid #404040;
                border-radius: 6px;
                color: #666;
            }
        """)
        
        captured_h_layout = QHBoxLayout()
        captured_h_layout.addStretch()
        captured_h_layout.addWidget(self.captured_label)
        captured_h_layout.addStretch()
        captured_layout.addLayout(captured_h_layout)
        
        # Upload button
        upload_btn = QPushButton("📁 Upload Image Instead")
        upload_btn.clicked.connect(self._upload_image)
        self._style_button(upload_btn, secondary=True)
        captured_layout.addWidget(upload_btn)
        
        # Status
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #888; font-size: 12px;")
        captured_layout.addWidget(self.status_label)
        
        layout.addWidget(captured_group)
        
        layout.addStretch()
        
        return panel
    
    def _create_form_panel(self) -> QWidget:
        """Create the form and table panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 0, 0, 0)
        
        # Form
        form_frame = QFrame()
        form_frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-radius: 8px;
            }
        """)
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(15, 15, 15, 15)
        
        self.person_form = PersonForm()
        self.person_form.submitted.connect(self._on_form_submitted)
        form_layout.addWidget(self.person_form)
        
        layout.addWidget(form_frame)
        
        layout.addSpacing(15)
        
        # Persons table
        table_frame = QFrame()
        table_frame.setStyleSheet(form_frame.styleSheet())
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(15, 15, 15, 15)
        
        self.persons_table = PersonsTable()
        self.persons_table.person_deleted.connect(self._on_person_deleted)
        self.persons_table.refresh_requested.connect(self._load_persons)
        table_layout.addWidget(self.persons_table)
        
        layout.addWidget(table_frame, 1)
        
        return panel
    
    def _style_button(self, button, secondary=False):
        """Apply button styling."""
        if secondary:
            button.setStyleSheet("""
                QPushButton {
                    background-color: #3d3d3d;
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: 10px 15px;
                    color: #ccc;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #4d4d4d;
                }
                QPushButton:pressed {
                    background-color: #2d2d2d;
                }
            """)
        else:
            button.setStyleSheet("""
                QPushButton {
                    background-color: #4a6fa5;
                    border: none;
                    border-radius: 4px;
                    padding: 10px 15px;
                    color: white;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #5a7fb5;
                }
                QPushButton:pressed {
                    background-color: #3a5f95;
                }
                QPushButton:disabled {
                    background-color: #555;
                    color: #888;
                }
            """)
    
    def _toggle_camera(self):
        """Start or stop camera."""
        if self._camera_active:
            self._stop_camera()
        else:
            self._start_camera()
    
    def _start_camera(self):
        """Start camera capture."""
        try:
            self._camera_thread = CameraThread(
                camera_id="registration",
                source="0",
                use_gstreamer=False,  # Use direct OpenCV for webcam
                target_fps=15
            )
            self._camera_thread.frame_ready.connect(self._on_frame_received)
            self._camera_thread.status_changed.connect(self._on_camera_status)
            self._camera_thread.start()
            
            self._camera_active = True
            self.start_cam_btn.setText("⏹️ Stop Camera")
            self.capture_btn.setEnabled(True)
            self.status_label.setText("Camera started")
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start camera:\n{e}")
    
    def _stop_camera(self):
        """Stop camera capture."""
        if self._camera_thread:
            self._camera_thread.stop()
            self._camera_thread = None
        
        self._camera_active = False
        self.start_cam_btn.setText("📷 Start Camera")
        self.capture_btn.setEnabled(False)
        self.preview_label.setText("Camera stopped")
        self.preview_label.setPixmap(QPixmap())
        self.status_label.setText("")
    
    @Slot(str, np.ndarray)
    def _on_frame_received(self, camera_id: str, frame: np.ndarray):
        """Handle received frame from camera thread."""
        self._current_frame = frame.copy()
        
        # Detect faces
        detections = self.detector.detect(frame)
        
        # Store first detection
        self._current_detection = detections[0] if detections else None
        
        # Draw on display frame
        display_frame = frame.copy()
        for det in detections:
            color = (0, 255, 0)  # Green
            draw_face_box(
                display_frame,
                det.bbox.tolist(),
                name="",
                confidence=det.confidence,
                color=color
            )
        
        # Update preview
        self._display_frame(display_frame, self.preview_label)
        
        # Update status
        if len(detections) == 0:
            self.status_label.setText("No face detected - Position your face in the frame")
            self.status_label.setStyleSheet("color: #F44336;")
        elif len(detections) == 1:
            self.status_label.setText("✓ Face detected - Ready to capture")
            self.status_label.setStyleSheet("color: #4CAF50;")
        else:
            self.status_label.setText(f"⚠️ Multiple faces ({len(detections)}) - Show only one face")
            self.status_label.setStyleSheet("color: #FFC107;")
    
    @Slot(str, str)
    def _on_camera_status(self, camera_id: str, status: str):
        """Handle camera status change."""
        if status == "error":
            QMessageBox.warning(self, "Camera Error", "Camera connection failed")
            self._stop_camera()
    
    def _display_frame(self, frame: np.ndarray, label: QLabel):
        """Display frame in a QLabel."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        scaled = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled)
    
    def _capture_face(self):
        """Capture current face for registration."""
        if self._current_frame is None:
            QMessageBox.warning(self, "No Frame", "No camera frame available")
            return
        
        if self._current_detection is None:
            QMessageBox.warning(self, "No Face", "No face detected in frame")
            return
        
        try:
            # Extract embedding
            embedding = self.recognizer.get_embedding(
                self._current_frame,
                self._current_detection.landmarks
            )
            
            # Crop face for display
            face_crop = crop_face(
                self._current_frame,
                self._current_detection.bbox.tolist(),
                margin=0.3
            )
            
            self._captured_face = face_crop
            self._captured_embedding = embedding
            
            # Display captured face
            self._display_frame(face_crop, self.captured_label)
            self.status_label.setText("✓ Face captured successfully!")
            self.status_label.setStyleSheet("color: #4CAF50;")
            
        except Exception as e:
            logger.error(f"Face capture failed: {e}")
            QMessageBox.critical(self, "Error", f"Failed to capture face:\n{e}")
    
    def _upload_image(self):
        """Upload image file for registration."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Face Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not file_path:
            return
        
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Failed to load image")
            
            # Detect face
            detections = self.detector.detect(image)
            
            if len(detections) == 0:
                QMessageBox.warning(self, "No Face", "No face detected in the image")
                return
            
            if len(detections) > 1:
                QMessageBox.warning(
                    self, "Multiple Faces",
                    "Multiple faces detected. Please use an image with a single face."
                )
                return
            
            det = detections[0]
            
            # Extract embedding
            embedding = self.recognizer.get_embedding(image, det.landmarks)
            
            # Crop face
            face_crop = crop_face(image, det.bbox.tolist(), margin=0.3)
            
            self._captured_face = face_crop
            self._captured_embedding = embedding
            
            # Display
            self._display_frame(face_crop, self.captured_label)
            self.status_label.setText("✓ Face loaded from file!")
            self.status_label.setStyleSheet("color: #4CAF50;")
            
        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")
    
    @Slot(dict)
    def _on_form_submitted(self, data: dict):
        """Handle form submission."""
        # Check if face is captured
        if self._captured_embedding is None:
            QMessageBox.warning(
                self, "No Face",
                "Please capture or upload a face image before registering."
            )
            return
        
        person_id = data['person_id']
        name = data['name']
        
        try:
            # Check if person_id already exists in FAISS
            existing = self.database.get_person_by_id(person_id)
            if existing:
                QMessageBox.warning(
                    self, "Duplicate ID",
                    f"Person ID '{person_id}' already exists.\n"
                    "Please use a unique ID."
                )
                return
            
            # Check PostgreSQL too
            existing_pg = PersonService.get_person(person_id)
            if existing_pg:
                QMessageBox.warning(
                    self, "Duplicate ID",
                    f"Person ID '{person_id}' already exists in database.\n"
                    "Please use a unique ID."
                )
                return
            
            # Save face image
            image_filename = f"{person_id}_{uuid.uuid4().hex[:8]}.jpg"
            image_path = settings.faces_dir / image_filename
            settings.faces_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), self._captured_face)
            
            # Add to PostgreSQL
            success, person, msg = PersonService.create_person(
                person_id=person_id,
                name=name,
                department=data.get('department'),
                email=data.get('email'),
                phone=data.get('phone')
            )
            
            if not success:
                raise Exception(msg)
            
            # Add to FAISS
            faiss_id = self.database.add_person(
                person_id=person_id,
                name=name,
                embedding=self._captured_embedding,
                extra_metadata={'image_path': str(image_path)}
            )
            
            # Link embedding
            PersonService.link_face_embedding(person_id, faiss_id, str(image_path))
            
            # Save database
            self.database.save()
            
            # Success!
            QMessageBox.information(
                self, "Success",
                f"Successfully registered:\n\n"
                f"ID: {person_id}\n"
                f"Name: {name}"
            )
            
            # Clear form and captured face
            self.person_form.clear_form()
            self._clear_captured()
            
            # Refresh table
            self._load_persons()
            
            # Emit signal
            self.person_registered.emit(person_id, name)
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            QMessageBox.critical(self, "Error", f"Registration failed:\n{e}")
    
    def _clear_captured(self):
        """Clear captured face data."""
        self._captured_face = None
        self._captured_embedding = None
        self.captured_label.setPixmap(QPixmap())
        self.captured_label.setText("No face captured")
    
    @Slot(str)
    def _on_person_deleted(self, person_id: str):
        """Handle person deletion."""
        try:
            # Remove from FAISS
            self.database.remove_person(person_id)
            self.database.save()
            
            # Note: PostgreSQL deletion would need cascade
            # For now just reload the list
            
            self._load_persons()
            self.person_deleted.emit(person_id)
            
            QMessageBox.information(
                self, "Deleted",
                f"Person '{person_id}' has been removed."
            )
            
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            QMessageBox.critical(self, "Error", f"Failed to delete person:\n{e}")
    
    def _load_persons(self):
        """Load registered persons into table."""
        try:
            # Get from FAISS database (has image paths)
            faiss_persons = self.database.get_all_persons()
            
            # Format for table
            persons_data = []
            for p in faiss_persons:
                persons_data.append({
                    'id': p.get('faiss_id'),
                    'person_id': p.get('person_id'),
                    'name': p.get('name'),
                    'department': p.get('department'),
                    'image_path': p.get('image_path'),
                    'created_at': p.get('created_at')
                })
            
            self.persons_table.set_data(persons_data)
            
        except Exception as e:
            logger.error(f"Failed to load persons: {e}")
    
    def showEvent(self, event):
        """Handle page shown."""
        super().showEvent(event)
        self._load_persons()
    
    def hideEvent(self, event):
        """Handle page hidden."""
        super().hideEvent(event)
        self._stop_camera()