"""
Add Camera Dialog

Dialog for adding new camera sources.
"""

from typing import Optional, Tuple
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QComboBox, QPushButton, QLabel,
    QGroupBox, QMessageBox, QSpinBox, QWidget
)
from PySide6.QtCore import Qt, Signal


class AddCameraDialog(QDialog):
    """
    Dialog for adding a new camera.
    
    Supports:
    - RTSP URLs
    - Webcam indices
    - Camera type (entry/exit)
    """
    
    # Signal emitted when camera is added
    camera_added = Signal(dict)
    
    def __init__(self, existing_ids: list = None, parent=None):
        super().__init__(parent)
        
        self.existing_ids = existing_ids or []
        self.setWindowTitle("Add Camera")
        self.setFixedSize(450, 380)
        self.setModal(True)
        
        self._setup_ui()
        self._apply_style()
    
    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Source type selection
        source_group = QGroupBox("Camera Source")
        source_layout = QVBoxLayout(source_group)
        
        # Source type combo
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Source Type:"))
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["RTSP Stream", "USB Webcam"])
        self.source_type_combo.currentIndexChanged.connect(self._on_source_type_changed)
        type_layout.addWidget(self.source_type_combo, 1)
        source_layout.addLayout(type_layout)
        
        # RTSP URL input
        self.rtsp_widget = QWidget()
        rtsp_layout = QVBoxLayout(self.rtsp_widget)
        rtsp_layout.setContentsMargins(0, 5, 0, 0)
        
        rtsp_label = QLabel("RTSP URL:")
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("rtsp://username:password@192.168.1.100:554/stream1")
        rtsp_layout.addWidget(rtsp_label)
        rtsp_layout.addWidget(self.rtsp_input)
        
        # Example URLs
        example_label = QLabel(
            "Examples:\n"
            "• rtsp://admin:pass@192.168.1.64:554/stream1\n"
            "• rtsp://192.168.1.100:554/Streaming/Channels/101"
        )
        example_label.setStyleSheet("color: #888; font-size: 11px;")
        rtsp_layout.addWidget(example_label)
        
        source_layout.addWidget(self.rtsp_widget)
        
        # Webcam index input
        self.webcam_widget = QWidget()
        webcam_layout = QHBoxLayout(self.webcam_widget)
        webcam_layout.setContentsMargins(0, 5, 0, 0)
        
        webcam_layout.addWidget(QLabel("Camera Index:"))
        self.webcam_index = QSpinBox()
        self.webcam_index.setRange(0, 10)
        self.webcam_index.setValue(0)
        webcam_layout.addWidget(self.webcam_index)
        webcam_layout.addStretch()
        
        source_layout.addWidget(self.webcam_widget)
        self.webcam_widget.hide()
        
        layout.addWidget(source_group)
        
        # Camera details
        details_group = QGroupBox("Camera Details")
        details_layout = QFormLayout(details_group)
        details_layout.setSpacing(10)
        
        # Camera ID
        self.camera_id_input = QLineEdit()
        self.camera_id_input.setPlaceholderText("e.g., CAM001, ENTRANCE, LOBBY")
        self.camera_id_input.setMaxLength(50)
        details_layout.addRow("Camera ID *:", self.camera_id_input)
        
        # Camera Name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Main Entrance Camera")
        details_layout.addRow("Display Name *:", self.name_input)
        
        # Camera Type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Entry", "Exit", "General"])
        details_layout.addRow("Camera Type:", self.type_combo)
        
        # Location
        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText("e.g., Building A, Floor 1")
        details_layout.addRow("Location:", self.location_input)
        
        layout.addWidget(details_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        self.add_btn = QPushButton("Add Camera")
        self.add_btn.clicked.connect(self._on_add_clicked)
        self.add_btn.setDefault(True)
        btn_layout.addWidget(self.add_btn)
        
        layout.addLayout(btn_layout)
    
    def _apply_style(self):
        """Apply dialog styling."""
        self.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
            }
            QGroupBox {
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
            QLabel {
                color: #ccc;
            }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px;
                color: white;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #4a6fa5;
            }
            QPushButton {
                background-color: #4a6fa5;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #5a7fb5;
            }
            QPushButton:pressed {
                background-color: #3a5f95;
            }
        """)
    
    def _on_source_type_changed(self, index: int):
        """Handle source type change."""
        if index == 0:  # RTSP
            self.rtsp_widget.show()
            self.webcam_widget.hide()
        else:  # Webcam
            self.rtsp_widget.hide()
            self.webcam_widget.show()
    
    def _validate(self) -> Tuple[bool, str]:
        """Validate inputs."""
        camera_id = self.camera_id_input.text().strip()
        name = self.name_input.text().strip()
        
        if not camera_id:
            return False, "Camera ID is required"
        
        if camera_id in self.existing_ids:
            return False, f"Camera ID '{camera_id}' already exists"
        
        if not name:
            return False, "Display name is required"
        
        if self.source_type_combo.currentIndex() == 0:  # RTSP
            rtsp_url = self.rtsp_input.text().strip()
            if not rtsp_url:
                return False, "RTSP URL is required"
            if not rtsp_url.startswith("rtsp://"):
                return False, "RTSP URL must start with 'rtsp://'"
        
        return True, ""
    
    def _on_add_clicked(self):
        """Handle add button click."""
        valid, error = self._validate()
        
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        # Build camera data
        if self.source_type_combo.currentIndex() == 0:  # RTSP
            source = self.rtsp_input.text().strip()
        else:  # Webcam
            source = str(self.webcam_index.value())
        
        camera_data = {
            'camera_id': self.camera_id_input.text().strip(),
            'name': self.name_input.text().strip(),
            'source': source,
            'camera_type': self.type_combo.currentText().lower(),
            'location': self.location_input.text().strip() or None
        }
        
        self.camera_added.emit(camera_data)
        self.accept()
    
    def get_camera_data(self) -> dict:
        """Get the entered camera data."""
        if self.source_type_combo.currentIndex() == 0:
            source = self.rtsp_input.text().strip()
        else:
            source = str(self.webcam_index.value())
        
        return {
            'camera_id': self.camera_id_input.text().strip(),
            'name': self.name_input.text().strip(),
            'source': source,
            'camera_type': self.type_combo.currentText().lower(),
            'location': self.location_input.text().strip() or None
        }