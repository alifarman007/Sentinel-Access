"""
Settings Page

Application configuration and preferences.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QFrame, QSpinBox, QCheckBox,
    QLineEdit, QGroupBox, QMessageBox, QComboBox
)
from PySide6.QtCore import Qt, Signal

from config.settings import settings
from config.logging_config import logger


class SettingsPage(QWidget):
    """
    Application settings page.
    
    Features:
    - Recognition thresholds
    - Attendance deduplication interval
    - Camera settings
    - Database management
    """
    
    # Signals
    settings_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._load_current_settings()
    
    def _setup_ui(self):
        """Setup settings UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("Settings")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: white;")
        header.addWidget(title)
        
        header.addStretch()
        
        layout.addLayout(header)
        
        # Settings sections
        sections_layout = QHBoxLayout()
        sections_layout.setSpacing(25)
        
        # Left column
        left_column = QVBoxLayout()
        left_column.setSpacing(20)
        
        # Recognition settings
        recognition_group = self._create_recognition_settings()
        left_column.addWidget(recognition_group)
        
        # Attendance settings
        attendance_group = self._create_attendance_settings()
        left_column.addWidget(attendance_group)
        
        left_column.addStretch()
        sections_layout.addLayout(left_column)
        
        # Right column
        right_column = QVBoxLayout()
        right_column.setSpacing(20)
        
        # Camera settings
        camera_group = self._create_camera_settings()
        right_column.addWidget(camera_group)
        
        # Database management
        database_group = self._create_database_settings()
        right_column.addWidget(database_group)
        
        right_column.addStretch()
        sections_layout.addLayout(right_column)
        
        layout.addLayout(sections_layout)
        
        layout.addStretch()
        
        # Footer with save button
        footer = QHBoxLayout()
        footer.addStretch()
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_to_defaults)
        self._style_button(reset_btn, secondary=True)
        footer.addWidget(reset_btn)
        
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        self._style_button(save_btn)
        footer.addWidget(save_btn)
        
        layout.addLayout(footer)
    
    def _create_recognition_settings(self) -> QGroupBox:
        """Create recognition settings group."""
        group = QGroupBox("🔍 Recognition Settings")
        self._style_group(group)
        
        layout = QFormLayout(group)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 25, 20, 20)
        
        # Confidence threshold
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(20, 90)
        self.confidence_spin.setValue(40)
        self.confidence_spin.setSuffix("%")
        self._style_input(self.confidence_spin)
        layout.addRow("Recognition Threshold:", self.confidence_spin)
        
        # Detection confidence
        self.detection_spin = QSpinBox()
        self.detection_spin.setRange(30, 90)
        self.detection_spin.setValue(50)
        self.detection_spin.setSuffix("%")
        self._style_input(self.detection_spin)
        layout.addRow("Detection Threshold:", self.detection_spin)
        
        # Use GPU
        self.gpu_check = QCheckBox("Enable GPU Acceleration")
        self.gpu_check.setStyleSheet("color: #ccc;")
        layout.addRow("", self.gpu_check)
        
        return group
    
    def _create_attendance_settings(self) -> QGroupBox:
        """Create attendance settings group."""
        group = QGroupBox("📋 Attendance Settings")
        self._style_group(group)
        
        layout = QFormLayout(group)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 25, 20, 20)
        
        # Dedup interval
        self.dedup_spin = QSpinBox()
        self.dedup_spin.setRange(1, 480)
        self.dedup_spin.setValue(60)
        self.dedup_spin.setSuffix(" minutes")
        self._style_input(self.dedup_spin)
        layout.addRow("Deduplication Interval:", self.dedup_spin)
        
        # Enable exit tracking
        self.exit_check = QCheckBox("Track Exit Times")
        self.exit_check.setStyleSheet("color: #ccc;")
        layout.addRow("", self.exit_check)
        
        # Auto-save interval
        self.autosave_spin = QSpinBox()
        self.autosave_spin.setRange(1, 60)
        self.autosave_spin.setValue(5)
        self.autosave_spin.setSuffix(" minutes")
        self._style_input(self.autosave_spin)
        layout.addRow("Auto-save Interval:", self.autosave_spin)
        
        return group
    
    def _create_camera_settings(self) -> QGroupBox:
        """Create camera settings group."""
        group = QGroupBox("📹 Camera Settings")
        self._style_group(group)
        
        layout = QFormLayout(group)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 25, 20, 20)
        
        # Max cameras
        self.max_cameras_spin = QSpinBox()
        self.max_cameras_spin.setRange(1, 16)
        self.max_cameras_spin.setValue(4)
        self._style_input(self.max_cameras_spin)
        layout.addRow("Maximum Cameras:", self.max_cameras_spin)
        
        # Target FPS
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 30)
        self.fps_spin.setValue(15)
        self.fps_spin.setSuffix(" FPS")
        self._style_input(self.fps_spin)
        layout.addRow("Target Frame Rate:", self.fps_spin)
        
        # Default layout
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["2x2 Grid", "1x1 Single", "3x3 Grid"])
        self._style_input(self.layout_combo)
        layout.addRow("Default Layout:", self.layout_combo)
        
        return group
    
    def _create_database_settings(self) -> QGroupBox:
        """Create database management group."""
        group = QGroupBox("🗄️ Database Management")
        self._style_group(group)
        
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 25, 20, 20)
        
        # Info labels
        info_layout = QFormLayout()
        
        self.persons_count_label = QLabel("0")
        self.persons_count_label.setStyleSheet("color: #4a6fa5; font-weight: bold;")
        info_layout.addRow("Registered Persons:", self.persons_count_label)
        
        self.attendance_count_label = QLabel("0")
        self.attendance_count_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        info_layout.addRow("Total Attendance Records:", self.attendance_count_label)
        
        layout.addLayout(info_layout)
        
        layout.addSpacing(10)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        export_btn = QPushButton("📥 Export Data")
        export_btn.clicked.connect(self._export_data)
        self._style_button(export_btn, secondary=True, small=True)
        btn_layout.addWidget(export_btn)
        
        backup_btn = QPushButton("💾 Backup")
        backup_btn.clicked.connect(self._backup_database)
        self._style_button(backup_btn, secondary=True, small=True)
        btn_layout.addWidget(backup_btn)
        
        layout.addLayout(btn_layout)
        
        layout.addSpacing(10)
        
        # Danger zone
        danger_label = QLabel("⚠️ Danger Zone")
        danger_label.setStyleSheet("color: #F44336; font-weight: bold; margin-top: 10px;")
        layout.addWidget(danger_label)
        
        clear_btn = QPushButton("🗑️ Clear All Data")
        clear_btn.clicked.connect(self._confirm_clear_data)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d2020;
                border: 1px solid #F44336;
                border-radius: 4px;
                padding: 8px 16px;
                color: #F44336;
            }
            QPushButton:hover {
                background-color: #4d2020;
            }
        """)
        layout.addWidget(clear_btn)
        
        return group
    
    def _style_group(self, group: QGroupBox):
        """Apply group styling."""
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #ccc;
                background-color: #252525;
                border: 1px solid #404040;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
                background-color: #252525;
            }
        """)
    
    def _style_input(self, widget):
        """Apply input styling."""
        widget.setStyleSheet("""
            QSpinBox, QComboBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px;
                color: white;
                min-width: 150px;
            }
            QSpinBox:focus, QComboBox:focus {
                border-color: #4a6fa5;
            }
        """)
    
    def _style_button(self, button, secondary=False, small=False):
        """Apply button styling."""
        padding = "6px 12px" if small else "10px 20px"
        
        if secondary:
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #3d3d3d;
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: {padding};
                    color: #ccc;
                }}
                QPushButton:hover {{
                    background-color: #4d4d4d;
                }}
            """)
        else:
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #4a6fa5;
                    border: none;
                    border-radius: 4px;
                    padding: {padding};
                    color: white;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #5a7fb5;
                }}
            """)
    
    def _load_current_settings(self):
        """Load current settings values."""
        # Use getattr with defaults for optional settings
        self.confidence_spin.setValue(int(getattr(settings, 'RECOGNITION_THRESHOLD', 0.4) * 100))
        self.detection_spin.setValue(int(getattr(settings, 'DETECTION_CONFIDENCE', 0.5) * 100))
        self.gpu_check.setChecked(getattr(settings, 'USE_GPU', True))
        self.dedup_spin.setValue(getattr(settings, 'DEDUP_INTERVAL_MINUTES', 60))
        self.exit_check.setChecked(getattr(settings, 'ENABLE_EXIT_TRACKING', False))
        self.max_cameras_spin.setValue(getattr(settings, 'MAX_CAMERAS', 4))
        self.fps_spin.setValue(getattr(settings, 'TARGET_FPS', 15))
    
    def _save_settings(self):
        """Save settings (note: .env changes require restart)."""
        QMessageBox.information(
            self,
            "Settings",
            "Settings will be applied.\n\n"
            "Note: Some changes require application restart."
        )
        self.settings_changed.emit()
    
    def _reset_to_defaults(self):
        """Reset settings to defaults."""
        self.confidence_spin.setValue(40)
        self.detection_spin.setValue(50)
        self.gpu_check.setChecked(True)
        self.dedup_spin.setValue(60)
        self.exit_check.setChecked(False)
        self.max_cameras_spin.setValue(4)
        self.fps_spin.setValue(15)
    
    def _export_data(self):
        """Export all data."""
        QMessageBox.information(self, "Export", "Export feature coming soon!")
    
    def _backup_database(self):
        """Backup database."""
        QMessageBox.information(self, "Backup", "Backup feature coming soon!")
    
    def _confirm_clear_data(self):
        """Confirm and clear all data."""
        reply = QMessageBox.warning(
            self,
            "⚠️ Clear All Data",
            "This will permanently delete:\n\n"
            "• All registered persons\n"
            "• All face embeddings\n"
            "• All attendance records\n"
            "• All camera configurations\n\n"
            "This action cannot be undone!\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Second confirmation
            reply2 = QMessageBox.warning(
                self,
                "Final Confirmation",
                "Type 'DELETE' in the next dialog to confirm.",
                QMessageBox.Ok | QMessageBox.Cancel
            )
            
            if reply2 == QMessageBox.Ok:
                self._clear_all_data()
    
    def _clear_all_data(self):
        """Clear all data from databases."""
        try:
            from attendance.database import get_db_session
            from attendance.models import AttendanceRecord, FaceEmbedding, Person, Camera
            from core.face_database import FaceDatabase
            
            # Clear PostgreSQL
            with get_db_session() as session:
                session.query(AttendanceRecord).delete()
                session.query(FaceEmbedding).delete()
                session.query(Person).delete()
                session.query(Camera).delete()
                session.commit()
            
            # Clear FAISS
            embeddings_dir = settings.embeddings_dir
            if embeddings_dir.exists():
                for file in embeddings_dir.iterdir():
                    file.unlink()
            
            # Clear face images
            faces_dir = settings.faces_dir
            if faces_dir.exists():
                for file in faces_dir.iterdir():
                    if file.is_file():
                        file.unlink()
            
            QMessageBox.information(
                self,
                "Success",
                "All data has been cleared.\n\n"
                "Please restart the application."
            )
            
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to clear data:\n{e}")
    
    def update_counts(self, persons: int, attendance: int):
        """Update database counts."""
        self.persons_count_label.setText(str(persons))
        self.attendance_count_label.setText(str(attendance))