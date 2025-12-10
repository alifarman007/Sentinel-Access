"""
Main Application Window

The primary GUI window with navigation and pages.
"""

import sys
from typing import Optional, Dict
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QLabel, QFrame,
    QSizePolicy, QMessageBox, QApplication, QStatusBar,
    QDialog, QFormLayout, QSpinBox
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QFont

from config.logging_config import logger, setup_logging
from config.settings import settings
from core.recognition_pipeline import RecognitionPipeline
from gui.threads.recognition_thread import RecognitionThread
from gui.pages.add_person_page import AddPersonPage
from gui.pages.camera_view_page import CameraViewPage
from gui.pages.attendance_page import AttendancePage

from gui.pages.dashboard_page import DashboardPage
from gui.pages.settings_page import SettingsPage

class SidebarButton(QPushButton):
    """Custom styled sidebar button."""
    
    def __init__(self, text: str, icon_text: str = "", parent=None):
        super().__init__(parent)
        self.setText(f" {icon_text}  {text}" if icon_text else text)
        self.setCheckable(True)
        self.setMinimumHeight(45)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 8px;
                padding: 10px 15px;
                text-align: left;
                font-size: 14px;
                color: #ccc;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:checked {
                background-color: #4a6fa5;
                color: white;
            }
        """)


class StatusDetailDialog(QDialog):
    """Dialog showing detailed system status."""
    
    def __init__(self, status_info: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Status")
        self.setFixedSize(400, 300)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("📊 System Status")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        layout.addWidget(title)
        
        # Status items
        form = QFormLayout()
        form.setSpacing(10)
        
        # GPU Status
        gpu_status = "✅ CUDA Enabled" if status_info.get('gpu') else "⚠️ CPU Only"
        gpu_label = QLabel(gpu_status)
        gpu_label.setStyleSheet(f"color: {'#4CAF50' if status_info.get('gpu') else '#FFC107'};")
        form.addRow("GPU:", gpu_label)
        
        # Persons count
        form.addRow("Registered Persons:", QLabel(str(status_info.get('persons', 0))))
        
        # Recognition time
        rec_time = status_info.get('recognition_time', 0)
        rec_label = QLabel(f"{rec_time:.1f} ms")
        rec_color = '#4CAF50' if rec_time < 100 else '#FFC107' if rec_time < 200 else '#F44336'
        rec_label.setStyleSheet(f"color: {rec_color};")
        form.addRow("Avg Recognition Time:", rec_label)
        
        # Queue size
        form.addRow("Processing Queue:", QLabel(str(status_info.get('queue_size', 0))))
        
        # Active cameras
        form.addRow("Active Cameras:", QLabel(str(status_info.get('cameras', 0))))
        
        layout.addLayout(form)
        layout.addStretch()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a6fa5;
                border: none;
                border-radius: 4px;
                padding: 10px 30px;
                color: white;
            }
            QPushButton:hover {
                background-color: #5a7fb5;
            }
        """)
        layout.addWidget(close_btn, 0, Qt.AlignCenter)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
            }
            QLabel {
                color: #ddd;
                font-size: 13px;
            }
        """)


class MainWindow(QMainWindow):
    """
    Main application window.
    
    Features:
    - Sidebar navigation
    - Stacked pages (Dashboard, Add Person, Camera View, Attendance)
    - Status bar with clickable items
    - Recognition thread management
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Face Re-ID Access Management System")
        self.setMinimumSize(1200, 800)
        
        # Status tracking
        self._status_info = {
            'gpu': settings.USE_GPU,
            'persons': 0,
            'recognition_time': 0,
            'queue_size': 0,
            'cameras': 0
        }
        
        # Initialize core components
        self._init_core()
        
        # Setup UI
        self._setup_ui()
        
        # Setup status bar
        self._setup_statusbar()
        
        # Connect signals
        self._connect_signals()
        
        # Start recognition thread
        self._start_recognition()
        
        # Status update timer
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(1000)  # Update every second
        
        logger.info("Main window initialized")
    
    def _init_core(self):
        """Initialize core components."""
        # Recognition pipeline (shared)
        logger.info("Initializing recognition pipeline...")
        self.pipeline = RecognitionPipeline(use_gpu=settings.USE_GPU)
        
        # Recognition thread
        self.recognition_thread = RecognitionThread(
            pipeline=self.pipeline,
            max_queue_size=10
        )
        
        # Update initial status
        self._status_info['persons'] = self.pipeline.database.get_count()
    
    def _setup_ui(self):
        """Setup the main UI."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar)
        
        # Content area
        content = self._create_content_area()
        main_layout.addWidget(content, 1)
        
        # Apply dark theme
        self._apply_theme()
    
    def _create_sidebar(self) -> QFrame:
        """Create the sidebar navigation."""
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-right: 1px solid #333;
            }
        """)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 15, 10, 15)
        layout.setSpacing(5)
        
        # Logo/Title
        title = QLabel("🏢 Face Re-ID")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: white;
            padding: 10px;
        """)
        layout.addWidget(title)
        
        subtitle = QLabel("Access Management System")
        subtitle.setStyleSheet("color: #888; font-size: 11px; padding-left: 10px;")
        layout.addWidget(subtitle)
        
        layout.addSpacing(20)
        
        # Navigation buttons
        self.nav_buttons: Dict[str, SidebarButton] = {}
        
        nav_items = [
            ("dashboard", "Dashboard", "🏠"),
            ("add_person", "Add Person", "👤"),
            ("camera_view", "Live Cameras", "📹"),
            ("attendance", "Attendance Log", "📋"),
        ]
        
        for page_id, text, icon in nav_items:
            btn = SidebarButton(text, icon)
            btn.clicked.connect(lambda checked, pid=page_id: self._navigate_to(pid))
            self.nav_buttons[page_id] = btn
            layout.addWidget(btn)
        
        layout.addStretch()
        
        # Settings button at bottom
        settings_btn = SidebarButton("Settings", "⚙️")
        settings_btn.clicked.connect(lambda: self._navigate_to("settings"))
        self.nav_buttons["settings"] = settings_btn
        layout.addWidget(settings_btn)
        
        # Select dashboard by default
        self.nav_buttons["dashboard"].setChecked(True)
        
        return sidebar
    
    def _create_content_area(self) -> QWidget:
        """Create the main content area."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Stacked widget for pages
        self.page_stack = QStackedWidget()
        layout.addWidget(self.page_stack)
        
        # Create pages
        self._create_pages()
        
        return content
    
    def _create_pages(self):
        """Create all application pages."""
        # Dashboard page
        dashboard = DashboardPage(database=self.pipeline.database)
        dashboard.navigate_to.connect(self._navigate_to)
        self.page_stack.addWidget(dashboard)
        self._pages = {"dashboard": dashboard}
        
        # Add Person page
        add_person = AddPersonPage(
            detector=self.pipeline.detector,
            recognizer=self.pipeline.recognizer,
            database=self.pipeline.database
        )
        add_person.person_registered.connect(self._on_person_registered)
        add_person.person_deleted.connect(self._on_person_deleted)
        self.page_stack.addWidget(add_person)
        self._pages["add_person"] = add_person
        
        # Camera View page
        camera_view = CameraViewPage(
            pipeline=self.pipeline,
            recognition_thread=self.recognition_thread
        )
        camera_view.attendance_recorded.connect(self._on_attendance_recorded)
        self.page_stack.addWidget(camera_view)
        self._pages["camera_view"] = camera_view
        
        # Attendance page
        attendance = AttendancePage()
        self.page_stack.addWidget(attendance)
        self._pages["attendance"] = attendance
        
        # Settings page
        settings_page = SettingsPage()
        settings_page.settings_changed.connect(self._on_settings_changed)
        self.page_stack.addWidget(settings_page)
        self._pages["settings"] = settings_page
    
    def _create_dashboard_page(self) -> QWidget:
        """Create the dashboard page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title = QLabel("Dashboard")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        layout.addWidget(title)
        
        layout.addSpacing(20)
        
        # Stats cards
        cards_layout = QHBoxLayout()
        
        # Persons card
        self.persons_card = self._create_stat_card(
            "👥 Registered Persons",
            str(self.pipeline.database.get_count()),
            "#4a6fa5"
        )
        cards_layout.addWidget(self.persons_card)
        
        # Today's attendance card
        self.attendance_card = self._create_stat_card(
            "📋 Today's Attendance",
            "0",
            "#4CAF50"
        )
        cards_layout.addWidget(self.attendance_card)
        
        # Cameras card
        self.cameras_card = self._create_stat_card(
            "📹 Active Cameras",
            "0",
            "#FF9800"
        )
        cards_layout.addWidget(self.cameras_card)
        
        layout.addLayout(cards_layout)
        
        layout.addSpacing(30)
        
        # Quick actions
        actions_label = QLabel("Quick Actions")
        actions_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #ccc;")
        layout.addWidget(actions_label)
        
        layout.addSpacing(10)
        
        actions_layout = QHBoxLayout()
        
        add_person_btn = QPushButton("➕ Add New Person")
        add_person_btn.clicked.connect(lambda: self._navigate_to("add_person"))
        add_person_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a6fa5;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                color: white;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5a7fb5;
            }
        """)
        actions_layout.addWidget(add_person_btn)
        
        view_cameras_btn = QPushButton("📹 View Live Cameras")
        view_cameras_btn.clicked.connect(lambda: self._navigate_to("camera_view"))
        view_cameras_btn.setStyleSheet(add_person_btn.styleSheet())
        actions_layout.addWidget(view_cameras_btn)
        
        view_attendance_btn = QPushButton("📋 View Attendance")
        view_attendance_btn.clicked.connect(lambda: self._navigate_to("attendance"))
        view_attendance_btn.setStyleSheet(add_person_btn.styleSheet())
        actions_layout.addWidget(view_attendance_btn)
        
        actions_layout.addStretch()
        
        layout.addLayout(actions_layout)
        
        layout.addStretch()
        
        return page
    
    def _create_stat_card(self, title: str, value: str, color: str) -> QFrame:
        """Create a statistics card widget."""
        card = QFrame()
        card.setFixedSize(200, 120)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 10px;
                border-left: 4px solid {color};
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 15, 15, 15)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(title_label)
        
        value_label = QLabel(value)
        value_label.setObjectName("value")
        value_label.setStyleSheet(f"color: {color}; font-size: 32px; font-weight: bold;")
        layout.addWidget(value_label)
        
        layout.addStretch()
        
        return card
    
    def _create_placeholder_page(self, title: str, message: str) -> QWidget:
        """Create a placeholder page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        layout.addWidget(title_label)
        
        layout.addSpacing(20)
        
        msg_label = QLabel(message)
        msg_label.setStyleSheet("font-size: 14px; color: #aaa;")
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)
        
        layout.addStretch()
        
        return page
    
    def _setup_statusbar(self):
        """Setup the status bar with clickable items."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Make status bar clickable
        self.statusbar.setStyleSheet("""
            QStatusBar {
                background-color: #252525;
                border-top: 1px solid #333;
            }
            QStatusBar::item {
                border: none;
            }
        """)
        
        # GPU status (clickable)
        gpu_status = "✓ GPU" if settings.USE_GPU else "CPU"
        self._gpu_label = QPushButton(f"🖥️ {gpu_status}")
        self._gpu_label.setFlat(True)
        self._gpu_label.setCursor(Qt.PointingHandCursor)
        self._gpu_label.clicked.connect(self._show_status_dialog)
        self._gpu_label.setStyleSheet("""
            QPushButton {
                color: #4CAF50;
                border: none;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                border-radius: 4px;
            }
        """)
        self.statusbar.addPermanentWidget(self._gpu_label)
        
        # Database status (clickable)
        self._db_label = QPushButton(f"👥 {self.pipeline.database.get_count()} persons")
        self._db_label.setFlat(True)
        self._db_label.setCursor(Qt.PointingHandCursor)
        self._db_label.clicked.connect(self._show_status_dialog)
        self._db_label.setStyleSheet("""
            QPushButton {
                color: #ccc;
                border: none;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                border-radius: 4px;
            }
        """)
        self.statusbar.addPermanentWidget(self._db_label)
        
        # Recognition stats (clickable)
        self._recognition_label = QPushButton("⏱️ -- ms")
        self._recognition_label.setFlat(True)
        self._recognition_label.setCursor(Qt.PointingHandCursor)
        self._recognition_label.clicked.connect(self._show_status_dialog)
        self._recognition_label.setStyleSheet(self._db_label.styleSheet())
        self.statusbar.addPermanentWidget(self._recognition_label)
        
        self.statusbar.showMessage("Ready")
    
    def _show_status_dialog(self):
        """Show detailed status dialog."""
        dialog = StatusDetailDialog(self._status_info, self)
        dialog.exec()
    
    def _connect_signals(self):
        """Connect signals and slots."""
        self.recognition_thread.stats_updated.connect(self._on_recognition_stats)
        self.recognition_thread.person_recognized.connect(self._on_person_recognized_status)
    
    def _start_recognition(self):
        """Start the recognition thread."""
        self.recognition_thread.start()
        logger.info("Recognition thread started")
    
    def _update_status(self):
        """Update status periodically."""
        # Update camera count from camera view page
        camera_page = self._pages.get("camera_view")
        if camera_page:
            self._status_info['cameras'] = len(camera_page._camera_threads)
    
    @Slot(str)
    def _navigate_to(self, page_id: str):
        """Navigate to a page."""
        # Update button states
        for pid, btn in self.nav_buttons.items():
            btn.setChecked(pid == page_id)
        
        # Switch page
        if page_id in self._pages:
            self.page_stack.setCurrentWidget(self._pages[page_id])
            self.statusbar.showMessage(f"Page: {page_id.replace('_', ' ').title()}")
            
            # Refresh attendance page when navigating to it
            if page_id == "attendance":
                self._pages["attendance"].refresh()
    
    @Slot(float, int)
    def _on_recognition_stats(self, process_time_ms: float, queue_size: int):
        """Update recognition stats in status bar."""
        self._status_info['recognition_time'] = process_time_ms
        self._status_info['queue_size'] = queue_size
        self._recognition_label.setText(f"⏱️ {process_time_ms:.0f}ms | Q:{queue_size}")
    
    @Slot(str, str, str, float)
    def _on_person_recognized_status(self, camera_id: str, person_id: str, name: str, confidence: float):
        """Handle person recognized event for status bar."""
        self.statusbar.showMessage(f"Recognized: {name} ({confidence:.2f})", 3000)
    
    @Slot(str, str)
    def _on_person_registered(self, person_id: str, name: str):
        """Handle new person registered."""
        self.update_database_count()
        self.statusbar.showMessage(f"Registered: {name} ({person_id})", 5000)
    
    @Slot(str)
    def _on_person_deleted(self, person_id: str):
        """Handle person deleted."""
        self.update_database_count()
        self.statusbar.showMessage(f"Deleted person: {person_id}", 5000)
    
    @Slot(str, str, str)
    def _on_attendance_recorded(self, person_id: str, name: str, camera_id: str):
        """Handle attendance recorded."""
        self.statusbar.showMessage(f"Attendance: {name} at {camera_id}", 3000)
    
    @Slot()
    def _on_settings_changed(self):
        """Handle settings changed."""
        self.statusbar.showMessage("Settings updated", 3000)

    def update_database_count(self):
        """Update the database count in status bar and dashboard."""
        count = self.pipeline.database.get_count()
        self._status_info['persons'] = count
        self._db_label.setText(f"👥 {count} persons")
        
        # Update dashboard card
        if hasattr(self, 'persons_card'):
            value_label = self.persons_card.findChild(QLabel, "value")
            if value_label:
                value_label.setText(str(count))
    
    def _apply_theme(self):
        """Apply dark theme to application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ddd;
            }
            QLabel {
                color: #ddd;
            }
        """)
    
    def closeEvent(self, event):
        """Handle window close."""
        # Stop recognition thread
        self.recognition_thread.stop()
        
        # Stop camera threads
        camera_page = self._pages.get("camera_view")
        if camera_page:
            camera_page.stop_all_cameras()
        
        # Save database
        self.pipeline.database.save()
        
        logger.info("Application closed")
        event.accept()


def main():
    """Application entry point."""
    setup_logging(log_level="INFO", log_to_file=True)
    
    logger.info("=" * 50)
    logger.info("Starting Face Re-ID Access Management System")
    logger.info("=" * 50)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()