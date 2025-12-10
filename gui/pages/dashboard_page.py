"""
Dashboard Page

Main dashboard with statistics and quick actions.
"""

from datetime import date, datetime
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer

from config.logging_config import logger
from core.face_database import FaceDatabase
from attendance.attendance_service import AttendanceService


class StatCard(QFrame):
    """Statistics card widget."""
    
    clicked = Signal()
    
    def __init__(
        self,
        title: str,
        value: str,
        icon: str,
        color: str,
        parent=None
    ):
        super().__init__(parent)
        
        self.color = color
        self._setup_ui(title, value, icon)
        self.setCursor(Qt.PointingHandCursor)
    
    def _setup_ui(self, title: str, value: str, icon: str):
        """Setup card UI."""
        self.setFixedSize(220, 130)
        self.setStyleSheet(f"""
            StatCard {{
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 12px;
                border-left: 5px solid {self.color};
            }}
            StatCard:hover {{
                background-color: #353535;
                border-color: #505050;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 15, 18, 15)
        layout.setSpacing(8)
        
        # Header with icon
        header = QHBoxLayout()
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 24px;")
        header.addWidget(icon_label)
        
        header.addStretch()
        
        layout.addLayout(header)
        
        # Value
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"""
            font-size: 36px;
            font-weight: bold;
            color: {self.color};
        """)
        layout.addWidget(self.value_label)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #888; font-size: 13px;")
        layout.addWidget(title_label)
    
    def set_value(self, value: str):
        """Update the displayed value."""
        self.value_label.setText(value)
    
    def mousePressEvent(self, event):
        """Handle click."""
        self.clicked.emit()
        super().mousePressEvent(event)


class RecentActivityItem(QFrame):
    """Recent activity list item."""
    
    def __init__(self, icon: str, text: str, time: str, parent=None):
        super().__init__(parent)
        
        self.setStyleSheet("""
            RecentActivityItem {
                background-color: #2d2d2d;
                border-radius: 8px;
                padding: 5px;
            }
            RecentActivityItem:hover {
                background-color: #353535;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        
        # Icon
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 20px;")
        icon_label.setFixedWidth(30)
        layout.addWidget(icon_label)
        
        # Text
        text_label = QLabel(text)
        text_label.setStyleSheet("color: #ddd; font-size: 13px;")
        layout.addWidget(text_label, 1)
        
        # Time
        time_label = QLabel(time)
        time_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(time_label)


class DashboardPage(QWidget):
    """
    Dashboard page with live statistics.
    
    Features:
    - Real-time stats cards
    - Recent activity feed
    - Quick action buttons
    """
    
    # Signals for navigation
    navigate_to = Signal(str)
    
    def __init__(
        self,
        database: FaceDatabase,
        parent=None
    ):
        super().__init__(parent)
        
        self.database = database
        self.attendance_service = AttendanceService()
        
        self._setup_ui()
        
        # Refresh timer
        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._refresh_stats)
        self._refresh_timer.start(5000)  # Refresh every 5 seconds
    
    def _setup_ui(self):
        """Setup dashboard UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("Dashboard")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: white;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Current date/time
        self.datetime_label = QLabel()
        self.datetime_label.setStyleSheet("color: #888; font-size: 14px;")
        self._update_datetime()
        header.addWidget(self.datetime_label)
        
        layout.addLayout(header)
        
        # Stats cards
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(20)
        
        # Registered persons
        self.persons_card = StatCard(
            "Registered Persons",
            "0",
            "👥",
            "#4a6fa5"
        )
        self.persons_card.clicked.connect(lambda: self.navigate_to.emit("add_person"))
        cards_layout.addWidget(self.persons_card)
        
        # Today's attendance
        self.today_card = StatCard(
            "Today's Check-ins",
            "0",
            "✅",
            "#4CAF50"
        )
        self.today_card.clicked.connect(lambda: self.navigate_to.emit("attendance"))
        cards_layout.addWidget(self.today_card)
        
        # Active cameras
        self.cameras_card = StatCard(
            "Active Cameras",
            "0",
            "📹",
            "#FF9800"
        )
        self.cameras_card.clicked.connect(lambda: self.navigate_to.emit("camera_view"))
        cards_layout.addWidget(self.cameras_card)
        
        # Unique persons today
        self.unique_card = StatCard(
            "Unique Today",
            "0",
            "👤",
            "#9C27B0"
        )
        self.unique_card.clicked.connect(lambda: self.navigate_to.emit("attendance"))
        cards_layout.addWidget(self.unique_card)
        
        cards_layout.addStretch()
        layout.addLayout(cards_layout)
        
        # Main content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(25)
        
        # Left: Recent Activity
        activity_section = self._create_activity_section()
        content_layout.addWidget(activity_section, 2)
        
        # Right: Quick Actions
        actions_section = self._create_actions_section()
        content_layout.addWidget(actions_section, 1)
        
        layout.addLayout(content_layout)
        
        layout.addStretch()
        
        # Initial data load
        self._refresh_stats()
    
    def _create_activity_section(self) -> QFrame:
        """Create recent activity section."""
        section = QFrame()
        section.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-radius: 12px;
            }
        """)
        
        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("📋 Recent Activity")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ccc;")
        header.addWidget(title)
        header.addStretch()
        
        view_all_btn = QPushButton("View All →")
        view_all_btn.clicked.connect(lambda: self.navigate_to.emit("attendance"))
        view_all_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #4a6fa5;
                font-size: 12px;
            }
            QPushButton:hover {
                color: #6a9fd5;
            }
        """)
        header.addWidget(view_all_btn)
        
        layout.addLayout(header)
        
        # Activity list
        self.activity_list = QVBoxLayout()
        self.activity_list.setSpacing(8)
        
        # Placeholder
        placeholder = QLabel("No recent activity")
        placeholder.setStyleSheet("color: #666; padding: 20px;")
        placeholder.setAlignment(Qt.AlignCenter)
        self.activity_list.addWidget(placeholder)
        
        layout.addLayout(self.activity_list)
        layout.addStretch()
        
        return section
    
    def _create_actions_section(self) -> QFrame:
        """Create quick actions section."""
        section = QFrame()
        section.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-radius: 12px;
            }
        """)
        
        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        title = QLabel("⚡ Quick Actions")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ccc;")
        layout.addWidget(title)
        
        # Action buttons
        actions = [
            ("➕ Add New Person", "add_person", "#4a6fa5"),
            ("📹 View Live Cameras", "camera_view", "#FF9800"),
            ("📋 Attendance Report", "attendance", "#4CAF50"),
            ("⚙️ Settings", "settings", "#9E9E9E"),
        ]
        
        for text, page_id, color in actions:
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, p=page_id: self.navigate_to.emit(p))
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #3d3d3d;
                    border: 1px solid #505050;
                    border-left: 4px solid {color};
                    border-radius: 8px;
                    padding: 15px 20px;
                    color: #ddd;
                    font-size: 14px;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background-color: #454545;
                }}
            """)
            layout.addWidget(btn)
        
        layout.addStretch()
        
        return section
    
    def _update_datetime(self):
        """Update datetime display."""
        now = datetime.now()
        self.datetime_label.setText(now.strftime("%A, %B %d, %Y  •  %H:%M"))
    
    def _refresh_stats(self):
        """Refresh all statistics."""
        try:
            # Update datetime
            self._update_datetime()
            
            # Persons count
            persons_count = self.database.get_count()
            self.persons_card.set_value(str(persons_count))
            
            # Today's attendance
            today_records = self.attendance_service.get_today_attendance()
            self.today_card.set_value(str(len(today_records)))
            
            # Unique persons today
            unique_persons = set(r.get('person_id') for r in today_records)
            self.unique_card.set_value(str(len(unique_persons)))
            
            # Update recent activity
            self._update_activity(today_records[:5])
            
        except Exception as e:
            logger.error(f"Failed to refresh stats: {e}")
    
    def _update_activity(self, records: list):
        """Update recent activity list."""
        # Clear existing items
        while self.activity_list.count():
            item = self.activity_list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not records:
            placeholder = QLabel("No recent activity")
            placeholder.setStyleSheet("color: #666; padding: 20px;")
            placeholder.setAlignment(Qt.AlignCenter)
            self.activity_list.addWidget(placeholder)
            return
        
        for record in records:
            icon = "🟢" if record.get('record_type') == 'entry' else "🟠"
            name = record.get('name', 'Unknown')
            record_type = record.get('record_type', '').upper()
            time_str = record.get('time', '')
            
            text = f"{name} - {record_type}"
            
            item = RecentActivityItem(icon, text, time_str)
            self.activity_list.addWidget(item)
    
    def set_camera_count(self, count: int):
        """Update camera count from external source."""
        self.cameras_card.set_value(str(count))
    
    def refresh(self):
        """Manual refresh."""
        self._refresh_stats()
    
    def showEvent(self, event):
        """Handle page shown."""
        super().showEvent(event)
        self._refresh_stats()