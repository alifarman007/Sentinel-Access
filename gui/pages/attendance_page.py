"""
Attendance Log Page

Displays attendance records with filtering and export.
"""

from datetime import date, datetime, timedelta
from typing import List, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QLabel, QPushButton,
    QLineEdit, QDateEdit, QComboBox, QFrame, QMessageBox,
    QAbstractItemView, QFileDialog, QGroupBox
)
from PySide6.QtCore import Qt, Signal, Slot, QDate
from PySide6.QtGui import QColor

from config.logging_config import logger
from attendance.attendance_service import AttendanceService, PersonService


class AttendancePage(QWidget):
    """
    Attendance log viewing page.
    
    Features:
    - Date range filtering
    - Search by name/ID
    - Record type filter (entry/exit/all)
    - Export to CSV
    - Auto-refresh
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.attendance_service = AttendanceService()
        self._all_records: List[dict] = []
        
        self._setup_ui()
        self._load_data()
    
    def _setup_ui(self):
        """Setup page UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("Attendance Log")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Refresh button
        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.clicked.connect(self._load_data)
        self._style_button(refresh_btn, secondary=True)
        header.addWidget(refresh_btn)
        
        # Export button
        export_btn = QPushButton("📥 Export CSV")
        export_btn.clicked.connect(self._export_csv)
        self._style_button(export_btn)
        header.addWidget(export_btn)
        
        layout.addLayout(header)
        layout.addSpacing(15)
        
        # Filters
        filters = self._create_filters()
        layout.addWidget(filters)
        
        layout.addSpacing(10)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Date", "Time", "Person ID", "Name", "Type", "Confidence", "Camera"
        ])
        
        # Table styling
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 6px;
                gridline-color: #404040;
            }
            QTableWidget::item {
                padding: 8px;
                color: #ddd;
            }
            QTableWidget::item:selected {
                background-color: #4a6fa5;
            }
            QHeaderView::section {
                background-color: #353535;
                color: #ccc;
                padding: 10px;
                border: none;
                border-bottom: 2px solid #4a6fa5;
                font-weight: bold;
                font-size: 13px;
            }
        """)
        
        # Table settings
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(True)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        
        # Column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Date
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Time
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Person ID
        header.setSectionResizeMode(3, QHeaderView.Stretch)           # Name
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Type
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Confidence
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Camera
        
        self.table.verticalHeader().setDefaultSectionSize(40)
        
        layout.addWidget(self.table)
        
        # Footer with stats
        footer = QHBoxLayout()
        
        self.stats_label = QLabel("0 records")
        self.stats_label.setStyleSheet("color: #888; font-size: 12px;")
        footer.addWidget(self.stats_label)
        
        footer.addStretch()
        
        self.last_update_label = QLabel("")
        self.last_update_label.setStyleSheet("color: #666; font-size: 11px;")
        footer.addWidget(self.last_update_label)
        
        layout.addLayout(footer)
    
    def _create_filters(self) -> QWidget:
        """Create filter controls."""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-radius: 8px;
                padding: 5px;
            }
        """)
        
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Date range
        layout.addWidget(QLabel("From:"))
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate())
        self.date_from.setCalendarPopup(True)
        self.date_from.dateChanged.connect(self._apply_filters)
        self._style_date_edit(self.date_from)
        layout.addWidget(self.date_from)
        
        layout.addWidget(QLabel("To:"))
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        self.date_to.dateChanged.connect(self._apply_filters)
        self._style_date_edit(self.date_to)
        layout.addWidget(self.date_to)
        
        layout.addSpacing(20)
        
        # Quick date buttons
        today_btn = QPushButton("Today")
        today_btn.clicked.connect(self._set_today)
        self._style_button(today_btn, small=True)
        layout.addWidget(today_btn)
        
        week_btn = QPushButton("This Week")
        week_btn.clicked.connect(self._set_this_week)
        self._style_button(week_btn, small=True)
        layout.addWidget(week_btn)
        
        month_btn = QPushButton("This Month")
        month_btn.clicked.connect(self._set_this_month)
        self._style_button(month_btn, small=True)
        layout.addWidget(month_btn)
        
        layout.addSpacing(20)
        
        # Type filter
        layout.addWidget(QLabel("Type:"))
        self.type_filter = QComboBox()
        self.type_filter.addItems(["All", "Entry", "Exit"])
        self.type_filter.currentIndexChanged.connect(self._apply_filters)
        self._style_combo(self.type_filter)
        layout.addWidget(self.type_filter)
        
        layout.addSpacing(20)
        
        # Search
        layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Name or ID...")
        self.search_input.textChanged.connect(self._apply_filters)
        self.search_input.setMaximumWidth(200)
        self._style_input(self.search_input)
        layout.addWidget(self.search_input)
        
        layout.addStretch()
        
        # Clear filters
        clear_btn = QPushButton("Clear Filters")
        clear_btn.clicked.connect(self._clear_filters)
        self._style_button(clear_btn, secondary=True, small=True)
        layout.addWidget(clear_btn)
        
        return frame
    
    def _style_button(self, button, secondary=False, small=False):
        """Apply button styling."""
        padding = "6px 12px" if small else "8px 16px"
        
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
                }}
                QPushButton:hover {{
                    background-color: #5a7fb5;
                }}
            """)
    
    def _style_input(self, widget):
        """Style input widget."""
        widget.setStyleSheet("""
            QLineEdit {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 10px;
                color: white;
            }
            QLineEdit:focus {
                border-color: #4a6fa5;
            }
        """)
    
    def _style_combo(self, combo):
        """Style combo box."""
        combo.setStyleSheet("""
            QComboBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 10px;
                color: white;
                min-width: 80px;
            }
        """)
    
    def _style_date_edit(self, date_edit):
        """Style date edit."""
        date_edit.setStyleSheet("""
            QDateEdit {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 10px;
                color: white;
            }
            QDateEdit::drop-down {
                border: none;
                width: 20px;
            }
        """)
    
    def _set_today(self):
        """Set date range to today."""
        today = QDate.currentDate()
        self.date_from.setDate(today)
        self.date_to.setDate(today)
    
    def _set_this_week(self):
        """Set date range to this week."""
        today = QDate.currentDate()
        start_of_week = today.addDays(-today.dayOfWeek() + 1)
        self.date_from.setDate(start_of_week)
        self.date_to.setDate(today)
    
    def _set_this_month(self):
        """Set date range to this month."""
        today = QDate.currentDate()
        start_of_month = QDate(today.year(), today.month(), 1)
        self.date_from.setDate(start_of_month)
        self.date_to.setDate(today)
    
    def _clear_filters(self):
        """Clear all filters."""
        self.date_from.setDate(QDate.currentDate())
        self.date_to.setDate(QDate.currentDate())
        self.type_filter.setCurrentIndex(0)
        self.search_input.clear()
    
    def _load_data(self):
        """Load attendance data from database."""
        try:
            start_date = self.date_from.date().toPython()
            end_date = self.date_to.date().toPython()
            
            self._all_records = self.attendance_service.get_attendance_by_date(
                start_date, end_date
            )
            
            self._apply_filters()
            
            self.last_update_label.setText(
                f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load attendance: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load attendance:\n{e}")
    
    def _apply_filters(self):
        """Apply filters to loaded data."""
        filtered = self._all_records.copy()
        
        # Type filter
        type_filter = self.type_filter.currentText().lower()
        if type_filter != "all":
            filtered = [r for r in filtered if r.get('record_type') == type_filter]
        
        # Search filter
        search_text = self.search_input.text().strip().lower()
        if search_text:
            filtered = [
                r for r in filtered
                if search_text in r.get('name', '').lower()
                or search_text in r.get('person_id', '').lower()
            ]
        
        self._populate_table(filtered)
    
    def _populate_table(self, records: List[dict]):
        """Populate table with records."""
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        
        for record in records:
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # Date
            date_str = record.get('date', '')
            self.table.setItem(row, 0, QTableWidgetItem(date_str))
            
            # Time
            time_str = record.get('time', '')
            self.table.setItem(row, 1, QTableWidgetItem(time_str))
            
            # Person ID
            self.table.setItem(row, 2, QTableWidgetItem(record.get('person_id', '')))
            
            # Name
            self.table.setItem(row, 3, QTableWidgetItem(record.get('name', '')))
            
            # Type with color
            record_type = record.get('record_type', '')
            type_item = QTableWidgetItem(record_type.upper())
            if record_type == 'entry':
                type_item.setForeground(QColor("#4CAF50"))  # Green
            else:
                type_item.setForeground(QColor("#FF9800"))  # Orange
            self.table.setItem(row, 4, type_item)
            
            # Confidence
            confidence = record.get('confidence', 0)
            conf_str = f"{confidence:.2f}" if confidence else "-"
            self.table.setItem(row, 5, QTableWidgetItem(conf_str))
            
            # Camera
            self.table.setItem(row, 6, QTableWidgetItem(record.get('camera_id', '-')))
        
        self.table.setSortingEnabled(True)
        
        # Update stats
        self.stats_label.setText(f"{len(records)} records")
    
    def _export_csv(self):
        """Export current view to CSV."""
        if self.table.rowCount() == 0:
            QMessageBox.information(self, "Info", "No records to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Attendance",
            f"attendance_{date.today().strftime('%Y%m%d')}.csv",
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Header
                headers = []
                for col in range(self.table.columnCount()):
                    headers.append(self.table.horizontalHeaderItem(col).text())
                f.write(','.join(headers) + '\n')
                
                # Data
                for row in range(self.table.rowCount()):
                    row_data = []
                    for col in range(self.table.columnCount()):
                        item = self.table.item(row, col)
                        text = item.text() if item else ''
                        # Escape commas and quotes
                        if ',' in text or '"' in text:
                            text = '"' + text.replace('"', '""') + '"'
                        row_data.append(text)
                    f.write(','.join(row_data) + '\n')
            
            QMessageBox.information(
                self, "Success",
                f"Exported {self.table.rowCount()} records to:\n{file_path}"
            )
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}")
    
    def refresh(self):
        """Refresh the attendance data."""
        self._load_data()
    
    def showEvent(self, event):
        """Handle page shown."""
        super().showEvent(event)
        self._load_data()