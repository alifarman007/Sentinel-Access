"""
Registered Persons Table Widget

Displays list of registered persons with photos.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QLabel, QPushButton,
    QLineEdit, QAbstractItemView, QMenu, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QPixmap, QImage, QIcon, QAction

from config.settings import settings
from config.logging_config import logger



class PersonsTable(QWidget):
    """
    Table widget showing registered persons.
    
    Features:
    - Photo thumbnails
    - Search/filter
    - Delete action
    - Selection
    """
    
    # Signals
    person_selected = Signal(str)  # person_id
    person_deleted = Signal(str)   # person_id
    refresh_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._persons_data: List[dict] = []
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("Registered Persons")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Search box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search by name or ID...")
        self.search_input.setMaximumWidth(200)
        self.search_input.textChanged.connect(self._filter_table)
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 10px;
                color: white;
            }
        """)
        header_layout.addWidget(self.search_input)
        
        # Refresh button
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedSize(32, 32)
        refresh_btn.setToolTip("Refresh list")
        refresh_btn.clicked.connect(self.refresh_requested.emit)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                color: white;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        layout.addSpacing(10)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Photo", "Person ID", "Name", "Department", "Created"
        ])
        
        # Table styling
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 4px;
                gridline-color: #404040;
            }
            QTableWidget::item {
                padding: 5px;
                color: #ddd;
            }
            QTableWidget::item:selected {
                background-color: #4a6fa5;
            }
            QHeaderView::section {
                background-color: #353535;
                color: #ccc;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #404040;
                font-weight: bold;
            }
        """)
        
        # Table settings
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(True)
        
        # Column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)  # Photo
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(2, QHeaderView.Stretch)  # Name
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Dept
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Created
        
        self.table.setColumnWidth(0, 60)  # Photo column
        self.table.setIconSize(QSize(50, 50))
        
        # Row height
        self.table.verticalHeader().setDefaultSectionSize(55)
        
        # Context menu
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        
        # Selection signal
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        
        layout.addWidget(self.table)
        
        # Count label
        self.count_label = QLabel("0 persons registered")
        self.count_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self.count_label)
    
    def set_data(self, persons: List[dict]):
        """
        Set table data.
        
        Args:
            persons: List of person dicts with keys:
                     id, person_id, name, department, created_at, image_path
        """
        self._persons_data = persons
        self._populate_table(persons)
    
    def _populate_table(self, persons: List[dict]):
        """Populate table with person data."""
        self.table.setRowCount(0)
        
        for person in persons:
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # Photo
            photo_label = QLabel()
            photo_label.setAlignment(Qt.AlignCenter)
            pixmap = self._load_photo(person.get('image_path'))
            if pixmap:
                photo_label.setPixmap(pixmap.scaled(
                    50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
            else:
                photo_label.setText("👤")
                photo_label.setStyleSheet("font-size: 24px;")
            self.table.setCellWidget(row, 0, photo_label)
            
            # Person ID
            id_item = QTableWidgetItem(person.get('person_id', ''))
            id_item.setData(Qt.UserRole, person.get('id'))  # Store UUID
            self.table.setItem(row, 1, id_item)
            
            # Name
            self.table.setItem(row, 2, QTableWidgetItem(person.get('name', '')))
            
            # Department
            dept = person.get('department') or '-'
            self.table.setItem(row, 3, QTableWidgetItem(dept))
            
            # Created date
            created = person.get('created_at')
            if created:
                if hasattr(created, 'strftime'):
                    created_str = created.strftime('%Y-%m-%d')
                else:
                    created_str = str(created)[:10]
            else:
                created_str = '-'
            self.table.setItem(row, 4, QTableWidgetItem(created_str))
        
        self.count_label.setText(f"{len(persons)} persons registered")
    
    def _load_photo(self, image_path: Optional[str]) -> Optional[QPixmap]:
        """Load photo from path."""
        if not image_path:
            return None
        
        try:
            path = Path(image_path)
            if not path.exists():
                # Try relative to faces dir
                path = settings.faces_dir / image_path
            
            if path.exists():
                pixmap = QPixmap(str(path))
                if not pixmap.isNull():
                    return pixmap
        except Exception as e:
            logger.debug(f"Failed to load photo: {e}")
        
        return None
    
    def _filter_table(self, text: str):
        """Filter table by search text."""
        text = text.lower()
        
        for row in range(self.table.rowCount()):
            show = False
            
            if not text:
                show = True
            else:
                # Check person_id and name columns
                for col in [1, 2]:
                    item = self.table.item(row, col)
                    if item and text in item.text().lower():
                        show = True
                        break
            
            self.table.setRowHidden(row, not show)
    
    def _show_context_menu(self, pos):
        """Show right-click context menu."""
        item = self.table.itemAt(pos)
        if not item:
            return
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #3d3d3d;
                border: 1px solid #555;
                color: white;
            }
            QMenu::item:selected {
                background-color: #4a6fa5;
            }
        """)
        
        # Delete action
        delete_action = QAction("🗑️ Delete Person", self)
        delete_action.triggered.connect(lambda: self._delete_person(item.row()))
        menu.addAction(delete_action)
        
        menu.exec_(self.table.viewport().mapToGlobal(pos))
    
    def _delete_person(self, row: int):
        """Handle delete person action."""
        id_item = self.table.item(row, 1)
        name_item = self.table.item(row, 2)
        
        if not id_item:
            return
        
        person_id = id_item.text()
        name = name_item.text() if name_item else person_id
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete '{name}' ({person_id})?\n\n"
            "This will remove all their data including face embeddings and attendance records.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.person_deleted.emit(person_id)
    
    def _on_selection_changed(self):
        """Handle selection change."""
        items = self.table.selectedItems()
        if items:
            row = items[0].row()
            id_item = self.table.item(row, 1)
            if id_item:
                self.person_selected.emit(id_item.text())
    
    def get_selected_person_id(self) -> Optional[str]:
        """Get currently selected person ID."""
        items = self.table.selectedItems()
        if items:
            row = items[0].row()
            id_item = self.table.item(row, 1)
            if id_item:
                return id_item.text()
        return None
    
    def clear_selection(self):
        """Clear table selection."""
        self.table.clearSelection()