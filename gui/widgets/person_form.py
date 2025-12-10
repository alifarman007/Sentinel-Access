"""
Person Registration Form Widget

Form for entering person details with validation.
"""

from typing import Optional, Tuple
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QComboBox, QPushButton, QLabel,
    QFrame, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression


class PersonForm(QWidget):
    """
    Form widget for person registration.
    
    Signals:
        submitted: Emitted when form is submitted with valid data
        cancelled: Emitted when form is cancelled
    """
    
    # Signals
    submitted = Signal(dict)  # Form data dict
    cancelled = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the form UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Form title
        title = QLabel("Person Details")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
        layout.addWidget(title)
        
        layout.addSpacing(10)
        
        # Form layout
        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        form_layout.setLabelAlignment(Qt.AlignRight)
        
        # Person ID (required, alphanumeric)
        self.person_id_input = QLineEdit()
        self.person_id_input.setPlaceholderText("e.g., EMP001, STU2024001")
        self.person_id_input.setMaxLength(50)
        # Allow alphanumeric and some special chars
        validator = QRegularExpressionValidator(QRegularExpression(r"[A-Za-z0-9_\-]+"))
        self.person_id_input.setValidator(validator)
        self._style_input(self.person_id_input)
        form_layout.addRow("Person ID *:", self.person_id_input)
        
        # Full Name (required)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Full name")
        self.name_input.setMaxLength(255)
        self._style_input(self.name_input)
        form_layout.addRow("Full Name *:", self.name_input)
        
        # Department (optional)
        self.department_input = QComboBox()
        self.department_input.setEditable(True)
        self.department_input.addItems([
            "",
            "Engineering",
            "Human Resources",
            "Finance",
            "Marketing",
            "Operations",
            "IT",
            "Security",
            "Administration",
            "Other"
        ])
        self._style_input(self.department_input)
        form_layout.addRow("Department:", self.department_input)
        
        # Email (optional)
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("email@example.com")
        self._style_input(self.email_input)
        form_layout.addRow("Email:", self.email_input)
        
        # Phone (optional)
        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("+880 1XXX-XXXXXX")
        self._style_input(self.phone_input)
        form_layout.addRow("Phone:", self.phone_input)
        
        layout.addLayout(form_layout)
        
        layout.addSpacing(15)
        
        # Validation message
        self.validation_label = QLabel("")
        self.validation_label.setStyleSheet("color: #F44336; font-size: 12px;")
        self.validation_label.setWordWrap(True)
        layout.addWidget(self.validation_label)
        
        layout.addStretch()
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_form)
        self._style_button(self.clear_btn, secondary=True)
        btn_layout.addWidget(self.clear_btn)
        
        btn_layout.addStretch()
        
        self.submit_btn = QPushButton("Register Person")
        self.submit_btn.clicked.connect(self._on_submit)
        self._style_button(self.submit_btn)
        btn_layout.addWidget(self.submit_btn)
        
        layout.addLayout(btn_layout)
    
    def _style_input(self, widget):
        """Apply consistent styling to input widgets."""
        widget.setStyleSheet("""
            QLineEdit, QComboBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px;
                color: white;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #4a6fa5;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #aaa;
                margin-right: 10px;
            }
        """)
        widget.setMinimumHeight(36)
    
    def _style_button(self, button, secondary=False):
        """Apply button styling."""
        if secondary:
            button.setStyleSheet("""
                QPushButton {
                    background-color: #3d3d3d;
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: 10px 20px;
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
                    padding: 10px 20px;
                    color: white;
                    font-size: 13px;
                    font-weight: bold;
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
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate form data.
        
        Returns:
            (is_valid, error_message)
        """
        person_id = self.person_id_input.text().strip()
        name = self.name_input.text().strip()
        
        if not person_id:
            return False, "Person ID is required"
        
        if len(person_id) < 3:
            return False, "Person ID must be at least 3 characters"
        
        if not name:
            return False, "Full name is required"
        
        if len(name) < 2:
            return False, "Name must be at least 2 characters"
        
        # Validate email if provided
        email = self.email_input.text().strip()
        if email and "@" not in email:
            return False, "Invalid email format"
        
        return True, ""
    
    def get_data(self) -> dict:
        """Get form data as dictionary."""
        return {
            'person_id': self.person_id_input.text().strip(),
            'name': self.name_input.text().strip(),
            'department': self.department_input.currentText().strip() or None,
            'email': self.email_input.text().strip() or None,
            'phone': self.phone_input.text().strip() or None
        }
    
    def set_data(self, data: dict):
        """Set form data from dictionary."""
        self.person_id_input.setText(data.get('person_id', ''))
        self.name_input.setText(data.get('name', ''))
        
        dept = data.get('department', '')
        idx = self.department_input.findText(dept)
        if idx >= 0:
            self.department_input.setCurrentIndex(idx)
        else:
            self.department_input.setCurrentText(dept)
        
        self.email_input.setText(data.get('email', '') or '')
        self.phone_input.setText(data.get('phone', '') or '')
    
    def clear_form(self):
        """Clear all form fields."""
        self.person_id_input.clear()
        self.name_input.clear()
        self.department_input.setCurrentIndex(0)
        self.email_input.clear()
        self.phone_input.clear()
        self.validation_label.clear()
    
    def show_error(self, message: str):
        """Show validation error message."""
        self.validation_label.setText(f"⚠️ {message}")
    
    def clear_error(self):
        """Clear validation error."""
        self.validation_label.clear()
    
    def _on_submit(self):
        """Handle form submission."""
        is_valid, error = self.validate()
        
        if not is_valid:
            self.show_error(error)
            return
        
        self.clear_error()
        self.submitted.emit(self.get_data())
    
    def set_submit_enabled(self, enabled: bool):
        """Enable or disable submit button."""
        self.submit_btn.setEnabled(enabled)