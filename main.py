"""
Face Re-ID Access Management & Attendance System

Main application entry point.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import main

if __name__ == "__main__":
    main()