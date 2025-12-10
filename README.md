# Face Re-ID Access Management & Attendance System

A production-grade face recognition attendance system using SCRFD detection, ArcFace recognition, FAISS vector database, and PostgreSQL.

## Features

- **Real-time Face Recognition** - CUDA-accelerated detection and recognition
- **Multi-Camera Support** - Up to 4 simultaneous RTSP/webcam streams
- **Automatic Attendance** - Deduplication to prevent duplicate entries
- **Entry/Exit Tracking** - Optional exit time tracking
- **Modern GUI** - Dark-themed PySide6 interface
- **Export Capability** - CSV export for attendance records

## System Requirements

- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA support (RTX 2050 or better recommended)
- Python 3.11+
- PostgreSQL 16+
- 8GB RAM minimum

## Installation

### 1. Install Dependencies
```bash
# CUDA 12.6
# cuDNN 9.x
# GStreamer 1.22+
# PostgreSQL 16
```

### 2. Setup Python Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Database
```bash
# Create PostgreSQL database
psql -U postgres -c "CREATE DATABASE face_attendance;"

# Initialize tables
python scripts/init_database.py
```

### 4. Configure Environment

Edit `.env` file with your settings:
- Database credentials
- Recognition thresholds
- Camera settings

### 5. Run Application
```bash
python main.py
```

Or double-click `run.bat`

## Usage

### Add Person
1. Navigate to "Add Person" page
2. Start camera or upload image
3. Capture face when detected
4. Fill in person details
5. Click "Register Person"

### Live Cameras
1. Navigate to "Live Cameras" page
2. Click "Add Camera"
3. Enter RTSP URL or select webcam
4. Face recognition runs automatically
5. Attendance is logged for recognized persons

### View Attendance
1. Navigate to "Attendance Log" page
2. Use date filters to select range
3. Search by name or ID
4. Export to CSV as needed

## Project Structure
```
Face-Re-ID-Access-Management-Attendance-System/
├── main.py                 # Application entry point
├── config/                 # Configuration
│   ├── settings.py         # Pydantic settings
│   └── logging_config.py   # Loguru configuration
├── core/                   # Core recognition modules
│   ├── face_detector.py    # SCRFD detector
│   ├── face_recognizer.py  # ArcFace recognizer
│   ├── face_database.py    # FAISS operations
│   └── recognition_pipeline.py
├── camera/                 # Camera handling
│   ├── rtsp_stream.py      # RTSP stream handler
│   └── camera_manager.py   # Multi-camera manager
├── attendance/             # Attendance logic
│   ├── models.py           # SQLAlchemy models
│   ├── database.py         # DB connection
│   └── attendance_service.py
├── gui/                    # PySide6 GUI
│   ├── main_window.py      # Main application window
│   ├── pages/              # Application pages
│   ├── widgets/            # Reusable widgets
│   ├── dialogs/            # Dialog windows
│   └── threads/            # Background threads
├── models/                 # ONNX models
│   ├── det_10g.onnx        # SCRFD detection
│   └── w600k_r50.onnx      # ArcFace recognition
├── data/                   # Runtime data
│   ├── faces/              # Registered face images
│   ├── embeddings/         # FAISS index
│   └── logs/               # Application logs
└── scripts/                # Utility scripts
    ├── init_database.py    # Database initialization
    ├── reset_database.py   # Clear all data
    └── test_*.py           # Test scripts
```

## Performance

| Component | Performance |
|-----------|-------------|
| Face Detection | ~24 FPS (CUDA) |
| Face Recognition | ~14 FPS (full pipeline) |
| Supported Cameras | Up to 4 simultaneous |
| Recognition Accuracy | ~99% (ArcFace) |

## Technologies

- **Detection**: SCRFD (det_10g) - InsightFace
- **Recognition**: ArcFace (w600k_r50) - InsightFace
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Database**: PostgreSQL + SQLAlchemy
- **GUI**: PySide6 (Qt for Python)
- **Inference**: ONNX Runtime with CUDA
- **Video**: GStreamer + NVDEC hardware decoding

## License

This project is for educational and internal use.

## Credits

- InsightFace for SCRFD and ArcFace models
- Facebook AI for FAISS
- NVIDIA for CUDA and TensorRT