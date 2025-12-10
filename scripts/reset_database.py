"""
Reset all databases - CAUTION: Deletes all data!
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from config.logging_config import setup_logging, logger
from attendance.database import get_db_session
from attendance.models import AttendanceRecord, FaceEmbedding, Person, Camera


def reset_all():
    """Reset all databases."""
    setup_logging(log_level="INFO", log_to_file=False)
    
    print("=" * 50)
    print("⚠️  WARNING: This will DELETE ALL DATA!")
    print("=" * 50)
    print()
    
    confirm = input("Type 'YES' to confirm: ").strip()
    
    if confirm != "YES":
        print("Cancelled.")
        return
    
    print()
    
    # 1. Clear PostgreSQL
    print("Clearing PostgreSQL tables...")
    try:
        with get_db_session() as session:
            session.query(AttendanceRecord).delete()
            session.query(FaceEmbedding).delete()
            session.query(Person).delete()
            session.query(Camera).delete()
            session.commit()
        print("✓ PostgreSQL cleared")
    except Exception as e:
        print(f"✗ PostgreSQL error: {e}")
    
    # 2. Clear FAISS
    print("Clearing FAISS database...")
    try:
        embeddings_dir = settings.embeddings_dir
        if embeddings_dir.exists():
            for file in embeddings_dir.iterdir():
                file.unlink()
        print("✓ FAISS cleared")
    except Exception as e:
        print(f"✗ FAISS error: {e}")
    
    # 3. Clear face images
    print("Clearing face images...")
    try:
        faces_dir = settings.faces_dir
        if faces_dir.exists():
            for file in faces_dir.iterdir():
                if file.is_file():
                    file.unlink()
        print("✓ Face images cleared")
    except Exception as e:
        print(f"✗ Face images error: {e}")
    
    print()
    print("=" * 50)
    print("✓ All data cleared! You can start fresh now.")
    print("=" * 50)


if __name__ == "__main__":
    reset_all()