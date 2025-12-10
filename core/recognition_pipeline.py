"""
Face Recognition Pipeline

Combines detection, recognition, and database lookup.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from config.settings import settings
from config.logging_config import logger
from core.face_detector import FaceDetector, Detection
from core.face_recognizer import FaceRecognizer
from core.face_database import FaceDatabase, SearchResult


@dataclass
class RecognitionResult:
    """Complete recognition result for a detected face."""
    detection: Detection           # Detection info (bbox, landmarks)
    embedding: np.ndarray         # Face embedding
    identity: Optional[SearchResult]  # Identified person (None if unknown)
    
    @property
    def is_known(self) -> bool:
        return self.identity is not None
    
    @property
    def name(self) -> str:
        return self.identity.name if self.identity else "Unknown"
    
    @property
    def person_id(self) -> Optional[str]:
        return self.identity.person_id if self.identity else None
    
    @property
    def confidence(self) -> float:
        return self.identity.similarity if self.identity else 0.0


class RecognitionPipeline:
    """
    Complete face recognition pipeline.
    
    Handles:
    - Face detection
    - Embedding extraction
    - Identity lookup
    """
    
    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
        database: Optional[FaceDatabase] = None,
        use_gpu: bool = True
    ):
        """
        Initialize recognition pipeline.
        
        Args:
            detector: Face detector (creates default if None)
            recognizer: Face recognizer (creates default if None)
            database: Face database (creates default if None)
            use_gpu: Use GPU acceleration
        """
        self.detector = detector or FaceDetector(use_gpu=use_gpu)
        self.recognizer = recognizer or FaceRecognizer(use_gpu=use_gpu)
        self.database = database or FaceDatabase()
        
        logger.info("RecognitionPipeline initialized")
    
    def process_frame(
        self,
        frame: np.ndarray,
        identify: bool = True
    ) -> List[RecognitionResult]:
        """
        Process a single frame for face recognition.
        
        Args:
            frame: BGR image
            identify: Whether to lookup identities
        
        Returns:
            List of RecognitionResult for each detected face
        """
        results = []
        
        # Detect faces
        detections = self.detector.detect(frame)
        
        for det in detections:
            # Extract embedding
            embedding = self.recognizer.get_embedding(frame, det.landmarks)
            
            # Identify (if enabled and database not empty)
            identity = None
            if identify and self.database.get_count() > 0:
                identity = self.database.identify(embedding)
            
            results.append(RecognitionResult(
                detection=det,
                embedding=embedding,
                identity=identity
            ))
        
        return results
    
    def register_person(
        self,
        frame: np.ndarray,
        person_id: str,
        name: str,
        save_db: bool = True
    ) -> Tuple[bool, str]:
        """
        Register a new person from an image.
        
        Args:
            frame: BGR image containing exactly one face
            person_id: Unique person ID
            name: Person's name
            save_db: Whether to save database after adding
        
        Returns:
            (success, message)
        """
        # Detect faces
        detections = self.detector.detect(frame)
        
        if len(detections) == 0:
            return False, "No face detected in image"
        
        if len(detections) > 1:
            return False, f"Multiple faces detected ({len(detections)}). Please use image with single face."
        
        # Check if person already exists
        if self.database.get_person_by_id(person_id):
            return False, f"Person with ID '{person_id}' already exists"
        
        # Extract embedding
        det = detections[0]
        embedding = self.recognizer.get_embedding(frame, det.landmarks)
        
        # Add to database
        faiss_id = self.database.add_person(
            person_id=person_id,
            name=name,
            embedding=embedding
        )
        
        if save_db:
            self.database.save()
        
        return True, f"Successfully registered {name} (ID: {person_id})"
    
    def register_from_embedding(
        self,
        embedding: np.ndarray,
        person_id: str,
        name: str,
        save_db: bool = True
    ) -> Tuple[bool, str]:
        """Register person from pre-computed embedding."""
        if self.database.get_person_by_id(person_id):
            return False, f"Person with ID '{person_id}' already exists"
        
        self.database.add_person(
            person_id=person_id,
            name=name,
            embedding=embedding
        )
        
        if save_db:
            self.database.save()
        
        return True, f"Successfully registered {name} (ID: {person_id})"