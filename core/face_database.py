"""
FAISS-based Face Embedding Database

Thread-safe face database with persistent storage.
"""

import os
import json
import threading
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import faiss

from config.settings import settings
from config.logging_config import logger


@dataclass
class SearchResult:
    """Face search result."""
    person_id: str
    name: str
    similarity: float
    faiss_id: int


class FaceDatabase:
    """
    FAISS-based face embedding database.
    
    Features:
    - Thread-safe operations
    - Persistent storage (save/load)
    - Fast similarity search
    - Metadata mapping
    """
    
    def __init__(
        self,
        dimension: int = 512,
        db_path: Optional[str] = None,
        similarity_threshold: float = 0.4
    ):
        """
        Initialize face database.
        
        Args:
            dimension: Embedding dimension (512 for ArcFace)
            db_path: Directory to store database files
            similarity_threshold: Minimum similarity for match
        """
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        
        # Database path
        if db_path is None:
            db_path = str(settings.embeddings_dir)
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.index_file = self.db_path / "faiss_index.bin"
        self.metadata_file = self.db_path / "metadata.json"
        
        # Thread lock for safe operations
        self._lock = threading.Lock()
        
        # Initialize or load database
        self._load_or_create()
        
        logger.info(
            f"FaceDatabase initialized: {len(self.metadata)} persons, "
            f"dimension={dimension}, path={db_path}"
        )
    
    def _load_or_create(self):
        """Load existing database or create new one."""
        if self.index_file.exists() and self.metadata_file.exists():
            self._load()
        else:
            self._create_new()
    
    def _create_new(self):
        """Create new empty database."""
        # Use L2 distance index (for normalized vectors, equivalent to cosine)
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine for normalized)
        
        # Metadata: faiss_id -> {person_id, name, ...}
        self.metadata: Dict[int, dict] = {}
        
        # Counter for FAISS IDs
        self._next_id = 0
        
        logger.info("Created new empty face database")
    
    def _load(self):
        """Load database from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            # Convert string keys back to int
            self.metadata = {int(k): v for k, v in data['metadata'].items()}
            self._next_id = data.get('next_id', len(self.metadata))
            
            logger.info(f"Loaded face database: {len(self.metadata)} persons")
            
        except Exception as e:
            logger.error(f"Failed to load database: {e}. Creating new.")
            self._create_new()
    
    def save(self):
        """Save database to disk."""
        with self._lock:
            try:
                # Save FAISS index
                faiss.write_index(self.index, str(self.index_file))
                
                # Save metadata
                data = {
                    'metadata': {str(k): v for k, v in self.metadata.items()},
                    'next_id': self._next_id
                }
                with open(self.metadata_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Saved face database: {len(self.metadata)} persons")
                
            except Exception as e:
                logger.error(f"Failed to save database: {e}")
                raise
    
    def add_person(
        self,
        person_id: str,
        name: str,
        embedding: np.ndarray,
        extra_metadata: dict = None
    ) -> int:
        """
        Add a person to the database.
        
        Args:
            person_id: Unique person identifier (e.g., employee ID)
            name: Person's name
            embedding: 512-d face embedding
            extra_metadata: Additional metadata to store
        
        Returns:
            FAISS index ID
        """
        with self._lock:
            # Normalize embedding
            embedding = embedding.astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.reshape(1, -1)
            
            # Add to FAISS
            self.index.add(embedding)
            faiss_id = self._next_id
            self._next_id += 1
            
            # Store metadata
            meta = {
                'person_id': person_id,
                'name': name,
                'faiss_id': faiss_id
            }
            if extra_metadata:
                meta.update(extra_metadata)
            
            self.metadata[faiss_id] = meta
            
            logger.info(f"Added person: {name} (ID: {person_id}, FAISS: {faiss_id})")
            
            return faiss_id
    
    def search(
        self,
        embedding: np.ndarray,
        k: int = 1,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar faces.
        
        Args:
            embedding: Query embedding
            k: Number of results to return
            threshold: Similarity threshold (uses default if None)
        
        Returns:
            List of SearchResult objects
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        with self._lock:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query
            embedding = embedding.astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.reshape(1, -1)
            
            # Search
            k = min(k, self.index.ntotal)
            similarities, indices = self.index.search(embedding, k)
            
            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < 0 or sim < threshold:
                    continue
                
                meta = self.metadata.get(idx)
                if meta:
                    results.append(SearchResult(
                        person_id=meta['person_id'],
                        name=meta['name'],
                        similarity=float(sim),
                        faiss_id=idx
                    ))
            
            return results
    
    def identify(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None
    ) -> Optional[SearchResult]:
        """
        Identify a face (get best match if above threshold).
        
        Args:
            embedding: Face embedding
            threshold: Similarity threshold
        
        Returns:
            Best matching SearchResult or None
        """
        results = self.search(embedding, k=1, threshold=threshold)
        return results[0] if results else None
    
    def get_person_by_id(self, person_id: str) -> Optional[dict]:
        """Get person metadata by person_id."""
        with self._lock:
            for meta in self.metadata.values():
                if meta['person_id'] == person_id:
                    return meta
            return None
    
    def remove_person(self, person_id: str) -> bool:
        """
        Remove a person from the database.
        Note: FAISS doesn't support removal, so we mark as deleted.
        For full removal, rebuild the index.
        
        Args:
            person_id: Person ID to remove
        
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            for faiss_id, meta in list(self.metadata.items()):
                if meta['person_id'] == person_id:
                    del self.metadata[faiss_id]
                    logger.info(f"Removed person: {person_id}")
                    return True
            return False
    
    def get_all_persons(self) -> List[dict]:
        """Get all registered persons."""
        with self._lock:
            return list(self.metadata.values())
    
    def get_count(self) -> int:
        """Get number of registered persons."""
        with self._lock:
            return len(self.metadata)
    
    def clear(self):
        """Clear entire database."""
        with self._lock:
            self._create_new()
            
            # Remove files
            if self.index_file.exists():
                os.remove(self.index_file)
            if self.metadata_file.exists():
                os.remove(self.metadata_file)
            
            logger.warning("Face database cleared")