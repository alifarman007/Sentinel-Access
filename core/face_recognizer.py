"""
ArcFace Face Recognizer with ONNX Runtime

Extracts 512-dimensional embeddings for face recognition.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import onnxruntime as ort

from config.settings import settings
from config.logging_config import logger
from core.utils import align_face, preprocess_face


class FaceRecognizer:
    """
    ArcFace Face Recognizer using ONNX Runtime.
    
    Features:
    - 512-dimensional face embeddings
    - Face alignment using landmarks
    - CUDA acceleration with CPU fallback
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize ArcFace recognizer.
        
        Args:
            model_path: Path to ONNX model file
            use_gpu: Whether to use CUDA if available
        """
        # Use default model path if not specified
        if model_path is None:
            model_path = str(settings.recognition_model_path)
        
        # Setup ONNX Runtime session
        self.session = self._create_session(model_path, use_gpu)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        # Input size (typically 112x112 for ArcFace)
        self.input_size = (self.input_shape[3], self.input_shape[2])  # (W, H)
        
        logger.info(
            f"FaceRecognizer initialized: model={model_path}, "
            f"input_size={self.input_size}, gpu={self._using_gpu}"
        )
    
    def _create_session(self, model_path: str, use_gpu: bool) -> ort.InferenceSession:
        """Create ONNX Runtime inference session."""
        providers = []
        self._using_gpu = False
        
        if use_gpu:
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'DEFAULT',
                }))
                self._using_gpu = True
        
        providers.append('CPUExecutionProvider')
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers
        )
        
        # Check which provider is actually being used
        actual_provider = session.get_providers()[0]
        self._using_gpu = 'CUDA' in actual_provider
        logger.info(f"Recognizer using provider: {actual_provider}")
        
        return session
    
    def get_embedding(
        self,
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Extract face embedding from image using landmarks for alignment.
        
        Args:
            image: BGR image containing face
            landmarks: 5 facial landmarks [[x,y], ...] from detector
        
        Returns:
            512-dimensional normalized embedding
        """
        # Align face using landmarks
        aligned_face = align_face(image, landmarks, output_size=self.input_size)
        
        # Preprocess for model
        face_input = preprocess_face(aligned_face, target_size=self.input_size)
        
        # Add batch dimension
        face_input = np.expand_dims(face_input, axis=0).astype(np.float32)
        
        # Run inference
        embedding = self.session.run(
            [self.output_name],
            {self.input_name: face_input}
        )[0]
        
        # Normalize embedding
        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def get_embedding_from_face(
        self,
        face_image: np.ndarray
    ) -> np.ndarray:
        """
        Extract embedding from already-cropped face image.
        Use this when face is already aligned/cropped.
        
        Args:
            face_image: BGR face image (will be resized to 112x112)
        
        Returns:
            512-dimensional normalized embedding
        """
        # Resize to input size
        if face_image.shape[:2] != self.input_size[::-1]:
            face_image = cv2.resize(face_image, self.input_size)
        
        # Preprocess
        face_input = preprocess_face(face_image, target_size=self.input_size)
        face_input = np.expand_dims(face_input, axis=0).astype(np.float32)
        
        # Run inference
        embedding = self.session.run(
            [self.output_name],
            {self.input_name: face_input}
        )[0]
        
        # Normalize
        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    @staticmethod
    def compute_similarity(
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding (512-d)
            embedding2: Second embedding (512-d)
        
        Returns:
            Similarity score between -1 and 1 (higher = more similar)
        """
        # Embeddings should already be normalized, but ensure it
        e1 = embedding1 / np.linalg.norm(embedding1)
        e2 = embedding2 / np.linalg.norm(embedding2)
        
        return float(np.dot(e1, e2))
    
    @staticmethod
    def is_same_person(
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = 0.4
    ) -> Tuple[bool, float]:
        """
        Determine if two embeddings belong to the same person.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            threshold: Similarity threshold (default 0.4)
        
        Returns:
            (is_match, similarity_score)
        """
        similarity = FaceRecognizer.compute_similarity(embedding1, embedding2)
        return similarity >= threshold, similarity