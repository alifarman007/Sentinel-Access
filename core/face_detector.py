"""
SCRFD Face Detector with ONNX Runtime

Supports both CUDA and CPU execution.
Compatible with det_10g.onnx model from yakhyo/face-reidentification
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import onnxruntime as ort

from config.settings import settings
from config.logging_config import logger


@dataclass
class Detection:
    """Face detection result."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float         # Detection confidence
    landmarks: np.ndarray     # 5 keypoints [[x,y], ...]


class FaceDetector:
    """
    SCRFD Face Detector using ONNX Runtime.
    
    Features:
    - Multi-scale face detection
    - 5-point facial landmark detection
    - CUDA acceleration with CPU fallback
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        use_gpu: bool = True
    ):
        """
        Initialize SCRFD detector.
        
        Args:
            model_path: Path to ONNX model file
            conf_threshold: Minimum detection confidence
            nms_threshold: NMS IoU threshold
            input_size: Model input size (width, height)
            use_gpu: Whether to use CUDA if available
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
        # Use default model path if not specified
        if model_path is None:
            model_path = str(settings.detection_model_path)
        
        # Setup ONNX Runtime session
        self.session = self._create_session(model_path, use_gpu)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Analyze model outputs
        self._analyze_outputs()
        
        # Feature map strides for SCRFD
        self.feat_stride_fpn = [8, 16, 32]
        self.num_anchors = 2
        
        # Pre-compute anchor centers
        self._anchor_centers = {}
        
        logger.info(
            f"FaceDetector initialized: model={model_path}, "
            f"input_size={input_size}, gpu={self._using_gpu}"
        )
    
    def _analyze_outputs(self):
        """Analyze model output structure."""
        outputs = self.session.get_outputs()
        logger.info(f"Model has {len(outputs)} outputs:")
        for i, o in enumerate(outputs):
            logger.info(f"  [{i}] {o.name}: {o.shape}")
    
    def _create_session(self, model_path: str, use_gpu: bool) -> ort.InferenceSession:
        """Create ONNX Runtime inference session."""
        providers = []
        self._using_gpu = False
        
        if use_gpu:
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                try:
                    # Test if CUDA actually works
                    providers.append(('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'cudnn_conv_algo_search': 'DEFAULT',
                    }))
                    self._using_gpu = True
                except Exception as e:
                    logger.warning(f"CUDA setup failed: {e}")
                    self._using_gpu = False
        
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
        logger.info(f"Using provider: {actual_provider}")
        
        return session
    
    def _get_anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        """Generate anchor centers for a feature map."""
        key = (height, width, stride)
        if key not in self._anchor_centers:
            grid_y, grid_x = np.mgrid[:height, :width].astype(np.float32)
            centers = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
            centers = (centers * stride) + (stride // 2)
            centers = np.repeat(centers, self.num_anchors, axis=0)
            self._anchor_centers[key] = centers
        return self._anchor_centers[key]
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for model input.
        
        Returns:
            Preprocessed image, scale factor, (new_height, new_width)
        """
        img_h, img_w = image.shape[:2]
        input_w, input_h = self.input_size
        
        # Calculate scale to fit image in input size
        scale = min(input_w / img_w, input_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image (RGB, float32)
        padded = np.zeros((input_h, input_w, 3), dtype=np.float32)
        padded[:new_h, :new_w, :] = resized.astype(np.float32)
        
        # Normalize: (x - 127.5) / 128
        padded = (padded - 127.5) / 128.0
        
        # HWC -> NCHW
        padded = padded.transpose(2, 0, 1)
        padded = np.expand_dims(padded, axis=0)
        
        return padded.astype(np.float32), scale, (new_h, new_w)
    
    def _distance2bbox(self, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """Convert distance predictions to bounding boxes."""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def _distance2kps(self, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """Convert distance predictions to keypoints."""
        num_points = distance.shape[1] // 2
        kps = np.zeros((points.shape[0], num_points, 2), dtype=np.float32)
        for i in range(num_points):
            kps[:, i, 0] = points[:, 0] + distance[:, i * 2]
            kps[:, i, 1] = points[:, 1] + distance[:, i * 2 + 1]
        return kps
    
    def _decode_single_scale(
        self,
        score_blob: np.ndarray,
        bbox_blob: np.ndarray,
        kps_blob: np.ndarray,
        stride: int,
        scale: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode outputs for a single feature map scale."""
        # Reshape score blob
        if len(score_blob.shape) == 4:
            # NCHW format: (1, num_anchors, H, W)
            scores = score_blob[0].transpose(1, 2, 0).reshape(-1)
        elif len(score_blob.shape) == 3:
            # (1, H*W*num_anchors, 1) or similar
            scores = score_blob.reshape(-1)
        else:
            scores = score_blob.reshape(-1)
        
        # Filter by confidence
        pos_inds = np.where(scores >= self.conf_threshold)[0]
        
        if len(pos_inds) == 0:
            return np.array([]), np.array([]), np.array([])
        
        pos_scores = scores[pos_inds]
        
        # Reshape bbox blob
        if len(bbox_blob.shape) == 4:
            # NCHW: (1, num_anchors*4, H, W)
            h, w = bbox_blob.shape[2], bbox_blob.shape[3]
            bbox_pred = bbox_blob[0].transpose(1, 2, 0).reshape(-1, 4)
        else:
            bbox_pred = bbox_blob.reshape(-1, 4)
        
        # Reshape keypoints blob
        if len(kps_blob.shape) == 4:
            kps_pred = kps_blob[0].transpose(1, 2, 0).reshape(-1, 10)
        else:
            kps_pred = kps_blob.reshape(-1, 10)
        
        # Get anchor centers
        if len(bbox_blob.shape) == 4:
            h, w = bbox_blob.shape[2], bbox_blob.shape[3]
        else:
            # Estimate from blob size
            total = bbox_pred.shape[0]
            h = int(np.sqrt(total / self.num_anchors))
            w = h
        
        anchor_centers = self._get_anchor_centers(h, w, stride)
        
        pos_bbox_pred = bbox_pred[pos_inds] * stride
        pos_kps_pred = kps_pred[pos_inds] * stride
        pos_centers = anchor_centers[pos_inds]
        
        # Decode boxes and keypoints
        bboxes = self._distance2bbox(pos_centers, pos_bbox_pred)
        kps = self._distance2kps(pos_centers, pos_kps_pred)
        
        # Scale back to original image
        bboxes = bboxes / scale
        kps = kps / scale
        
        return pos_scores, bboxes, kps
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect faces in an image.
        
        Args:
            image: BGR image (numpy array)
        
        Returns:
            List of Detection objects
        """
        if image is None or image.size == 0:
            return []
        
        # Preprocess
        input_tensor, scale, (new_h, new_w) = self._preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Collect all detections
        all_scores = []
        all_bboxes = []
        all_kps = []
        
        # Process each stride level
        # SCRFD outputs: [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]
        num_strides = len(self.feat_stride_fpn)
        
        for idx, stride in enumerate(self.feat_stride_fpn):
            score_blob = outputs[idx]
            bbox_blob = outputs[idx + num_strides]
            kps_blob = outputs[idx + num_strides * 2]
            
            scores, bboxes, kps = self._decode_single_scale(
                score_blob, bbox_blob, kps_blob, stride, scale
            )
            
            if len(scores) > 0:
                all_scores.append(scores)
                all_bboxes.append(bboxes)
                all_kps.append(kps)
        
        if not all_scores:
            return []
        
        # Concatenate all detections
        scores = np.concatenate(all_scores)
        bboxes = np.concatenate(all_bboxes)
        kps = np.concatenate(all_kps)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.nms_threshold
        )
        
        if len(indices) == 0:
            return []
        
        indices = indices.flatten()
        
        # Create Detection objects
        detections = []
        for i in indices:
            det = Detection(
                bbox=bboxes[i],
                confidence=float(scores[i]),
                landmarks=kps[i]
            )
            detections.append(det)
        
        return detections