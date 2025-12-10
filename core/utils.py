"""
Face Processing Utilities
- Face alignment using landmarks
- Image preprocessing
- Bounding box utilities
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


# Standard face template for alignment (112x112 output)
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],   # Left eye
    [73.5318, 51.5014],   # Right eye
    [56.0252, 71.7366],   # Nose tip
    [41.5493, 92.3655],   # Left mouth corner
    [70.7299, 92.2041]    # Right mouth corner
], dtype=np.float32)


def estimate_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transform matrix from source to destination points.
    
    Args:
        src_pts: Source landmarks (5x2)
        dst_pts: Destination landmarks (5x2)
    
    Returns:
        3x3 transformation matrix
    """
    num = src_pts.shape[0]
    dim = src_pts.shape[1]
    
    # Center the points
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)
    
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean
    
    # Compute scale
    src_std = np.sqrt(np.sum(src_centered ** 2) / num)
    dst_std = np.sqrt(np.sum(dst_centered ** 2) / num)
    
    src_normalized = src_centered / src_std
    dst_normalized = dst_centered / dst_std
    
    # Compute rotation using SVD
    H = np.dot(src_normalized.T, dst_normalized)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    scale = dst_std / src_std
    
    # Build transformation matrix
    T = np.eye(3)
    T[:2, :2] = scale * R
    T[:2, 2] = dst_mean - scale * np.dot(R, src_mean)
    
    return T


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: Tuple[int, int] = (112, 112)
) -> np.ndarray:
    """
    Align face using 5-point landmarks for ArcFace input.
    
    Args:
        image: Input BGR image
        landmarks: 5 facial landmarks [[x1,y1], [x2,y2], ...]
        output_size: Output image size (width, height)
    
    Returns:
        Aligned face image (112x112 for ArcFace)
    """
    landmarks = np.array(landmarks, dtype=np.float32)
    
    # Scale template to output size
    scale = output_size[0] / 112.0
    template = ARCFACE_TEMPLATE * scale
    
    # Estimate transformation
    M = estimate_transform(landmarks, template)
    
    # Apply transformation
    aligned = cv2.warpAffine(
        image,
        M[:2, :],
        output_size,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return aligned


def preprocess_face(
    face_image: np.ndarray,
    target_size: Tuple[int, int] = (112, 112)
) -> np.ndarray:
    """
    Preprocess face image for recognition model.
    
    Args:
        face_image: BGR face image
        target_size: Model input size
    
    Returns:
        Preprocessed image ready for model input
    """
    # Resize if needed
    if face_image.shape[:2] != target_size[::-1]:
        face_image = cv2.resize(face_image, target_size)
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [-1, 1] (ArcFace standard)
    face_normalized = (face_rgb.astype(np.float32) - 127.5) / 127.5
    
    # Transpose to CHW format
    face_transposed = face_normalized.transpose(2, 0, 1)
    
    return face_transposed


def expand_bbox(
    bbox: List[float],
    image_shape: Tuple[int, int],
    scale: float = 1.2
) -> List[int]:
    """
    Expand bounding box by a scale factor.
    
    Args:
        bbox: [x1, y1, x2, y2]
        image_shape: (height, width) of image
        scale: Expansion factor
    
    Returns:
        Expanded bbox clipped to image bounds
    """
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]
    
    # Calculate center and size
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = x2 - x1
    bh = y2 - y1
    
    # Expand
    new_bw = bw * scale
    new_bh = bh * scale
    
    # New coordinates
    new_x1 = int(max(0, cx - new_bw / 2))
    new_y1 = int(max(0, cy - new_bh / 2))
    new_x2 = int(min(w, cx + new_bw / 2))
    new_y2 = int(min(h, cy + new_bh / 2))
    
    return [new_x1, new_y1, new_x2, new_y2]


def crop_face(
    image: np.ndarray,
    bbox: List[float],
    margin: float = 0.2
) -> np.ndarray:
    """
    Crop face region from image with margin.
    
    Args:
        image: Input image
        bbox: [x1, y1, x2, y2]
        margin: Margin ratio to add
    
    Returns:
        Cropped face image
    """
    expanded = expand_bbox(bbox, image.shape, scale=1 + margin)
    x1, y1, x2, y2 = expanded
    return image[y1:y2, x1:x2].copy()


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union between two boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def draw_face_box(
    image: np.ndarray,
    bbox: List[float],
    name: str = "Unknown",
    confidence: float = 0.0,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box and label on image.
    
    Args:
        image: Input image (will be modified)
        bbox: [x1, y1, x2, y2]
        name: Person name to display
        confidence: Recognition confidence
        color: Box color (BGR)
        thickness: Line thickness
    
    Returns:
        Image with drawn box
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare label
    if confidence > 0:
        label = f"{name} ({confidence:.2f})"
    else:
        label = name
    
    # Draw label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
    
    cv2.rectangle(
        image,
        (x1, y1 - text_h - 10),
        (x1 + text_w + 5, y1),
        color,
        -1  # Filled
    )
    
    # Draw text
    cv2.putText(
        image,
        label,
        (x1 + 2, y1 - 5),
        font,
        font_scale,
        (255, 255, 255),  # White text
        1,
        cv2.LINE_AA
    )
    
    return image