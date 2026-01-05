"""
Liveness detection module
Detects if a face is live (real) or spoofed (photo/video)
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class LivenessDetector:
    """Liveness detection using multiple methods"""
    
    def __init__(self, methods: List[str] = None, threshold: float = 0.7):
        """
        Initialize liveness detector
        
        Args:
            methods: List of methods to use ['eye_blink', 'head_movement', 'texture_analysis']
            threshold: Overall liveness threshold
        """
        self.methods = methods or ['eye_blink', 'head_movement', 'texture_analysis']
        self.threshold = threshold
        
        # For eye blink detection
        self.eye_blink_history = {}  # Track eye aspect ratio history per face
        self.blink_count = {}  # Track blink counts
        
        # For head movement
        self.head_pose_history = {}  # Track head pose history
        
        logger.info(f"Liveness detector initialized with methods: {self.methods}")
    
    def detect_liveness(self, image: np.ndarray, face_bbox: List[float], 
                       face_kps: Optional[List] = None, face_id: Optional[str] = None) -> Dict:
        """
        Detect if face is live
        
        Args:
            image: Input image
            face_bbox: Face bounding box [x1, y1, x2, y2]
            face_kps: Face keypoints (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
            face_id: Optional face ID for tracking across frames
            
        Returns:
            Dictionary with liveness score and method results
        """
        results = {
            'is_live': False,
            'confidence': 0.0,
            'methods': {}
        }
        
        if not face_id:
            face_id = f"face_{id(face_bbox)}"
        
        # Extract face region
        x1, y1, x2, y2 = [int(coord) for coord in face_bbox]
        face_roi = image[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return results
        
        method_scores = []
        
        # Eye blink detection
        if 'eye_blink' in self.methods and face_kps:
            blink_score = self._detect_eye_blink(face_kps, face_id)
            results['methods']['eye_blink'] = blink_score
            method_scores.append(blink_score)
        
        # Head movement detection
        if 'head_movement' in self.methods and face_kps:
            movement_score = self._detect_head_movement(face_kps, face_id)
            results['methods']['head_movement'] = movement_score
            method_scores.append(movement_score)
        
        # Texture analysis
        if 'texture_analysis' in self.methods:
            texture_score = self._analyze_texture(face_roi)
            results['methods']['texture_analysis'] = texture_score
            method_scores.append(texture_score)
        
        # Calculate overall confidence (average of method scores)
        if method_scores:
            results['confidence'] = float(np.mean(method_scores))
            results['is_live'] = results['confidence'] >= self.threshold
        
        return results
    
    def _detect_eye_blink(self, face_kps: List, face_id: str, 
                         min_blinks: int = 1) -> float:
        """
        Detect eye blinks using eye aspect ratio (EAR)
        
        Args:
            face_kps: Face keypoints
            face_id: Face ID for tracking
            min_blinks: Minimum blinks required
            
        Returns:
            Liveness score (0-1)
        """
        if len(face_kps) < 5:
            return 0.5  # Neutral score if not enough keypoints
        
        # Extract eye keypoints (assuming standard 5-point format)
        # face_kps format: [left_eye, right_eye, nose, left_mouth, right_mouth]
        left_eye = np.array(face_kps[0])
        right_eye = np.array(face_kps[1])
        
        # Calculate eye aspect ratio (EAR)
        # EAR = (vertical distance) / (horizontal distance)
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Initialize history if needed
        if face_id not in self.eye_blink_history:
            self.eye_blink_history[face_id] = deque(maxlen=10)
            self.blink_count[face_id] = 0
        
        self.eye_blink_history[face_id].append(avg_ear)
        
        # Detect blink (EAR drops below threshold)
        ear_threshold = 0.25
        if len(self.eye_blink_history[face_id]) >= 3:
            recent_ears = list(self.eye_blink_history[face_id])
            # Check if EAR dropped significantly (blink)
            if recent_ears[-1] < ear_threshold and recent_ears[-2] >= ear_threshold:
                self.blink_count[face_id] += 1
        
        # Score based on blink count
        if self.blink_count[face_id] >= min_blinks:
            return 1.0
        elif self.blink_count[face_id] > 0:
            return 0.7
        else:
            return 0.3
    
    def _calculate_ear(self, eye_points: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio"""
        # Simplified EAR calculation
        # For 2-point eye, use distance as approximation
        if len(eye_points) >= 2:
            # Use distance between points as approximation
            dist = np.linalg.norm(eye_points[1] - eye_points[0]) if len(eye_points) > 1 else 1.0
            return min(1.0, dist / 50.0)  # Normalize
        return 0.3
    
    def _detect_head_movement(self, face_kps: List, face_id: str,
                              threshold: float = 0.1) -> float:
        """
        Detect head movement by tracking keypoint positions
        
        Args:
            face_kps: Face keypoints
            face_id: Face ID for tracking
            threshold: Movement threshold
            
        Returns:
            Liveness score (0-1)
        """
        if len(face_kps) < 3:
            return 0.5
        
        # Use nose position as reference
        nose = np.array(face_kps[2])
        
        # Initialize history
        if face_id not in self.head_pose_history:
            self.head_pose_history[face_id] = deque(maxlen=10)
        
        self.head_pose_history[face_id].append(nose)
        
        # Calculate movement variance
        if len(self.head_pose_history[face_id]) >= 3:
            positions = np.array(list(self.head_pose_history[face_id]))
            variance = np.var(positions, axis=0)
            movement = np.mean(variance)
            
            # Score based on movement
            if movement > threshold:
                return 1.0
            elif movement > threshold * 0.5:
                return 0.7
            else:
                return 0.3
        
        return 0.5  # Neutral if not enough history
    
    def _analyze_texture(self, face_roi: np.ndarray) -> float:
        """
        Analyze texture to detect photo/video spoofing
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            Liveness score (0-1)
        """
        if face_roi.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        
        # Method 1: Local Binary Pattern (LBP) variance
        # Higher variance indicates more texture (likely real face)
        lbp = self._local_binary_pattern(gray)
        lbp_variance = np.var(lbp)
        
        # Method 2: Color diversity
        # Real faces have more color variation
        if len(face_roi.shape) == 3:
            color_variance = np.var(face_roi, axis=2).mean()
        else:
            color_variance = np.var(gray)
        
        # Method 3: Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combine scores
        texture_score = (
            min(1.0, lbp_variance / 100.0) * 0.4 +
            min(1.0, color_variance / 50.0) * 0.3 +
            min(1.0, edge_density * 10.0) * 0.3
        )
        
        return float(texture_score)
    
    def _local_binary_pattern(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Calculate Local Binary Pattern"""
        # Simplified LBP implementation
        h, w = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if 0 <= x < h and 0 <= y < w:
                        if image[x, y] >= center:
                            code |= (1 << k)
                lbp[i, j] = code
        
        return lbp
    
    def reset_tracking(self, face_id: Optional[str] = None):
        """Reset tracking history for a face or all faces"""
        if face_id:
            self.eye_blink_history.pop(face_id, None)
            self.blink_count.pop(face_id, None)
            self.head_pose_history.pop(face_id, None)
        else:
            self.eye_blink_history.clear()
            self.blink_count.clear()
            self.head_pose_history.clear()

