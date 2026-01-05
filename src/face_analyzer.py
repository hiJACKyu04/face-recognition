"""
Face attribute analysis module
Analyzes age, gender, emotion, mask, and face quality
"""

import numpy as np
import cv2
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """Face attribute analyzer"""
    
    def __init__(self):
        """Initialize face analyzer"""
        logger.info("Face analyzer initialized")
    
    def analyze_face(self, image: np.ndarray, face_bbox: List[float],
                     face_data: Optional[Dict] = None) -> Dict:
        """
        Analyze face attributes
        
        Args:
            image: Input image
            face_bbox: Face bounding box [x1, y1, x2, y2]
            face_data: Optional face data from detection (may contain age/gender)
            
        Returns:
            Dictionary with face attributes
        """
        x1, y1, x2, y2 = [int(coord) for coord in face_bbox]
        face_roi = image[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return {}
        
        results = {}
        
        # Age and gender (from RetinaFace if available)
        if face_data:
            if 'age' in face_data:
                results['age'] = face_data['age']
            if 'gender' in face_data:
                results['gender'] = face_data['gender']
                results['gender_label'] = 'Male' if face_data['gender'] == 1 else 'Female'
        
        # Emotion detection
        results['emotion'] = self._detect_emotion(face_roi)
        
        # Mask detection
        results['mask'] = self._detect_mask(face_roi, face_bbox)
        
        # Face quality
        results['quality'] = self._assess_quality(image, face_bbox, face_roi)
        
        return results
    
    def _detect_emotion(self, face_roi: np.ndarray) -> Dict:
        """
        Detect emotion from face (simplified version)
        In production, use a trained emotion detection model
        
        Returns:
            Dictionary with emotion predictions
        """
        # Placeholder - in production, use FER or DeepFace emotion model
        # For now, return neutral
        return {
            'emotion': 'neutral',
            'confidence': 0.5,
            'probabilities': {
                'angry': 0.1,
                'disgust': 0.05,
                'fear': 0.05,
                'happy': 0.2,
                'sad': 0.1,
                'surprise': 0.1,
                'neutral': 0.3
            }
        }
    
    def _detect_mask(self, face_roi: np.ndarray, face_bbox: List[float]) -> Dict:
        """
        Detect if face mask is present
        
        Args:
            face_roi: Face region
            face_bbox: Face bounding box
            
        Returns:
            Dictionary with mask detection results
        """
        # Simple heuristic: check lower face region for mask-like patterns
        h, w = face_roi.shape[:2]
        
        # Lower third of face (where mask would be)
        lower_region = face_roi[int(h * 0.6):, :]
        
        if lower_region.size == 0:
            return {'has_mask': False, 'confidence': 0.5}
        
        # Convert to grayscale
        gray = cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY) if len(lower_region.shape) == 3 else lower_region
        
        # Check for uniform color/texture in lower region (mask characteristic)
        variance = np.var(gray)
        mean_brightness = np.mean(gray)
        
        # Heuristic: masks often have lower variance and specific brightness
        has_mask = variance < 500 and 80 < mean_brightness < 200
        
        confidence = 0.7 if has_mask else 0.3
        
        return {
            'has_mask': bool(has_mask),
            'confidence': float(confidence)
        }
    
    def _assess_quality(self, image: np.ndarray, face_bbox: List[float], 
                        face_roi: np.ndarray) -> Dict:
        """
        Assess face quality (blur, lighting, angle, pose)
        
        Args:
            image: Full image
            face_bbox: Face bounding box
            face_roi: Face region of interest
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {}
        
        # Blur detection (Laplacian variance)
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_metrics['blur'] = {
            'score': min(1.0, laplacian_var / 100.0),
            'is_blurry': laplacian_var < 50.0
        }
        
        # Lighting assessment
        brightness = np.mean(gray)
        contrast = np.std(gray)
        quality_metrics['lighting'] = {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'is_good': 50 < brightness < 200 and contrast > 30
        }
        
        # Face size assessment
        face_width = face_bbox[2] - face_bbox[0]
        face_height = face_bbox[3] - face_bbox[1]
        face_size = (face_width + face_height) / 2.0
        quality_metrics['size'] = {
            'pixels': float(face_size),
            'is_adequate': face_size >= 100
        }
        
        # Aspect ratio (should be close to 1:1 for frontal faces)
        aspect_ratio = face_width / face_height if face_height > 0 else 1.0
        quality_metrics['aspect_ratio'] = {
            'ratio': float(aspect_ratio),
            'is_normal': 0.7 < aspect_ratio < 1.3
        }
        
        # Overall quality score
        quality_scores = [
            quality_metrics['blur']['score'],
            1.0 if quality_metrics['lighting']['is_good'] else 0.5,
            1.0 if quality_metrics['size']['is_adequate'] else 0.5,
            1.0 if quality_metrics['aspect_ratio']['is_normal'] else 0.5
        ]
        quality_metrics['overall_score'] = float(np.mean(quality_scores))
        quality_metrics['is_good_quality'] = quality_metrics['overall_score'] > 0.7
        
        return quality_metrics

