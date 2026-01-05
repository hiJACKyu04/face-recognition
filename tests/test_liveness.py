"""
Tests for liveness detection
"""

import pytest
import numpy as np
from src.liveness_detector import LivenessDetector


def test_liveness_detector_initialization():
    """Test liveness detector initialization"""
    detector = LivenessDetector()
    assert detector is not None


def test_texture_analysis():
    """Test texture analysis"""
    detector = LivenessDetector()
    
    # Create dummy face ROI
    face_roi = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    score = detector._analyze_texture(face_roi)
    
    assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])

