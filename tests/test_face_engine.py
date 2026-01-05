"""
Tests for face engine
"""

import pytest
import numpy as np
import cv2
from src.face_engine import FaceEngine


def test_face_engine_initialization():
    """Test face engine initialization"""
    engine = FaceEngine(model_name="buffalo_l")
    assert engine is not None
    assert engine.model_name == "buffalo_l"


def test_face_detection():
    """Test face detection"""
    engine = FaceEngine(model_name="buffalo_l")
    
    # Create a dummy image
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # This will likely not detect a face, but should not crash
    faces = engine.detect_faces(image)
    assert isinstance(faces, list)


def test_face_comparison():
    """Test face comparison"""
    engine = FaceEngine(model_name="buffalo_l")
    
    # Create dummy embeddings
    emb1 = np.random.rand(512).astype(np.float32)
    emb2 = np.random.rand(512).astype(np.float32)
    
    similarity = engine.compare_faces(emb1, emb2)
    
    assert 0.0 <= similarity <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])

