"""
Liveness detection API routes
"""

import cv2
import numpy as np
from flask import Blueprint, request, jsonify, g
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.face_engine import FaceEngine
from src.liveness_detector import LivenessDetector
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from middleware import require_api_key

logger = logging.getLogger(__name__)

liveness_bp = Blueprint('liveness', __name__)


def get_face_engine():
    """Get face engine instance"""
    if 'face_engine' not in g:
        config = g.config.get('model', {})
        g.face_engine = FaceEngine(
            model_name=config.get('name', 'buffalo_l'),
            det_size=tuple(config.get('det_size', [640, 640])),
            det_thresh=config.get('detection_threshold', 0.5)
        )
    return g.face_engine


def get_liveness_detector():
    """Get liveness detector instance"""
    if 'liveness_detector' not in g:
        liveness_config = g.config.get('liveness', {})
        methods = liveness_config.get('methods', ['eye_blink', 'head_movement', 'texture_analysis'])
        threshold = liveness_config.get('threshold', 0.7)
        g.liveness_detector = LivenessDetector(methods=methods, threshold=threshold)
    return g.liveness_detector


def decode_image(file) -> np.ndarray:
    """Decode uploaded image file"""
    try:
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        return image
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise


@liveness_bp.route('/liveness/detect', methods=['POST'])
@require_api_key
def detect_liveness():
    """Detect face liveness"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        image = decode_image(file)
        
        face_engine = get_face_engine()
        liveness_detector = get_liveness_detector()
        
        # Detect face
        faces = face_engine.detect_faces(image)
        
        if not faces:
            return jsonify({'error': 'No face detected'}), 400
        
        # Use first face
        face = faces[0]
        
        # Detect liveness
        result = liveness_detector.detect_liveness(
            image=image,
            face_bbox=face['bbox'],
            face_kps=face.get('kps'),
            face_id=f"api_{id(image)}"
        )
        
        return jsonify({
            'success': True,
            'is_live': result['is_live'],
            'confidence': result['confidence'],
            'methods': result['methods']
        })
    
    except Exception as e:
        logger.error(f"Error in detect_liveness: {e}")
        return jsonify({'error': str(e)}), 500

