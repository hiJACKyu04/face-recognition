"""
Face Recognition Engine using InsightFace with RetinaFace buffalo_l model
"""

import numpy as np
import cv2
import insightface
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class FaceEngine:
    """Face recognition engine using RetinaFace buffalo_l model"""
    
    def __init__(self, model_name: str = "buffalo_l", providers: Optional[List[str]] = None, 
                 det_size: Tuple[int, int] = (640, 640), det_thresh: float = 0.5):
        """
        Initialize the face recognition engine
        
        Args:
            model_name: Model name (default: buffalo_l)
            providers: ONNX Runtime providers (auto-detected if None)
            det_size: Detection size (width, height)
            det_thresh: Detection threshold
        """
        self.model_name = model_name
        self.det_size = det_size
        self.det_thresh = det_thresh
        
        # Auto-detect providers if not specified
        if providers is None:
            providers = self._detect_providers()
        
        logger.info(f"Initializing InsightFace with model: {model_name}")
        logger.info(f"Using providers: {providers}")
        
        # Initialize InsightFace
        self.app = insightface.app.FaceAnalysis(
            name=model_name,
            providers=providers
        )
        self.app.prepare(ctx_id=0, det_size=det_size)
        
        logger.info("Face engine initialized successfully")
    
    def _detect_providers(self) -> List[str]:
        """Auto-detect available ONNX Runtime providers"""
        providers = ['CPUExecutionProvider']
        
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            
            # Prefer GPU providers if available
            if 'CUDAExecutionProvider' in available_providers:
                providers.insert(0, 'CUDAExecutionProvider')
                logger.info("CUDA provider detected")
            elif 'CoreMLExecutionProvider' in available_providers:
                providers.insert(0, 'CoreMLExecutionProvider')
                logger.info("CoreML provider detected (Apple Silicon)")
            elif 'DirectMLExecutionProvider' in available_providers:
                providers.insert(0, 'DirectMLExecutionProvider')
                logger.info("DirectML provider detected (Windows)")
        except Exception as e:
            logger.warning(f"Could not detect providers: {e}")
        
        return providers
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of face dictionaries with keys: bbox, kps, det_score, embedding, age, gender
        """
        try:
            faces = self.app.get(image)
            
            results = []
            for face in faces:
                result = {
                    'bbox': face.bbox.tolist(),  # [x1, y1, x2, y2]
                    'kps': face.kps.tolist() if hasattr(face, 'kps') else None,  # Keypoints
                    'det_score': float(face.det_score),
                    'embedding': face.normed_embedding.tolist() if hasattr(face, 'normed_embedding') else None,
                    'age': int(face.age) if hasattr(face, 'age') else None,
                    'gender': int(face.gender) if hasattr(face, 'gender') else None,
                }
                
                # Filter by detection threshold
                if result['det_score'] >= self.det_thresh:
                    results.append(result)
            
            logger.debug(f"Detected {len(results)} faces")
            return results
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def extract_embedding(self, image: np.ndarray, bbox: Optional[List[float]] = None) -> Optional[np.ndarray]:
        """
        Extract face embedding from image
        
        Args:
            image: Input image as numpy array
            bbox: Optional bounding box [x1, y1, x2, y2]. If None, detects first face.
            
        Returns:
            Face embedding as numpy array (512 dimensions) or None
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        # Use provided bbox or first detected face
        if bbox:
            # Find face matching bbox
            for face in faces:
                face_bbox = face['bbox']
                # Check if bboxes overlap significantly
                iou = self._calculate_iou(bbox, face_bbox)
                if iou > 0.5:
                    embedding = np.array(face['embedding'])
                    return embedding
            return None
        else:
            # Return first face embedding
            embedding = np.array(faces[0]['embedding'])
            return embedding
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Normalize embeddings
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Clamp to [0, 1]
        similarity = max(0.0, min(1.0, similarity))
        
        return float(similarity)
    
    def identify_face(self, query_embedding: np.ndarray, database_embeddings: List[np.ndarray], 
                      threshold: float = 0.6, max_results: int = 10) -> List[Tuple[int, float]]:
        """
        Identify a face from a database of embeddings (1:N search)
        
        Args:
            query_embedding: Query face embedding
            database_embeddings: List of database face embeddings
            threshold: Similarity threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of (index, similarity) tuples sorted by similarity
        """
        if not database_embeddings:
            return []
        
        similarities = []
        for idx, db_embedding in enumerate(database_embeddings):
            similarity = self.compare_faces(query_embedding, db_embedding)
            if similarity >= threshold:
                similarities.append((idx, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        return similarities[:max_results]
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def draw_faces(self, image: np.ndarray, faces: List[Dict[str, Any]], 
                   show_bbox: bool = True, show_kps: bool = False, 
                   show_info: bool = False) -> np.ndarray:
        """
        Draw face detection results on image
        
        Args:
            image: Input image
            faces: List of face dictionaries
            show_bbox: Show bounding boxes
            show_kps: Show keypoints
            show_info: Show age/gender info
            
        Returns:
            Image with drawn faces
        """
        result_image = image.copy()
        
        for face in faces:
            bbox = face['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            if show_bbox:
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if show_kps and face.get('kps'):
                kps = face['kps']
                for kp in kps:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)
            
            if show_info:
                info_text = []
                if face.get('age'):
                    info_text.append(f"Age: {face['age']}")
                if face.get('gender') is not None:
                    gender = "Male" if face['gender'] == 1 else "Female"
                    info_text.append(f"Gender: {gender}")
                if face.get('det_score'):
                    info_text.append(f"Score: {face['det_score']:.2f}")
                
                if info_text:
                    text = ", ".join(info_text)
                    cv2.putText(result_image, text, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_image

