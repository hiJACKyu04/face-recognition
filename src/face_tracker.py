"""
Face tracking module
Tracks the same person across multiple video frames
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a tracked face across frames"""
    track_id: int
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    bbox_history: List[List[float]] = field(default_factory=list)
    frame_history: List[int] = field(default_factory=list)
    timestamp_history: List[float] = field(default_factory=list)
    last_seen_frame: int = 0
    confidence: float = 0.0
    age: Optional[int] = None
    gender: Optional[int] = None
    
    def update(self, bbox: List[float], frame_num: int, timestamp: float, 
               embedding: Optional[np.ndarray] = None, person_id: Optional[int] = None,
               person_name: Optional[str] = None, confidence: float = 0.0,
               age: Optional[int] = None, gender: Optional[int] = None):
        """Update track with new detection"""
        self.bbox_history.append(bbox)
        self.frame_history.append(frame_num)
        self.timestamp_history.append(timestamp)
        self.last_seen_frame = frame_num
        
        if embedding is not None:
            # Update embedding (average with previous if exists)
            if self.embedding is None:
                self.embedding = embedding
            else:
                # Weighted average (more weight to recent)
                self.embedding = 0.7 * self.embedding + 0.3 * embedding
        
        if person_id is not None:
            self.person_id = person_id
        if person_name is not None:
            self.person_name = person_name
        
        self.confidence = max(self.confidence, confidence)
        if age is not None:
            self.age = age
        if gender is not None:
            self.gender = gender
    
    def get_current_bbox(self) -> Optional[List[float]]:
        """Get most recent bounding box"""
        return self.bbox_history[-1] if self.bbox_history else None
    
    def get_duration(self) -> float:
        """Get total duration this person appears"""
        if len(self.timestamp_history) < 2:
            return 0.0
        return self.timestamp_history[-1] - self.timestamp_history[0]
    
    def get_appearances(self) -> List[Dict]:
        """Get appearance segments (continuous appearances)"""
        if not self.frame_history:
            return []
        
        appearances = []
        start_frame = self.frame_history[0]
        start_time = self.timestamp_history[0]
        
        for i in range(1, len(self.frame_history)):
            # Check if there's a gap (more than 30 frames)
            if self.frame_history[i] - self.frame_history[i-1] > 30:
                # End of current appearance
                appearances.append({
                    'start_frame': start_frame,
                    'end_frame': self.frame_history[i-1],
                    'start_time': start_time,
                    'end_time': self.timestamp_history[i-1],
                    'frames': self.frame_history[self.frame_history.index(start_frame):i]
                })
                # Start new appearance
                start_frame = self.frame_history[i]
                start_time = self.timestamp_history[i]
        
        # Add last appearance
        appearances.append({
            'start_frame': start_frame,
            'end_frame': self.frame_history[-1],
            'start_time': start_time,
            'end_time': self.timestamp_history[-1],
            'frames': self.frame_history[self.frame_history.index(start_frame):]
        })
        
        return appearances


class FaceTracker:
    """Track faces across video frames"""
    
    def __init__(self, iou_threshold: float = 0.3, embedding_threshold: float = 0.7,
                 max_age: int = 30, min_hits: int = 3):
        """
        Initialize face tracker
        
        Args:
            iou_threshold: IoU threshold for bounding box matching
            embedding_threshold: Similarity threshold for identity matching
            max_age: Maximum frames to keep inactive track
            min_hits: Minimum detections to confirm a track
        """
        self.iou_threshold = iou_threshold
        self.embedding_threshold = embedding_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0
        
        logger.info("Face tracker initialized")
    
    def update(self, detections: List[Dict], frame_num: int, timestamp: float) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of face detections with bbox, embedding, etc.
            frame_num: Current frame number
            timestamp: Current timestamp in seconds
            
        Returns:
            List of tracked faces with track_id
        """
        self.frame_count = frame_num
        
        if not detections:
            # Remove old tracks
            self._remove_old_tracks()
            return []
        
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._associate_detections_to_tracks(detections)
        
        # Update matched tracks
        for track_id, det_idx in matched:
            detection = detections[det_idx]
            track = self.tracks[track_id]
            
            track.update(
                bbox=detection['bbox'],
                frame_num=frame_num,
                timestamp=timestamp,
                embedding=np.array(detection.get('embedding')) if detection.get('embedding') else None,
                person_id=detection.get('person_id'),
                person_name=detection.get('person_name'),
                confidence=detection.get('similarity', 0.0) if detection.get('identified') else 0.0,
                age=detection.get('age'),
                gender=detection.get('gender')
            )
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            track = Track(track_id=self.next_track_id)
            track.update(
                bbox=detection['bbox'],
                frame_num=frame_num,
                timestamp=timestamp,
                embedding=np.array(detection.get('embedding')) if detection.get('embedding') else None,
                person_id=detection.get('person_id'),
                person_name=detection.get('person_name'),
                confidence=detection.get('similarity', 0.0) if detection.get('identified') else 0.0,
                age=detection.get('age'),
                gender=detection.get('gender')
            )
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1
        
        # Remove old tracks
        self._remove_old_tracks()
        
        # Return tracked faces
        tracked_faces = []
        for track_id, track in self.tracks.items():
            if len(track.frame_history) >= self.min_hits:
                tracked_faces.append({
                    'track_id': track_id,
                    'person_id': track.person_id,
                    'person_name': track.person_name,
                    'bbox': track.get_current_bbox(),
                    'frame': frame_num,
                    'timestamp': timestamp,
                    'age': track.age,
                    'gender': track.gender,
                    'confidence': track.confidence,
                    'identified': track.person_id is not None
                })
        
        return tracked_faces
    
    def _associate_detections_to_tracks(self, detections: List[Dict]) -> Tuple[List, List, List]:
        """
        Associate detections to existing tracks
        
        Returns:
            (matched, unmatched_dets, unmatched_tracks)
        """
        if not self.tracks:
            return [], list(range(len(detections))), []
        
        # Calculate cost matrix (IoU + embedding similarity)
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        track_ids = list(self.tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            track_bbox = track.get_current_bbox()
            
            if track_bbox is None:
                continue
            
            for j, detection in enumerate(detections):
                det_bbox = detection['bbox']
                
                # Calculate IoU
                iou = self._calculate_iou(track_bbox, det_bbox)
                
                # Calculate embedding similarity if available
                embedding_sim = 0.0
                if track.embedding is not None and detection.get('embedding'):
                    det_embedding = np.array(detection['embedding'])
                    track_norm = track.embedding / np.linalg.norm(track.embedding)
                    det_norm = det_embedding / np.linalg.norm(det_embedding)
                    embedding_sim = float(np.dot(track_norm, det_norm))
                
                # Combined cost (lower is better, so use 1 - similarity)
                cost = 1.0 - (0.5 * iou + 0.5 * embedding_sim)
                cost_matrix[i, j] = cost
        
        # Simple greedy matching
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))
        
        # Sort by cost
        matches = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                if cost_matrix[i, j] < (1.0 - self.iou_threshold):
                    matches.append((cost_matrix[i, j], i, j))
        
        matches.sort(key=lambda x: x[0])
        used_tracks = set()
        used_dets = set()
        
        for cost, i, j in matches:
            if i not in used_tracks and j not in used_dets:
                matched.append((track_ids[i], j))
                used_tracks.add(i)
                used_dets.add(j)
        
        unmatched_dets = [j for j in range(len(detections)) if j not in used_dets]
        unmatched_tracks = [i for i in range(len(track_ids)) if i not in used_tracks]
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _remove_old_tracks(self):
        """Remove tracks that haven't been seen recently"""
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if self.frame_count - track.last_seen_frame > self.max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_track_summary(self) -> List[Dict]:
        """Get summary of all tracks"""
        summary = []
        for track_id, track in self.tracks.items():
            if len(track.frame_history) >= self.min_hits:
                appearances = track.get_appearances()
                summary.append({
                    'track_id': track_id,
                    'person_id': track.person_id,
                    'person_name': track.person_name or f"Unknown (Track {track_id})",
                    'appearances': appearances,
                    'total_time_seconds': track.get_duration(),
                    'total_appearances': len(appearances),
                    'total_frames': len(track.frame_history),
                    'age': track.age,
                    'gender': track.gender,
                    'identified': track.person_id is not None
                })
        
        return summary
    
    def reset(self):
        """Reset tracker"""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0

