"""
Video annotation module
Generates annotated videos with identified people
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict 

logger = logging.getLogger(__name__)


class VideoAnnotator:
    """Annotate video with face detection and identification results"""
    
    def __init__(self, show_trajectories: bool = False, show_timestamps: bool = True):
        """
        Initialize video annotator
        
        Args:
            show_trajectories: Draw face movement trajectories
            show_timestamps: Display timestamps on frames
        """
        self.show_trajectories = show_trajectories
        self.show_timestamps = show_timestamps
        logger.info("Video annotator initialized")
    
    def annotate_video(self, video_path: str, detections_by_frame: Dict[int, List[Dict]],
                      output_path: str, fps: float = 30.0, 
                      track_data: Optional[Dict] = None) -> str:
        """
        Create annotated video from detections
        
        Args:
            video_path: Input video path
            detections_by_frame: Dictionary mapping frame numbers to detections
            output_path: Output video path
            fps: Video FPS
            track_data: Optional tracking data for trajectories
            
        Returns:
            Path to annotated video
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid video dimensions: {width}x{height}")
        
        # Try different codecs for web compatibility
        # H.264 is most compatible, fallback to mp4v
        codecs_to_try = [
            ('avc1', 'H.264/AVC1'),
            ('h264', 'H.264'),
            ('mp4v', 'MPEG-4'),
        ]
        
        out = None
        used_codec = "unknown"
        for codec_name, codec_desc in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))
                if out.isOpened():
                    used_codec = codec_desc
                    logger.info(f"Using codec: {codec_desc} ({codec_name})")
                    break
                else:
                    out.release()
                    out = None
            except Exception as e:
                logger.warning(f"Failed to use codec {codec_name}: {e}")
                if out:
                    out.release()
                    out = None
        
        if out is None or not out.isOpened():
            raise RuntimeError(f"Could not create video writer for {output_path}. Tried codecs: {[c[0] for c in codecs_to_try]}")
        
        frame_num = 0
        trajectory_points = defaultdict(list)  # track_id -> list of (x, y) points
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                timestamp = frame_num / video_fps
                
                # Get detections for this frame
                detections = detections_by_frame.get(frame_num, [])
                
                # Draw trajectories if enabled
                if self.show_trajectories and track_data:
                    self._draw_trajectories(frame, trajectory_points, frame_num)
                
                # Draw detections
                for detection in detections:
                    self._draw_detection(frame, detection, timestamp)
                    
                    # Update trajectory
                    if self.show_trajectories and detection.get('track_id'):
                        bbox = detection['bbox']
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        trajectory_points[detection['track_id']].append((center_x, center_y))
                
                # Draw timestamp
                if self.show_timestamps:
                    cv2.putText(frame, f"Time: {timestamp:.2f}s", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if frame is not None and frame.size > 0:
                    out.write(frame)
            
            # Ensure all frames are written
            out.release()
            
            # Verify video was created
            if not Path(output_path).exists():
                raise RuntimeError(f"Annotated video was not created: {output_path}")
            
            file_size = Path(output_path).stat().st_size
            if file_size == 0:
                raise RuntimeError(f"Annotated video is empty: {output_path}")
            
            codec_info = used_codec if used_codec else "unknown"
            logger.info(f"Annotated video saved: {output_path} (size: {file_size} bytes, codec: {codec_info})")
            return output_path
        
        except Exception as e:
            logger.error(f"Error creating annotated video: {e}")
            if out and out.isOpened():
                out.release()
            # Clean up partial file
            if Path(output_path).exists():
                Path(output_path).unlink()
            raise
        finally:
            cap.release()
    
    def _draw_detection(self, frame: np.ndarray, detection: Dict, timestamp: float):
        """Draw a single detection on frame"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Determine color and label
        if detection.get('identified', False):
            color = (0, 255, 0)  # Green for identified
            person_name = detection.get('person_name', 'Unknown')
            similarity = detection.get('similarity', 0.0)
            label = f"{person_name} ({similarity:.2f})"
        else:
            color = (0, 0, 255)  # Red for unknown
            label = "Unknown"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(frame, (x1, y1 - text_height - 10),
                     (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw additional info
        if detection.get('age'):
            age_text = f"Age: {detection['age']}"
            cv2.putText(frame, age_text, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if detection.get('track_id'):
            track_text = f"Track: {detection['track_id']}"
            cv2.putText(frame, track_text, (x1, y2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_trajectories(self, frame: np.ndarray, trajectory_points: Dict, current_frame: int):
        """Draw face movement trajectories"""
        for track_id, points in trajectory_points.items():
            if len(points) < 2:
                continue
            
            # Draw trajectory line
            points_array = np.array(points, dtype=np.int32)
            cv2.polylines(frame, [points_array], False, (255, 255, 0), 2)
            
            # Draw current position
            if points:
                cv2.circle(frame, points[-1], 5, (255, 255, 0), -1)

