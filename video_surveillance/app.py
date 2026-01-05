"""
Video Surveillance Application
Real-time face detection and recognition from video streams
"""

import cv2
import numpy as np
import sys
import os
import logging
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.face_engine import FaceEngine
from src.database import FaceDatabase
from src.liveness_detector import LivenessDetector
from src.event_manager import EventManager, EventType
from src.utils import load_config, setup_logging
from stream_manager import StreamManager
from recorder import VideoRecorder

# Setup logging
logger = setup_logging()

# Load configuration
config = load_config()

# Initialize components
face_engine = FaceEngine(
    model_name=config.get('model', {}).get('name', 'buffalo_l'),
    det_size=tuple(config.get('model', {}).get('det_size', [640, 640])),
    det_thresh=config.get('model', {}).get('detection_threshold', 0.5)
)

db_config = config.get('database', {})
database = FaceDatabase(db_path=db_config.get('path', 'data/face_recognition.db'))

liveness_config = config.get('liveness', {})
liveness_detector = LivenessDetector(
    methods=liveness_config.get('methods', ['eye_blink', 'head_movement', 'texture_analysis']),
    threshold=liveness_config.get('threshold', 0.7)
)

event_manager = EventManager(config)
stream_manager = StreamManager()

# Video recorder
recording_config = config.get('events', {}).get('recording', {})
recorder = VideoRecorder(
    output_dir=recording_config.get('path', 'recordings'),
    duration=recording_config.get('duration', 30),
    fps=30
)


def process_frame(frame: np.ndarray, camera_id: str):
    """Process a single frame"""
    try:
        # Detect faces
        faces = face_engine.detect_faces(frame)
        
        if not faces:
            return frame
        
        # Process each face
        for face in faces:
            bbox = face['bbox']
            embedding = np.array(face['embedding'])
            
            # Search database
            threshold = config.get('model', {}).get('recognition_threshold', 0.6)
            matches = database.search_faces(embedding, threshold=threshold, max_results=1)
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            if matches:
                # Known face
                match = matches[0]
                color = (0, 255, 0)  # Green
                label = f"{match['person_name']} ({match['similarity']:.2f})"
                
                # Emit event
                event_manager.emit_event(
                    EventType.FACE_MATCHED,
                    {'person_name': match['person_name']},
                    person_id=match['person_id'],
                    camera_id=camera_id,
                    similarity=match['similarity']
                )
                
                # Start recording if configured
                if recording_config.get('on_match', False) and not recorder.is_recording:
                    recorder.start_recording()
            else:
                # Unknown face
                color = (0, 0, 255)  # Red
                label = "Unknown"
                
                # Emit event
                event_manager.emit_event(
                    EventType.UNKNOWN_FACE,
                    {},
                    camera_id=camera_id
                )
                
                # Start recording if configured
                if recording_config.get('on_unknown', False) and not recorder.is_recording:
                    recorder.start_recording()
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Liveness detection (optional)
            if liveness_config.get('enabled', True):
                liveness_result = liveness_detector.detect_liveness(
                    frame, bbox, face.get('kps'), f"{camera_id}_{id(face)}"
                )
                
                if not liveness_result['is_live']:
                    cv2.putText(frame, "SPOOF", (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add frame to recording if active
        if recorder.is_recording:
            recorder.add_frame(frame)
        
        return frame
    
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return frame


def main():
    """Main surveillance loop"""
    # Get camera configuration
    cameras_config = config.get('cameras', {})
    
    # Add default webcam if no cameras configured
    cameras = config.get('cameras_list', [])
    if not cameras:
        cameras = [{'id': 'webcam0', 'name': 'Webcam', 'source': '0', 'type': 'webcam'}]
    
    # Initialize streams
    for camera in cameras:
        stream_id = camera['id']
        source = camera.get('source', '0')
        stream_type = camera.get('type', 'auto')
        
        logger.info(f"Adding camera: {stream_id} ({source})")
        stream_manager.add_stream(stream_id, source, stream_type)
        stream_manager.start_stream(stream_id)
    
    # Main loop
    logger.info("Starting video surveillance...")
    try:
        while True:
            for stream_id in stream_manager.get_all_streams():
                frame = stream_manager.get_frame(stream_id, timeout=0.1)
                
                if frame is not None:
                    # Process frame
                    processed_frame = process_frame(frame, stream_id)
                    
                    # Display (optional)
                    cv2.imshow(f"Surveillance - {stream_id}", processed_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.01)  # Small delay
    
    except KeyboardInterrupt:
        logger.info("Stopping surveillance...")
    finally:
        # Cleanup
        stream_manager.stop_all()
        cv2.destroyAllWindows()
        if recorder.is_recording:
            recorder.stop_recording()
        logger.info("Surveillance stopped")


if __name__ == "__main__":
    main()

