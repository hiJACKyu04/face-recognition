"""
Face recognition API routes
"""

import cv2
import numpy as np
from flask import Blueprint, request, jsonify, g
from werkzeug.utils import secure_filename
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.face_engine import FaceEngine
from src.database import FaceDatabase
from src.face_analyzer import FaceAnalyzer
from src.face_tracker import FaceTracker
from src.video_annotator import VideoAnnotator
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from middleware import require_api_key

logger = logging.getLogger(__name__)

face_bp = Blueprint('face', __name__)


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


def get_database():
    """Get database instance"""
    if 'database' not in g:
        db_config = g.config.get('database', {})
        db_path = db_config.get('path', 'data/face_recognition.db')
        g.database = FaceDatabase(db_path)
    return g.database


def get_face_analyzer():
    """Get face analyzer instance"""
    if 'face_analyzer' not in g:
        g.face_analyzer = FaceAnalyzer()
    return g.face_analyzer


def decode_image(file) -> np.ndarray:
    """Decode uploaded image file"""
    try:
        # Read file content
        file_bytes = file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        return image
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise


@face_bp.route('/face/detect', methods=['POST'])
@require_api_key
def detect_faces():
    """Detect faces in an image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        image = decode_image(file)
        
        face_engine = get_face_engine()
        faces = face_engine.detect_faces(image)
        
        return jsonify({
            'success': True,
            'faces_detected': len(faces),
            'faces': faces
        })
    
    except Exception as e:
        logger.error(f"Error in detect_faces: {e}")
        return jsonify({'error': str(e)}), 500


@face_bp.route('/face/compare', methods=['POST'])
@require_api_key
def compare_faces():
    """Compare two faces (1:1 verification)"""
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Two image files required (image1, image2)'}), 400
        
        image1 = decode_image(request.files['image1'])
        image2 = decode_image(request.files['image2'])
        
        face_engine = get_face_engine()
        
        # Extract embeddings
        emb1 = face_engine.extract_embedding(image1)
        emb2 = face_engine.extract_embedding(image2)
        
        if emb1 is None:
            return jsonify({'error': 'No face detected in image1'}), 400
        if emb2 is None:
            return jsonify({'error': 'No face detected in image2'}), 400
        
        # Compare
        similarity = face_engine.compare_faces(emb1, emb2)
        threshold = g.config.get('model', {}).get('recognition_threshold', 0.6)
        is_match = similarity >= threshold
        
        return jsonify({
            'success': True,
            'similarity': float(similarity),
            'is_match': is_match,
            'threshold': threshold
        })
    
    except Exception as e:
        logger.error(f"Error in compare_faces: {e}")
        return jsonify({'error': str(e)}), 500


@face_bp.route('/face/analyze', methods=['POST'])
@require_api_key
def analyze_face():
    """Analyze face attributes"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        image = decode_image(file)
        
        face_engine = get_face_engine()
        face_analyzer = get_face_analyzer()
        
        # Detect faces
        faces = face_engine.detect_faces(image)
        
        if not faces:
            return jsonify({'error': 'No face detected'}), 400
        
        # Analyze first face
        face = faces[0]
        analysis = face_analyzer.analyze_face(image, face['bbox'], face)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'face': face
        })
    
    except Exception as e:
        logger.error(f"Error in analyze_face: {e}")
        return jsonify({'error': str(e)}), 500


@face_bp.route('/face/identify', methods=['POST'])
@require_api_key
def identify_face():
    """Identify face from database (1:N search)"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        image = decode_image(file)
        
        threshold = float(request.form.get('threshold', g.config.get('model', {}).get('recognition_threshold', 0.6)))
        max_results = int(request.form.get('max_results', 10))
        
        face_engine = get_face_engine()
        database = get_database()
        
        # Extract embedding
        embedding = face_engine.extract_embedding(image)
        
        if embedding is None:
            return jsonify({'error': 'No face detected'}), 400
        
        # Search database
        matches = database.search_faces(embedding, threshold=threshold, max_results=max_results)
        
        return jsonify({
            'success': True,
            'matches_found': len(matches),
            'matches': matches
        })
    
    except Exception as e:
        logger.error(f"Error in identify_face: {e}")
        return jsonify({'error': str(e)}), 500


@face_bp.route('/face/register', methods=['POST'])
@require_api_key
def register_face():
    """Register a new face to database"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        name = request.form.get('name', 'Unknown')
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        
        file = request.files['image']
        image = decode_image(file)
        
        face_engine = get_face_engine()
        database = get_database()
        
        # Detect and extract face
        faces = face_engine.detect_faces(image)
        if not faces:
            return jsonify({'error': 'No face detected'}), 400
        
        face = faces[0]
        embedding = np.array(face['embedding'])
        
        # Create person
        person_id = database.add_person(name)
        
        # Save face
        face_id = database.add_face(
            person_id=person_id,
            embedding=embedding,
            bbox=face['bbox'],
            age=face.get('age'),
            gender=face.get('gender')
        )
        
        return jsonify({
            'success': True,
            'person_id': person_id,
            'face_id': face_id,
            'name': name
        })
    
    except Exception as e:
        logger.error(f"Error in register_face: {e}")
        return jsonify({'error': str(e)}), 500


@face_bp.route('/face/list', methods=['GET'])
@require_api_key
def list_faces():
    """List all registered faces"""
    try:
        database = get_database()
        persons = database.list_persons()
        
        result = []
        for person in persons:
            faces = database.get_person_faces(person['id'])
            result.append({
                'person_id': person['id'],
                'name': person['name'],
                'faces_count': len(faces),
                'created_at': person['created_at']
            })
        
        return jsonify({
            'success': True,
            'persons': result,
            'total': len(result)
        })
    
    except Exception as e:
        logger.error(f"Error in list_faces: {e}")
        return jsonify({'error': str(e)}), 500


@face_bp.route('/face/<int:person_id>', methods=['DELETE'])
@require_api_key
def delete_face(person_id):
    """Delete a person and all associated faces"""
    try:
        database = get_database()
        success = database.delete_person(person_id)
        
        if success:
            return jsonify({'success': True, 'message': f'Person {person_id} deleted'})
        else:
            return jsonify({'error': 'Person not found'}), 404
    
    except Exception as e:
        logger.error(f"Error in delete_face: {e}")
        return jsonify({'error': str(e)}), 500


@face_bp.route('/face/detect_video', methods=['POST'])
@require_api_key
def detect_faces_video():
    """Detect and identify faces in a video file with tracking"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        # Get parameters
        frame_skip = int(request.form.get('frame_skip', 30))  # Process every Nth frame
        max_frames = int(request.form.get('max_frames', 1000))  # Maximum frames to process
        identify = request.form.get('identify', 'true').lower() == 'true'  # Whether to identify faces
        track_faces = request.form.get('track_faces', 'true').lower() == 'true'  # Enable face tracking
        generate_annotated_video = request.form.get('generate_annotated_video', 'false').lower() == 'true'
        
        # Save uploaded file temporarily
        import tempfile
        import os
        from pathlib import Path
        from datetime import datetime
        
        # Create temp directory if it doesn't exist
        temp_dir = Path('uploads/temp')
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_dir = Path('uploads/annotated')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save video file
        filename = secure_filename(file.filename)
        temp_path = temp_dir / filename
        file.save(str(temp_path))
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(temp_path))
            
            if not cap.isOpened():
                return jsonify({'error': 'Could not open video file'}), 400
            
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            face_engine = get_face_engine()
            database = get_database() if identify else None
            
            # Initialize tracker if enabled
            tracker = FaceTracker() if track_faces else None
            
            results = {
                'video_info': {
                    'fps': float(fps),
                    'total_frames': total_frames,
                    'duration_seconds': float(duration),
                    'filename': filename
                },
                'detections': [],
                'people': [],
                'summary': {
                    'frames_processed': 0,
                    'total_faces_detected': 0,
                    'unique_people': 0,
                    'identified_people': 0
                }
            }
            
            frame_count = 0
            processed_count = 0
            detections_by_frame = {}  # For video annotation
            
            # Process video frames
            while processed_count < max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames based on frame_skip parameter
                if frame_count % frame_skip != 0:
                    continue
                
                # Detect faces in frame
                faces = face_engine.detect_faces(frame)
                
                if faces:
                    timestamp = frame_count / fps if fps > 0 else frame_count
                    
                    frame_detections = []
                    for face in faces:
                        face_data = {
                            'bbox': face['bbox'],
                            'det_score': face['det_score'],
                            'age': face.get('age'),
                            'gender': face.get('gender'),
                            'timestamp': float(timestamp),
                            'embedding': face.get('embedding')
                        }
                        
                        # If identification is requested, search database
                        if identify and database:
                            embedding = np.array(face['embedding'])
                            threshold = g.config.get('model', {}).get('recognition_threshold', 0.6)
                            matches = database.search_faces(embedding, threshold=threshold, max_results=1)
                            
                            if matches:
                                face_data['identified'] = True
                                face_data['person_name'] = matches[0]['person_name']
                                face_data['person_id'] = matches[0]['person_id']
                                face_data['similarity'] = matches[0]['similarity']
                            else:
                                face_data['identified'] = False
                        
                        frame_detections.append(face_data)
                    
                    # Update tracker if enabled
                    if tracker:
                        tracked_faces = tracker.update(frame_detections, frame_count, timestamp)
                        # Add track_id to detections
                        for i, tracked in enumerate(tracked_faces):
                            if i < len(frame_detections):
                                frame_detections[i]['track_id'] = tracked['track_id']
                                # Update with tracked person info
                                if tracked.get('person_id'):
                                    frame_detections[i]['person_id'] = tracked['person_id']
                                    frame_detections[i]['person_name'] = tracked['person_name']
                                    frame_detections[i]['identified'] = True
                    
                    if frame_detections:
                        results['detections'].append({
                            'frame_number': frame_count,
                            'timestamp': float(timestamp),
                            'faces': frame_detections
                        })
                        detections_by_frame[frame_count] = frame_detections
                        results['summary']['total_faces_detected'] += len(frame_detections)
                
                processed_count += 1
            
            # Get track summary if tracking was enabled
            if tracker:
                track_summary = tracker.get_track_summary()
                results['people'] = track_summary
                results['summary']['unique_people'] = len(track_summary)
                results['summary']['identified_people'] = sum(1 for p in track_summary if p.get('identified', False))
            else:
                # Group by person_id if identified
                if identify:
                    people_dict = {}
                    for detection_group in results['detections']:
                        for face in detection_group['faces']:
                            if face.get('identified') and face.get('person_id'):
                                person_id = face['person_id']
                                if person_id not in people_dict:
                                    people_dict[person_id] = {
                                        'person_id': person_id,
                                        'person_name': face.get('person_name', 'Unknown'),
                                        'appearances': [],
                                        'total_frames': 0
                                    }
                                people_dict[person_id]['appearances'].append({
                                    'timestamp': face['timestamp'],
                                    'frame': detection_group['frame_number']
                                })
                                people_dict[person_id]['total_frames'] += 1
                    
                    results['people'] = list(people_dict.values())
                    results['summary']['unique_people'] = len(people_dict)
                    results['summary']['identified_people'] = len(people_dict)
            
            results['summary']['frames_processed'] = processed_count
            
            # Generate annotated video if requested
            annotated_video_path = None
            if generate_annotated_video and detections_by_frame:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                annotated_filename = f"annotated_{timestamp_str}_{filename}"
                annotated_path = output_dir / annotated_filename
                
                annotator = VideoAnnotator(show_trajectories=track_faces, show_timestamps=True)
                
                # Reopen video for annotation
                cap.release()
                cap = cv2.VideoCapture(str(temp_path))
                
                annotator.annotate_video(
                    str(temp_path),
                    detections_by_frame,
                    str(annotated_path),
                    fps=fps,
                    track_data=tracker.get_track_summary() if tracker else None
                )
                
                annotated_video_path = str(annotated_path)
                results['annotated_video_path'] = annotated_video_path
            
            cap.release()
            
            return jsonify({
                'success': True,
                'results': results
            })
        
        finally:
            # Clean up temp file
            if temp_path.exists():
                os.remove(str(temp_path))
    
    except Exception as e:
        logger.error(f"Error in detect_faces_video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

