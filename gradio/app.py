"""
Gradio web UI for Face Recognition System
"""

import gradio as gr
import cv2
import numpy as np
import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.face_engine import FaceEngine
from src.database import FaceDatabase
from src.liveness_detector import LivenessDetector
from src.face_analyzer import FaceAnalyzer
from src.face_tracker import FaceTracker
from src.video_annotator import VideoAnnotator
from src.utils import load_config, setup_logging
import pandas as pd

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

face_analyzer = FaceAnalyzer()


def compare_faces(image1, image2):
    """Compare two faces"""
    if image1 is None or image2 is None:
        return "Please upload both images", None
    
    try:
        # Convert Gradio images to OpenCV format
        img1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
        
        # Extract embeddings
        emb1 = face_engine.extract_embedding(img1)
        emb2 = face_engine.extract_embedding(img2)
        
        if emb1 is None:
            return "No face detected in image 1", None
        if emb2 is None:
            return "No face detected in image 2", None
        
        # Compare
        similarity = face_engine.compare_faces(emb1, emb2)
        threshold = config.get('model', {}).get('recognition_threshold', 0.6)
        is_match = similarity >= threshold
        
        # Draw results
        result_img1 = face_engine.draw_faces(img1, face_engine.detect_faces(img1))
        result_img2 = face_engine.draw_faces(img2, face_engine.detect_faces(img2))
        
        # Combine images
        combined = np.hstack([result_img1, result_img2])
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        
        result_text = f"Similarity: {similarity:.3f}\n"
        result_text += f"Threshold: {threshold}\n"
        result_text += f"Match: {'Yes' if is_match else 'No'}"
        
        return result_text, combined_rgb
    
    except Exception as e:
        logger.error(f"Error in compare_faces: {e}")
        return f"Error: {str(e)}", None


def analyze_face(image):
    """Analyze face attributes"""
    if image is None:
        return "Please upload an image", None
    
    try:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = face_engine.detect_faces(img)
        
        if not faces:
            return "No face detected", None
        
        # Analyze first face
        face = faces[0]
        analysis = face_analyzer.analyze_face(img, face['bbox'], face)
        
        # Draw face
        result_img = face_engine.draw_faces(img, [face], show_info=True)
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Format results
        result_text = "Face Analysis Results:\n\n"
        result_text += f"Age: {analysis.get('age', 'N/A')}\n"
        result_text += f"Gender: {analysis.get('gender_label', 'N/A')}\n"
        result_text += f"Emotion: {analysis.get('emotion', {}).get('emotion', 'N/A')}\n"
        result_text += f"Mask: {'Yes' if analysis.get('mask', {}).get('has_mask', False) else 'No'}\n"
        
        quality = analysis.get('quality', {})
        result_text += f"\nQuality Score: {quality.get('overall_score', 0):.2f}\n"
        result_text += f"Blur: {'Yes' if quality.get('blur', {}).get('is_blurry', False) else 'No'}\n"
        
        return result_text, result_rgb
    
    except Exception as e:
        logger.error(f"Error in analyze_face: {e}")
        return f"Error: {str(e)}", None


def detect_liveness(image):
    """Detect face liveness"""
    if image is None:
        return "Please upload an image or use webcam", None
    
    try:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect face
        faces = face_engine.detect_faces(img)
        
        if not faces:
            return "No face detected", None
        
        face = faces[0]
        
        # Detect liveness
        result = liveness_detector.detect_liveness(
            image=img,
            face_bbox=face['bbox'],
            face_kps=face.get('kps'),
            face_id=f"gradio_{id(img)}"
        )
        
        # Draw face
        result_img = face_engine.draw_faces(img, [face])
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Format results
        result_text = "Liveness Detection Results:\n\n"
        result_text += f"Is Live: {'Yes' if result['is_live'] else 'No'}\n"
        result_text += f"Confidence: {result['confidence']:.3f}\n\n"
        result_text += "Method Scores:\n"
        for method, score in result['methods'].items():
            result_text += f"  {method}: {score:.3f}\n"
        
        return result_text, result_rgb
    
    except Exception as e:
        logger.error(f"Error in detect_liveness: {e}")
        return f"Error: {str(e)}", None


def identify_face(image):
    """Identify face from database"""
    if image is None:
        return "Please upload an image", None
    
    try:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Extract embedding
        embedding = face_engine.extract_embedding(img)
        
        if embedding is None:
            return "No face detected", None
        
        # Search database
        threshold = config.get('model', {}).get('recognition_threshold', 0.6)
        matches = database.search_faces(embedding, threshold=threshold, max_results=5)
        
        # Draw face
        faces = face_engine.detect_faces(img)
        result_img = face_engine.draw_faces(img, faces)
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Format results
        if matches:
            result_text = f"Found {len(matches)} match(es):\n\n"
            for i, match in enumerate(matches, 1):
                result_text += f"{i}. {match['person_name']}\n"
                result_text += f"   Similarity: {match['similarity']:.3f}\n"
                result_text += f"   Person ID: {match['person_id']}\n\n"
        else:
            result_text = "No matches found in database"
        
        return result_text, result_rgb
    
    except Exception as e:
        logger.error(f"Error in identify_face: {e}")
        return f"Error: {str(e)}", None


def identify_video(video, frame_skip, identify_people, generate_annotated):
    """Identify people in a video"""
    if video is None:
        return "Please upload a video file", None, None, None
    
    try:
        import tempfile
        from pathlib import Path
        
        # Log what we received for debugging
        logger.info(f"Video input type: {type(video)}, value: {video}")
        
        # Handle Gradio File component output (returns file path string)
        # Also handle Video component output for backward compatibility
        video_path = None
        
        if isinstance(video, str):
            # File component returns string path directly
            video_path = video
        elif isinstance(video, tuple) and len(video) > 0:
            # Video component tuple format: (video_path, subtitle_path)
            video_path = video[0]
        elif isinstance(video, dict):
            # Dictionary format with metadata
            video_path = video.get('video', video.get('path', video.get('name', None)))
        elif hasattr(video, 'name'):
            # File-like object with name attribute
            video_path = video.name
        else:
            error_msg = f"Unsupported video format: {type(video)}. Please upload a valid video file."
            logger.error(error_msg)
            return error_msg, None, None, None
        
        if video_path is None:
            error_msg = "Could not extract video path from upload"
            logger.error(error_msg)
            return error_msg, None, None, None
        
        # Convert to Path object
        temp_video = Path(video_path)
        
        # Check if file exists
        if not temp_video.exists():
            error_msg = f"Video file not found: {video_path}"
            logger.error(error_msg)
            return error_msg, None, None, None
        
        # Verify it's a file (not directory)
        if not temp_video.is_file():
            error_msg = f"Path is not a file: {video_path}"
            logger.error(error_msg)
            return error_msg, None, None, None
        
        logger.info(f"Processing video: {temp_video} (size: {temp_video.stat().st_size} bytes)")
        
        # Open video
        cap = cv2.VideoCapture(str(temp_video))
        if not cap.isOpened():
            return f"Could not open video file: {temp_video}. Please ensure the file is a valid video format (MP4, AVI, MOV, etc.)", None, None, None
        
        # Validate video can be read
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            cap.release()
            return "Video file appears corrupted or empty. Please try a different video file.", None, None, None
        
        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Initialize tracker
        tracker = FaceTracker() if identify_people else None
        
        frame_count = 0
        processed_count = 0
        detections_by_frame = {}
        max_frames = min(1000, total_frames)  # Limit processing
        
        # Process video
        while processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            timestamp = frame_count / fps
            
            # Detect faces
            faces = face_engine.detect_faces(frame)
            
            if faces:
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
                    
                    # Identify if requested
                    if identify_people:
                        embedding = np.array(face['embedding'])
                        threshold = config.get('model', {}).get('recognition_threshold', 0.6)
                        matches = database.search_faces(embedding, threshold=threshold, max_results=1)
                        
                        if matches:
                            face_data['identified'] = True
                            face_data['person_name'] = matches[0]['person_name']
                            face_data['person_id'] = matches[0]['person_id']
                            face_data['similarity'] = matches[0]['similarity']
                        else:
                            face_data['identified'] = False
                    
                    frame_detections.append(face_data)
                
                # Update tracker
                if tracker:
                    tracked = tracker.update(frame_detections, frame_count, timestamp)
                    for i, track in enumerate(tracked):
                        if i < len(frame_detections):
                            frame_detections[i]['track_id'] = track['track_id']
                            if track.get('person_id'):
                                frame_detections[i]['person_id'] = track['person_id']
                                frame_detections[i]['person_name'] = track['person_name']
                                frame_detections[i]['identified'] = True
                
                detections_by_frame[frame_count] = frame_detections
            
            processed_count += 1
        
        cap.release()
        
        # Get results
        if tracker:
            people_summary = tracker.get_track_summary()
        else:
            # Group by person_id
            people_dict = {}
            for frame_num, detections in detections_by_frame.items():
                for face in detections:
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
                            'frame': frame_num
                        })
                        people_dict[person_id]['total_frames'] += 1
            people_summary = list(people_dict.values())
        
        # Format results text
        result_text = f"Video Analysis Results:\n\n"
        result_text += f"Video Duration: {duration:.2f} seconds\n"
        result_text += f"Frames Processed: {processed_count}\n"
        result_text += f"Total Faces Detected: {sum(len(d) for d in detections_by_frame.values())}\n\n"
        
        if people_summary:
            result_text += f"People Identified: {len(people_summary)}\n\n"
            for person in people_summary:
                result_text += f"• {person.get('person_name', 'Unknown')}\n"
                result_text += f"  Total Time: {person.get('total_time_seconds', 0):.2f}s\n"
                result_text += f"  Appearances: {person.get('total_appearances', 0)}\n"
                result_text += f"  Frames: {person.get('total_frames', 0)}\n\n"
        else:
            result_text += "No people identified in video.\n"
            result_text += "Make sure you have registered faces in the database.\n"
        
        # Create results dataframe
        if people_summary:
            df_data = []
            for person in people_summary:
                df_data.append({
                    'Name': person.get('person_name', 'Unknown'),
                    'Total Time (s)': f"{person.get('total_time_seconds', 0):.2f}",
                    'Appearances': person.get('total_appearances', 0),
                    'Frames': person.get('total_frames', 0),
                    'Identified': 'Yes' if person.get('identified') else 'No'
                })
            results_df = pd.DataFrame(df_data)
        else:
            results_df = pd.DataFrame(columns=['Name', 'Total Time (s)', 'Appearances', 'Frames', 'Identified'])
        
        # Generate annotated video if requested
        annotated_video = None
        if generate_annotated and detections_by_frame:
            try:
                output_dir = Path('uploads/annotated')
                output_dir.mkdir(parents=True, exist_ok=True)
                from datetime import datetime
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                annotated_path = output_dir / f"annotated_{timestamp_str}.mp4"
                
                annotator = VideoAnnotator(show_trajectories=identify_people, show_timestamps=True)
                annotator.annotate_video(
                    str(temp_video),
                    detections_by_frame,
                    str(annotated_path),
                    fps=fps
                )
                
                # Verify file exists and is readable
                if annotated_path.exists() and annotated_path.stat().st_size > 0:
                    annotated_video = str(annotated_path.absolute())  # Use absolute path for Gradio
                else:
                    logger.warning(f"Annotated video was not created properly: {annotated_path}")
                    result_text += "\n\nWarning: Could not generate annotated video. The video file may not be compatible."
            except Exception as e:
                logger.error(f"Error generating annotated video: {e}")
                result_text += f"\n\nWarning: Error generating annotated video: {str(e)}"
        
        return result_text, results_df, annotated_video, None
    
    except Exception as e:
        logger.error(f"Error in identify_video: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", None, None, None


# Create Gradio interface
with gr.Blocks(title="Face Recognition System") as demo:
    gr.Markdown("# Face Recognition System")
    gr.Markdown("Upload images or use webcam to test face recognition, liveness detection, and face analysis.")
    
    with gr.Tabs():
        with gr.Tab("Face Comparison"):
            gr.Markdown("### Compare two faces (1:1 verification)")
            with gr.Row():
                img1_input = gr.Image(label="Image 1", type="pil")
                img2_input = gr.Image(label="Image 2", type="pil")
            compare_btn = gr.Button("Compare Faces")
            compare_output = gr.Textbox(label="Result")
            compare_image = gr.Image(label="Result Images")
            compare_btn.click(compare_faces, inputs=[img1_input, img2_input], outputs=[compare_output, compare_image])
        
        with gr.Tab("Face Analysis"):
            gr.Markdown("### Analyze face attributes")
            analyze_input = gr.Image(label="Upload Image", type="pil")
            analyze_btn = gr.Button("Analyze Face")
            analyze_output = gr.Textbox(label="Analysis Results")
            analyze_image = gr.Image(label="Detected Face")
            analyze_btn.click(analyze_face, inputs=[analyze_input], outputs=[analyze_output, analyze_image])
        
        with gr.Tab("Liveness Detection"):
            gr.Markdown("### Detect if face is live")
            liveness_input = gr.Image(label="Upload Image or Use Webcam", type="pil", sources=["upload", "webcam"])
            liveness_btn = gr.Button("Detect Liveness")
            liveness_output = gr.Textbox(label="Liveness Results")
            liveness_image = gr.Image(label="Detected Face")
            liveness_btn.click(detect_liveness, inputs=[liveness_input], outputs=[liveness_output, liveness_image])
        
        with gr.Tab("Face Identification"):
            gr.Markdown("### Identify face from database (1:N search)")
            identify_input = gr.Image(label="Upload Image", type="pil")
            identify_btn = gr.Button("Identify Face")
            identify_output = gr.Textbox(label="Identification Results")
            identify_image = gr.Image(label="Detected Face")
            identify_btn.click(identify_face, inputs=[identify_input], outputs=[identify_output, identify_image])
        
        with gr.Tab("Video Identification"):
            gr.Markdown("### Identify people in a video")
            gr.Markdown("Upload a video file to detect and identify all people appearing in it.")
            
            with gr.Row():
                with gr.Column():
                    video_input = gr.File(
                        label="Upload Video File", 
                        file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                        type="filepath"
                    )
                    with gr.Row():
                        frame_skip_slider = gr.Slider(
                            minimum=1, maximum=60, value=30, step=1,
                            label="Frame Skip (process every Nth frame)",
                            info="Higher values = faster processing, less accuracy"
                        )
                    with gr.Row():
                        identify_checkbox = gr.Checkbox(
                            label="Identify People", value=True,
                            info="Match faces against database"
                        )
                        annotated_checkbox = gr.Checkbox(
                            label="Generate Annotated Video", value=False,
                            info="Create video with bounding boxes and labels"
                        )
                    video_btn = gr.Button("Process Video", variant="primary")
                
                with gr.Column():
                    video_output = gr.Textbox(label="Analysis Results", lines=15)
                    video_table = gr.Dataframe(
                        label="People Summary",
                        headers=["Name", "Total Time (s)", "Appearances", "Frames", "Identified"]
                    )
                    annotated_video_output = gr.Video(label="Annotated Video (if generated)")
            
            video_btn.click(
                identify_video,
                inputs=[video_input, frame_skip_slider, identify_checkbox, annotated_checkbox],
                outputs=[video_output, video_table, annotated_video_output, gr.Textbox(visible=False)]
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

