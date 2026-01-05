# Video Person Identification Guide

## Overview

The video identification feature allows you to upload a video and automatically identify all people appearing in it. The system uses face tracking to follow the same person across multiple frames, avoiding duplicate identifications.

## Features

- **Face Detection**: Detects all faces in video frames
- **Face Tracking**: Tracks the same person across frames using IoU and embedding similarity
- **Person Identification**: Matches detected faces against registered faces in database
- **Annotated Video**: Generates output video with bounding boxes and person names
- **Statistics**: Provides detailed statistics about each person's appearance

## Usage

### Via API

**Endpoint:** `POST /api/v1/face/detect_video`

**Parameters:**
- `video` (required): Video file (MP4, AVI, MOV, etc.)
- `frame_skip` (optional, default: 30): Process every Nth frame
- `max_frames` (optional, default: 1000): Maximum frames to process
- `identify` (optional, default: true): Enable person identification
- `track_faces` (optional, default: true): Enable face tracking
- `generate_annotated_video` (optional, default: false): Generate annotated output video

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/face/detect_video \
  -F "video=@/path/to/video.mp4" \
  -F "identify=true" \
  -F "track_faces=true" \
  -F "generate_annotated_video=true" \
  -F "frame_skip=30"
```

**Response:**
```json
{
  "success": true,
  "results": {
    "video_info": {
      "fps": 30.0,
      "total_frames": 900,
      "duration_seconds": 30.0,
      "filename": "video.mp4"
    },
    "people": [
      {
        "person_id": 1,
        "person_name": "John Doe",
        "appearances": [
          {
            "start_time": 1.0,
            "end_time": 5.0,
            "frames": [30, 60, 90]
          }
        ],
        "total_time_seconds": 4.0,
        "total_appearances": 1,
        "total_frames": 3,
        "identified": true
      }
    ],
    "annotated_video_path": "uploads/annotated/annotated_20260106_123456_video.mp4",
    "summary": {
      "frames_processed": 30,
      "total_faces_detected": 45,
      "unique_people": 3,
      "identified_people": 2
    }
  }
}
```

### Via Gradio UI

1. Start Gradio UI:
   ```bash
   source venv/bin/activate
   python gradio/app.py
   ```

2. Open browser: http://localhost:7860

3. Go to "Video Identification" tab

4. Upload a video file

5. Configure options:
   - **Frame Skip**: How many frames to skip (higher = faster)
   - **Identify People**: Enable/disable identification
   - **Generate Annotated Video**: Create output video with labels

6. Click "Process Video"

7. View results:
   - Text summary with statistics
   - Table showing all identified people
   - Annotated video (if generated)

## Face Tracking

The system uses advanced tracking to follow the same person across frames:

- **IoU Matching**: Matches bounding boxes using Intersection over Union
- **Embedding Similarity**: Uses face embeddings for identity matching
- **Track Management**: Maintains track history and handles occlusions
- **Appearance Segments**: Tracks continuous appearances of each person

## Configuration

Edit `config/config.yaml` to customize tracking:

```yaml
tracking:
  enabled: true
  iou_threshold: 0.3
  embedding_threshold: 0.7
  max_age: 30
  min_hits: 3
```

## Tips

1. **Register faces first**: Make sure you have registered faces in the database before identifying
2. **Frame skip**: Use higher values (30-60) for faster processing on long videos
3. **Quality**: Lower frame_skip values provide better accuracy but slower processing
4. **Annotated video**: Enable this to get a visual output with bounding boxes and names

## Performance

- Processing speed depends on video length and frame_skip setting
- Typical processing: ~1-2 seconds per 100 frames
- Face tracking adds minimal overhead (~10-15%)
- Annotated video generation adds ~20-30% processing time

