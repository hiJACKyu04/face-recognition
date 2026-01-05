# Architecture Documentation

## System Overview

The Face Recognition System is a comprehensive solution for face detection, recognition, liveness detection, and video surveillance.

## Components

### Core Engine

- **FaceEngine**: Face detection and recognition using RetinaFace buffalo_l
- **LivenessDetector**: Liveness detection using multiple methods
- **FaceAnalyzer**: Face attribute analysis (age, gender, emotion, mask)
- **Database**: SQLite/PostgreSQL for face storage and events
- **StreamProcessor**: Video stream processing (RTSP, webcam, files)
- **EventManager**: Event handling and alerting

### Services

- **Flask API**: RESTful API for face recognition operations
- **Gradio UI**: Interactive web UI for testing
- **Video Surveillance**: Real-time face detection from video streams
- **Dashboard**: Web dashboard for monitoring and management
- **CLI**: Command-line interface for system management

## Data Flow

1. **Input**: Images or video streams
2. **Processing**: Face detection → Embedding extraction → Recognition/Liveness
3. **Storage**: Face embeddings and events stored in database
4. **Output**: API responses, alerts, recordings

## Model Architecture

- **Model**: RetinaFace buffalo_l
- **Provider**: InsightFace
- **Embedding Size**: 512 dimensions
- **Detection**: RetinaFace detector
- **Recognition**: ArcFace embeddings

## Database Schema

- **persons**: Person information
- **faces**: Face embeddings and metadata
- **events**: Event log
- **cameras**: Camera configuration

## API Architecture

- RESTful design
- Versioned endpoints (`/api/v1/`)
- Authentication via API keys
- Rate limiting
- CORS support

## Deployment

- **Local**: Python virtual environment
- **Docker**: Containerized deployment
- **Production**: Multi-container setup with separate services

## Performance

- **Detection**: Real-time capable
- **Recognition**: Fast 1:1 and 1:N matching
- **Streaming**: Optimized frame processing
- **Scalability**: Horizontal scaling support

