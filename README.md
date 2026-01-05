# Face Recognition System

A comprehensive cross-platform face recognition system with Flask API, Gradio UI, Video Surveillance, CLI interface, and Web Dashboard. Built using InsightFace with RetinaFace buffalo_l model.

## Features

- **Face Recognition**: 1:1 verification and 1:N identification using RetinaFace buffalo_l
- **Liveness Detection**: Eye blink detection, head movement analysis, and anti-spoofing
- **Face Attribute Analysis**: Age, gender, emotion, and mask detection
- **Video Surveillance**: Real-time face detection from webcam, RTSP streams, and video files
- **RESTful API**: Complete Flask API with authentication and rate limiting
- **Web Interfaces**: Gradio UI for testing and Web Dashboard for monitoring
- **CLI Tools**: Command-line interface for system management

## System Requirements

- Python 3.8+
- 8GB RAM (recommended)
- CPU or GPU (CUDA/MPS/Apple Silicon supported)
- macOS, Windows, or Linux

## Installation

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd face-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy configuration template
cp config/config.example.yaml config/config.yaml

# Run setup script
bash scripts/setup.sh
```

### Docker Installation  (Optional)

```bash
# Development
docker-compose -f docker/docker-compose.yml up

# Production
docker-compose -f docker/docker-compose.prod.yml up
```

## Usage

### Start Services

```bash
# Using CLI
face-recognition start

# Or individually
python flask/app.py          # Flask API (port 8000)
python gradio/app.py         # Gradio UI (port 7860)
python video_surveillance/app.py  # Video Surveillance
python dashboard/app.py      # Web Dashboard (port 5000)
```

### CLI Commands

```bash
face-recognition start              # Start all services
face-recognition stop               # Stop all services
face-recognition register-face      # Register a new face
face-recognition list-faces         # List registered faces
face-recognition status             # System status
```

### API Endpoints

- `POST /api/v1/face/compare` - Compare two faces
- `POST /api/v1/face/analyze` - Analyze face attributes
- `POST /api/v1/face/detect` - Face detection only
- `POST /api/v1/liveness/detect` - Liveness detection
- `POST /api/v1/face/identify` - 1:N face identification
- `POST /api/v1/face/register` - Register new face
- `GET /api/v1/face/list` - List registered faces
- `GET /api/v1/health` - Health check

## Configuration

Edit `config/config.yaml` to customize:
- Model settings (thresholds, detection size)
- API settings (port, authentication)
- Database settings
- Camera/stream settings
- Performance settings

## Model Information

- **Model**: RetinaFace buffalo_l
- **Provider**: InsightFace
- **Size**: ~100MB
- **Embedding**: 512-dimensional vectors
- **Auto-download**: Models download automatically on first run

## Documentation

- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Architecture](docs/architecture.md)

## License

MIT License

## References

- [Recognito Vision](https://github.com/recognito-vision/Windows-FaceRecognition-FaceLivenessDetection-Python)
- [SharpAI DeepCamera](https://github.com/SharpAI/DeepCamera)
- [Kerberos Agent](https://github.com/kerberos-io/agent)

