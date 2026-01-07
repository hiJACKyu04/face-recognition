# Face Recognition System

A comprehensive, enterprise-grade cross-platform face recognition system with Flask API, Gradio UI, Video Surveillance, CLI interface, and Web Dashboard. Built using InsightFace with RetinaFace buffalo_l model for high-accuracy face detection, recognition, and liveness detection.

## 🎯 Key Features

- ✅ **High Accuracy**: State-of-the-art RetinaFace buffalo_l model with 99%+ recognition accuracy
- ✅ **Real-time Processing**: Optimized for real-time face detection and recognition
- ✅ **Multiple Interfaces**: REST API, Web UI, CLI, and Video Surveillance
- ✅ **Cross-platform**: Works on macOS, Windows, and Linux
- ✅ **GPU Support**: CUDA, CoreML, and Apple Silicon acceleration
- ✅ **Production Ready**: Docker support, authentication, rate limiting, and monitoring

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

## 🤖 Model Information

### RetinaFace buffalo_l

- **Model Type**: Face Detection + Recognition
- **Provider**: InsightFace
- **Model Size**: ~100MB
- **Embedding Dimension**: 512-dimensional vectors
- **Detection Accuracy**: >95%
- **Recognition Accuracy**: >99% (with proper threshold)
- **Auto-download**: Models download automatically on first run
- **Supported Platforms**: CPU, CUDA, CoreML, Apple Silicon

### Model Capabilities

- **Face Detection**: Detect multiple faces in images/videos
- **Face Recognition**: 1:1 verification and 1:N identification
- **Face Alignment**: Automatic face alignment and normalization
- **Attribute Extraction**: Age, gender, and landmark detection
- **Real-time Performance**: Optimized for real-time processing

## 🏗️ System Architecture

The system follows a modular, layered architecture:

```
┌─────────────────────────────────────────┐
│         Interface Layer                 │
│  Flask API │ Gradio UI │ CLI │ Dashboard│
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Core Engine Layer               │
│  FaceEngine │ Liveness │ Analyzer       │
│  Database │ Tracker │ StreamProcessor   │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Model & Data Layer              │
│  InsightFace │ RetinaFace │ Database    │
└─────────────────────────────────────────┘
```

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 🔒 Security Features

- **API Authentication**: API key-based authentication
- **Rate Limiting**: Configurable rate limits to prevent abuse
- **CORS Support**: Cross-origin resource sharing configuration
- **Input Validation**: Comprehensive input sanitization
- **Data Privacy**: Optional face image storage, embedding-only mode
- **Secure Transmission**: HTTPS/TLS support recommended

## 📊 Performance

### Benchmarks

- **Face Detection**: 50-100ms per image (CPU), 10-20ms (GPU)
- **Face Recognition**: 10-20ms per comparison
- **1:N Search**: 100-200ms for 1000 faces
- **Video Processing**: Real-time capable (30 FPS with frame skipping)

### Optimization

- GPU acceleration (CUDA/CoreML/Apple Silicon)
- Frame skipping for video processing
- Database indexing for fast searches
- Connection pooling
- Batch processing support

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

### Essential Reading

- **[📖 Documentation Index](docs/INDEX.md)** - Start here! Complete documentation navigation
- **[🚀 Getting Started Guide](docs/GETTING_STARTED.md)** - Installation, configuration, and first steps
- **[📋 System Overview](docs/SYSTEM_OVERVIEW.md)** - Complete system overview, capabilities, and use cases
- **[🏗️ Architecture Documentation](docs/ARCHITECTURE.md)** - Detailed system architecture, components, and design patterns

### Reference Documentation

- **[🔌 API Documentation](docs/api.md)** - Complete REST API reference with examples
- **[🚢 Deployment Guide](docs/deployment.md)** - Production deployment instructions
- **[🎥 Video Identification Guide](docs/video_identification.md)** - Video processing and identification features

### Quick Links

- **New to the system?** → Start with [Getting Started Guide](docs/GETTING_STARTED.md)
- **Want to integrate?** → See [API Documentation](docs/api.md)
- **Understanding the system?** → Read [Architecture Documentation](docs/ARCHITECTURE.md)
- **Deploying to production?** → Follow [Deployment Guide](docs/deployment.md)

## License

MIT License

## References

- [Recognito Vision](https://github.com/recognito-vision/Windows-FaceRecognition-FaceLivenessDetection-Python)
- [SharpAI DeepCamera](https://github.com/SharpAI/DeepCamera)
- [Kerberos Agent](https://github.com/kerberos-io/agent)

