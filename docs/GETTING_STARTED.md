# Getting Started Guide

This guide will help you get started with the Face Recognition System quickly.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [First Steps](#first-steps)
5. [Basic Usage](#basic-usage)
6. [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements

- **Operating System**: macOS, Windows, or Linux
- **Python**: 3.8 or higher (3.11+ recommended)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Disk Space**: 2GB minimum (for models and dependencies)
- **GPU**: Optional but recommended for better performance

### Software Dependencies

- Python 3.8+
- pip (Python package manager)
- Git (for cloning repository)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd face-recognition
```

### Step 2: Create Virtual Environment

**macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install the Package

```bash
pip install -e .
```

This installs the CLI tool `face-recognition` and makes the package importable.

### Step 5: Run Setup Script

```bash
bash scripts/setup.sh
```

This script will:
- Create necessary directories
- Copy configuration files
- Set up logging
- Download models (on first run)

---

## Configuration

### Basic Configuration

1. Copy the example configuration:
```bash
cp config/config.example.yaml config/config.yaml
```

2. Edit `config/config.yaml` with your settings:

```yaml
# API Configuration
api:
  host: 0.0.0.0
  port: 8000
  auth:
    enabled: false  # Set to true for production
    api_key: ""     # Set your API key

# Database Configuration
database:
  type: sqlite
  path: data/face_recognition.db

# Model Configuration
model:
  name: buffalo_l
  detection_threshold: 0.5
  recognition_threshold: 0.6
```

### Environment Variables

Create a `.env` file (optional):
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
```

---

## First Steps

### Step 1: Verify Installation

Check if everything is installed correctly:

```bash
face-recognition status
```

You should see system status information.

### Step 2: Start the Flask API

```bash
python flask/app.py
```

Or using the CLI:
```bash
face-recognition start
```

The API will be available at `http://localhost:8000`

### Step 3: Test the API

Open a new terminal and test the health endpoint:

```bash
curl http://localhost:8000/api/v1/health
```

You should see:
```json
{
  "service": "face-recognition-api",
  "status": "healthy"
}
```

### Step 4: Start Gradio UI (Optional)

In a new terminal:
```bash
python gradio/app.py
```

Open your browser to `http://localhost:7860`

---

## Basic Usage

### Register a Face

**Using CLI**:
```bash
face-recognition register-face --image path/to/image.jpg --name "John Doe"
```

**Using API**:
```bash
curl -X POST http://localhost:8000/api/v1/face/register \
  -F "image=@path/to/image.jpg" \
  -F "name=John Doe"
```

**Using Gradio UI**:
1. Open `http://localhost:7860`
2. Go to "Face Identification" tab
3. Upload an image
4. Enter person name
5. Click "Register Face"

### Identify a Face

**Using API**:
```bash
curl -X POST http://localhost:8000/api/v1/face/identify \
  -F "image=@path/to/unknown.jpg"
```

**Using Gradio UI**:
1. Open `http://localhost:7860`
2. Go to "Face Identification" tab
3. Upload an image
4. Click "Identify Face"

### Compare Two Faces

**Using API**:
```bash
curl -X POST http://localhost:8000/api/v1/face/compare \
  -F "image1=@path/to/image1.jpg" \
  -F "image2=@path/to/image2.jpg"
```

**Using Gradio UI**:
1. Open `http://localhost:7860`
2. Go to "Face Comparison" tab
3. Upload two images
4. Click "Compare Faces"

### Process a Video

**Using API**:
```bash
curl -X POST http://localhost:8000/api/v1/face/detect_video \
  -F "video=@path/to/video.mp4" \
  -F "identify=true" \
  -F "generate_annotated_video=true"
```

**Using Gradio UI**:
1. Open `http://localhost:7860`
2. Go to "Video Identification" tab
3. Upload a video file
4. Configure options (frame skip, identify, etc.)
5. Click "Process Video"

---

## Common Tasks

### List Registered Faces

**CLI**:
```bash
face-recognition list-faces
```

**API**:
```bash
curl http://localhost:8000/api/v1/face/list
```

### Check System Status

**CLI**:
```bash
face-recognition status
```

**API**:
```bash
curl http://localhost:8000/api/v1/admin/stats
```

### View Logs

```bash
tail -f logs/face_recognition.log
```

---

## Next Steps

### 1. Read the Documentation

- **[System Overview](SYSTEM_OVERVIEW.md)**: Complete system capabilities
- **[Architecture Documentation](ARCHITECTURE.md)**: System design and architecture
- **[API Documentation](api.md)**: Complete API reference

### 2. Configure for Production

- Enable API authentication
- Set up PostgreSQL database
- Configure HTTPS/TLS
- Set up monitoring
- Configure backups

See [Deployment Guide](deployment.md) for details.

### 3. Integrate with Your Application

- Use the REST API for integration
- Set up webhooks for events
- Configure MQTT for IoT integration
- Use the CLI for automation

### 4. Customize Configuration

- Adjust recognition thresholds
- Configure performance settings
- Set up event notifications
- Customize video processing

### 5. Advanced Features

- Set up video surveillance
- Configure multiple cameras
- Set up email/webhook notifications
- Enable video recording
- Use face tracking features

---

## Troubleshooting

### Models Not Downloading

If models don't download automatically:

```bash
python -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=0)"
```

### Import Errors

Make sure you're in the virtual environment:
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### Port Already in Use

If port 8000 is already in use, change it in `config/config.yaml`:
```yaml
api:
  port: 8001  # Change to available port
```

### Database Errors

If you see database errors:
```bash
# Remove old database and recreate
rm data/face_recognition.db
python -c "from src.database import FaceDatabase; db = FaceDatabase('data/face_recognition.db')"
```

---

## Examples

### Python Integration Example

```python
import requests

# Initialize
api_url = "http://localhost:8000/api/v1"
api_key = "your-api-key"

# Register a face
with open("person.jpg", "rb") as f:
    response = requests.post(
        f"{api_url}/face/register",
        files={"image": f},
        data={"name": "John Doe"},
        headers={"X-API-Key": api_key}
    )
    print(response.json())

# Identify a face
with open("unknown.jpg", "rb") as f:
    response = requests.post(
        f"{api_url}/face/identify",
        files={"image": f},
        headers={"X-API-Key": api_key}
    )
    print(response.json())
```

### Video Processing Example

```python
import requests

# Process video
with open("video.mp4", "rb") as f:
    response = requests.post(
        f"{api_url}/face/detect_video",
        files={"video": f},
        data={
            "identify": "true",
            "frame_skip": "30",
            "generate_annotated_video": "true"
        },
        headers={"X-API-Key": api_key}
    )
    result = response.json()
    print(f"Found {result['results']['summary']['unique_people']} people")
```

---

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Report on GitHub
- **Questions**: Check existing issues or create new one

---

## License

MIT License

