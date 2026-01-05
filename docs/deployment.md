# Deployment Guide

## Prerequisites

- Python 3.8+
- 8GB RAM (recommended)
- Docker (optional, for containerized deployment)

## Local Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd face-recognition
```

### 2. Setup Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure

```bash
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your settings
```

### 5. Run Setup Script

```bash
bash scripts/setup.sh
```

## Docker Deployment

### Development

```bash
cd docker
docker-compose up
```

### Production

```bash
cd docker
docker-compose -f docker-compose.prod.yml up -d
```

## Running Services

### Flask API

```bash
python flask/app.py
```

Access at: http://localhost:8000

### Gradio UI

```bash
python gradio/app.py
```

Access at: http://localhost:7860

### Video Surveillance

```bash
python video_surveillance/app.py
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

Access at: http://localhost:5000

### CLI

```bash
face-recognition start
```

## Configuration

Edit `config/config.yaml` to customize:

- Model settings
- API settings
- Database settings
- Camera settings
- Event settings

## Model Download

Models are automatically downloaded on first run. To pre-download:

```bash
bash scripts/download_models.sh
```

## Troubleshooting

### Model Download Issues

If models fail to download automatically:

1. Check internet connection
2. Manually download from InsightFace repository
3. Place in `~/.insightface/models/buffalo_l/`

### GPU Support

For GPU acceleration:

1. Install CUDA (NVIDIA) or use CoreML (Apple Silicon)
2. Install appropriate ONNX Runtime:
   ```bash
   pip install onnxruntime-gpu  # For CUDA
   ```

### Database Issues

If database errors occur:

1. Check database path in config
2. Ensure write permissions
3. For PostgreSQL, ensure connection settings are correct

