#!/bin/bash

# Download models script
# Models will be downloaded automatically on first use, but this script can pre-download them

set -e

echo "Downloading InsightFace models..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run Python script to download models
python3 << EOF
import insightface
import sys

print("Downloading RetinaFace buffalo_l model...")
try:
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model downloaded successfully!")
    print(f"Model location: ~/.insightface/models/buffalo_l/")
except Exception as e:
    print(f"Error downloading model: {e}")
    sys.exit(1)
EOF

echo "Model download complete!"

