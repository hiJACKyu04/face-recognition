#!/bin/bash

# Face Recognition System Setup Script

set -e

echo "Setting up Face Recognition System..."

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p logs
mkdir -p recordings
mkdir -p face_db
mkdir -p uploads
mkdir -p models

# Copy config if it doesn't exist
if [ ! -f config/config.yaml ]; then
    echo "Creating config file from example..."
    cp config/config.example.yaml config/config.yaml
    echo "Please edit config/config.yaml with your settings"
fi

# Copy .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit .env with your settings"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config/config.yaml with your settings"
echo "2. Edit .env with your environment variables"
echo "3. Run: source venv/bin/activate"
echo "4. Run: python flask/app.py (or use face-recognition start)"

